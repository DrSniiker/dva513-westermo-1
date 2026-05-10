import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    matthews_corrcoef)
import os
import joblib


# This script applies Post-pruning to a set of pre-trained ANN models
# evaluating their performance at various sparsity levels and saving the results to CSV files.

Dataset = None
method = None

while not Dataset:
    choice = input(
        "Which data do you want to use? (1 for BOTH, 2 for Ereno, 3 for PowerDuck): "
    )

    if choice == "1":
        Dataset = "BOTH"
    elif choice == "2":
        Dataset = "Ereno"
    elif choice == "3":
        Dataset = "PowerDuck"
    else:
        print("Invalid choice, please try again.")

while not method:
    choice = input(
        "Which pruning method do you want to use? \n 1 - Unstructured Magnitude-based \n 2 - Structured Magnitude-based \n 3 - Unstructured Gradient-based \n 4 - Structured Gradient-based \n Enter the number corresponding to your choice: "
    )

    if choice == "1":
        method = "Unstructured_Magnitude-based"
    elif choice == "2":
        method = "Structured_Magnitude-based"
    elif choice == "3":
        method = "Unstructured_Gradient-based"
    elif choice == "4":
        method = "Structured_Gradient-based"
    else:
        print("Invalid choice, please try again.")


output_dir = f"./data/ANN_post_pruning/{Dataset}/"
scaler = joblib.load(f"./data/ANN_model/{Dataset}_scaler_powertransformer.pkl")
threshold_csv_path = (
    f"./data/ANN_model/thresholds_{Dataset}.csv"  # CSV: filnamn, threshold
)

features = ["gooseLengthDiff", "stDiff", "sqDiff", "timestampDiff", "delay", "gooseLen"]


# --- preparing - Only the test dataset ---
print(f"Preparing test dataset for: {Dataset}...")
if Dataset == "Ereno":
    df = pd.read_csv("./data/Ereno/train.csv")
    split_idx = int(0.8 * len(df))
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_val = df.iloc[split_idx:].copy().reset_index(drop=True)

    df_test = pd.read_csv("./data/Ereno/test.csv")

    dataframes = [df_train, df_val, df_test]
    for i in range(len(dataframes)):
        if "id" in dataframes[i].columns:
            new_cols = dataframes[i].columns[1:]
            temp_df = dataframes[i].iloc[:, :-1]
            temp_df.columns = new_cols
            dataframes[i] = temp_df

    df_train, df_val, df_test = dataframes


elif Dataset == "PowerDuck":
    df = pd.read_csv("./data/powerduck/powerduck-labeled_final.csv")
    df = df[df["gooseTimeAllowedtoLive"] != 0]
    df = df.reset_index(drop=True)
    print(f"df.head()  \n {df.head()}")

    df_val = df[df["split"] == "val"].copy().reset_index(drop=True)
    df_test = df[df["split"] == "test"].copy().reset_index(drop=True)
    df_train = df[df["split"] == "train"].copy().reset_index(drop=True)


else:
    df_ereno_test = pd.read_csv("./data/Ereno/test.csv")

    if "id" in df_ereno_test.columns:
        new_cols = df_ereno_test.columns[1:]
        temp_df = df_ereno_test.iloc[:, :-1]
        temp_df.columns = new_cols
        df_ereno_test = temp_df

    df_test = df_ereno_test

    df_powerduck = pd.read_csv("./data/powerduck/powerduck-labeled_final.csv")
    df_powerduck = df_powerduck[df_powerduck["gooseTimeAllowedtoLive"] != 0]
    df_powerduck = df_powerduck.reset_index(drop=True)

    df_powerduck_test = (
        df_powerduck[df_powerduck["split"] == "test"].copy().reset_index(drop=True)
    )

    selection = features + ["class"]
    df_test = pd.concat(
        [df_test[selection], df_powerduck_test[selection]], ignore_index=True
    )


# Split features to X och y (class 0 = normal, class 1 = attack)
x_test = df_test[features].astype(float).copy()
y_test = (df_test["class"] != "normal").astype(int).copy().reset_index(drop=True)


# A separate function to compute all metrics at once to avoid DivisionByZeroError issues
def compute_metrics(y_true, y_pred, loss=None):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (TP + TN) / len(y_true)
    mcc = matthews_corrcoef(y_true, y_pred)

    metrics_dict = {
        "mcc": mcc,
        "f1_score": f1,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "loss": loss,
        "True_Positives": TP,
        "True_Negatives": TN,
        "False_Negatives": FN,
        "False_Positives": FP,
    }

    return metrics_dict


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load the CSV with name and thresholds for each model
df_meta = pd.read_csv(threshold_csv_path)


def _clone(model):
    new_model = tf.keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())
    return new_model


# ── Unstructured (weight-level) ──────────────────────────────────────────────

def post_unstructured_magnitude(model, target_sparsity):

    """Zero-out the lowest-magnitude individual weights."""
    if target_sparsity == 0:
        return model

    new_model = _clone(model)
    new_model.set_weights(model.get_weights())
    all_weights = tf.concat([
            tf.reshape(tf.abs(layer.kernel), [-1])
            for layer in new_model.layers
            if isinstance(layer, tf.keras.layers.Dense)
        ], axis=0)
    threshold = np.percentile(
            all_weights.numpy(),
            target_sparsity * 100
        )

    for layer in new_model.layers:
        if not isinstance(layer, tf.keras.layers.Dense):
            continue

        mask = tf.cast(
            tf.abs(layer.kernel) > threshold,
            layer.kernel.dtype
        )

        layer.kernel.assign(layer.kernel * mask)



    return new_model


def post_unstructured_gradient(model, target_sparsity):

    print("post_unstructured_gradient is not implemented yet.")
    
    return model

# ── Structured (neuron-level) ────────────────────────────────────────────────


def post_structured_magnitude(model, target_sparsity):
  
    print("post_structured_magnitude is not implemented yet.")

    return model


def post_structured_gradient(model, target_sparsity):

    print("post_structured_gradient is not implemented yet.")

    return model


all_results = []

# --- Main Loop ---
for index, row in df_meta.iterrows():
    model_path = f"./data/ANN_model/{row['filnamn']}.keras"
    current_threshold = row["threshold"]

    print(f" \n --- model_path: {model_path}, current_threshold: {current_threshold}")
    if not os.path.exists(model_path):
        print(f"Skippar: {model_path} (Filen hittades inte)")
        continue

    print(
        f"Applying {method} for: {os.path.basename(model_path)}, Using threshold: {current_threshold:.4f} ---"
    )

    base_model = load_model(model_path)
    x_test_scaled = scaler.transform(x_test)

    # Loop through sparsity levels from 30% to 55% with a step of 5%
    for sparsity in np.arange(0.30, 0.60, 0.05):
        sparsity_pct = int(sparsity * 100)

        print(f"Sparsity: {sparsity:.2%}", end="\r")

        # Apply Pruning
        if method == "Unstructured_Magnitude-based":
            pruned_model = post_unstructured_magnitude(base_model, sparsity)
        elif method == "Structured_Magnitude-based":
            pruned_model = post_structured_magnitude(base_model, sparsity)
        elif method == "Unstructured_Gradient-based":
            pruned_model = post_unstructured_gradient(base_model, sparsity)
        elif method == "Structured_Gradient-based":
            pruned_model = post_structured_gradient(base_model, sparsity)
        else:
            print(f"Unknown pruning method: {method}")
            continue

        y_probs = pruned_model.predict(x_test_scaled, verbose=0).ravel()

        y_pred = (y_probs >= current_threshold).astype(int)

        # Save the pruned model
        file_name_only = os.path.basename(model_path).replace(".keras", "")
        save_path = f"{output_dir}{file_name_only}_pruned_{sparsity_pct}.keras"
        pruned_model.save(save_path)

        # Compute and store the results for this sparsity level
        metrics_dict = compute_metrics(y_test, y_pred)
        metrics_dict["Dataset"] = Dataset
        metrics_dict["architecture"] = file_name_only
        metrics_dict["sparsity"] = sparsity
        metrics_dict["threshold"] = current_threshold
        metrics_dict["Model_layer"] = len(pruned_model.layers)
        metrics_dict["ModelModel_parametrar_layer"] = pruned_model.count_params()
        all_results.append(metrics_dict)

    # At the end of each model, save the results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_csv_path = f"{output_dir}{Dataset}_{method}_results.csv"
    results_df.to_csv(results_csv_path, index=False, sep=",", encoding="utf-8")
