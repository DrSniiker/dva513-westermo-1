import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier
import joblib
from pathlib import Path


# pip install scikit-learn


# TensorFlow use Keras 2
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# Disable oneDNN optimizations to avoid potential issues on some platforms
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Disable TensorFlow-loggar 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable warnings from libraries, some scikit-learn
warnings.filterwarnings('ignore')



# ==========================================
# Preparing the folder structure and global variables
# ==========================================

Dataset = None
# Ereno or PowerDuck , else to use the BOTH datasets

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


features_goose = [
    "gooseLengthDiff",
    "stDiff",
    "sqDiff",
    "timestampDiff",
    "delay",
    "gooseLen",
]
features_extra = [
    "gooseLengthDiff",
    "stDiff",
    "sqDiff",
    "timestampDiff",
    "delay",
    "gooseLen",
    "extra1",
    "extra2",
    "extra3"
]

features = features_extra
batch_size = 256
epochs = 100
learning_rate = 0.001
Early_stop_patience = 10
reduce_LR_factor = 0.75
reduce_LR_patience = 4

if Dataset == "Ereno":
    static_weight = {0: 1.0, 1: 10.0}
elif Dataset == "PowerDuck":
    static_weight = {0: 1.0, 1: 10.0}
else:
    static_weight = {0: 1.0, 1: 10.0}



output_dir = f"./data/ANN_model/features_{len(features)}/{Dataset}/"

folder_structure = {
    f"./data/ANN_model/features_{len(features)}": ["BOTH", "Ereno", "PowerDuck"],
    "./data/ANN_multiclass": ["BOTH", "Ereno", "PowerDuck"],
}
def setup_project_structure(base_structure):
    for main_folder, sub_folders in base_structure.items():
        for sub in sub_folders:
            folder_path = Path(main_folder) / sub
            folder_path.mkdir(parents=True, exist_ok=True)

setup_project_structure(folder_structure)



# ==========================================
# Select extra features and load data
# ==========================================


rename_map_extra_features = {
    "isbARmsValue":  "extra1",
    "data__SIP1_Abgang1MEAS/LLN0$Meas_SIP1-000":  "extra1",
    "iisbCRmsValue": "extra2",
    "data__SIP1_Abgang1PROT/LLN0$UMZ1_Fak-000":  "extra2",
    "vsbCRmsValue":  "extra3",
    "data__SIP3_Abgang3MEAS/LLN0$Meas-000":  "extra3"
}


if Dataset == "Ereno":
    df_ereno = pd.read_csv("./data/Ereno/train.csv")
    df_ereno_test = pd.read_csv("./data/Ereno/test.csv")
    
    df_ereno = df_ereno.rename(columns=rename_map_extra_features)
    df_ereno_test = df_ereno_test.rename(columns=rename_map_extra_features)


    dataframes = [df_ereno, df_ereno_test]
    # To address the issue of "id" and "class" column, shift the columns to the left and rename them
    for i in range(len(dataframes)):
        if "id" in dataframes[i].columns:
            new_cols = dataframes[i].columns[1:]
            temp_df = dataframes[i].iloc[:, :-1]
            temp_df.columns = new_cols
            dataframes[i] = temp_df

    df_ereno, df_ereno_test = dataframes
    print("df_head", df_ereno.head())
    print(df_ereno['class'].unique())

    df_ereno['split'] = None

    train_idx, val_idx = train_test_split(
        df_ereno.index, 
        test_size=0.20, 
        stratify=df_ereno['class'],
        random_state=42
    )

    df_ereno.loc[train_idx, 'split'] = 'train'
    df_ereno.loc[val_idx, 'split'] = 'val'


    df_val = df_ereno[df_ereno['split'] == 'val'].copy().reset_index(drop=True)
    df_train = df_ereno[df_ereno['split'] == 'train'].copy().reset_index(drop=True)
    df_test = df_ereno_test.copy().reset_index(drop=True)


elif Dataset == "PowerDuck":
    df = pd.read_csv("./data/powerduck/powerduck-labeled_final.csv")
    df = df[df['gooseTimeAllowedtoLive'] != 0]
    df = df.reset_index(drop=True)

    df = df.rename(columns=rename_map_extra_features)


    print(f"df.head()  \n {df.head()}")
    print("df class",df['class'].unique())
    print("df attack_tag", df['attack_tag'].unique())
    df['class'] = df['attack_tag']

    df.drop(columns=['split'], inplace=True)
    df['split'] = None

    train_idx, temp_idx = train_test_split(
        df.index, 
        test_size=0.25, 
        stratify=df['class'], 
        random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=0.5, 
        stratify=df.loc[temp_idx, 'class'], 
        random_state=42
    )
    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'

    df_val = df[df['split'] == 'val'].copy().reset_index(drop=True)
    df_test = df[df['split'] == 'test'].copy().reset_index(drop=True)
    df_train = df[df['split'] == 'train'].copy().reset_index(drop=True)

else:
    df_ereno = pd.read_csv("./data/Ereno/train.csv")
    df_ereno_test = pd.read_csv("./data/Ereno/test.csv")
    
    df_ereno = df_ereno.rename(columns=rename_map_extra_features)
    df_ereno_test = df_ereno_test.rename(columns=rename_map_extra_features)


    dataframes = [df_ereno, df_ereno_test]
    # To address the issue of "id" and "class" column, shift the columns to the left and rename them
    for i in range(len(dataframes)):
        if "id" in dataframes[i].columns:
            new_cols = dataframes[i].columns[1:]
            temp_df = dataframes[i].iloc[:, :-1]
            temp_df.columns = new_cols
            dataframes[i] = temp_df

    df_ereno, df_ereno_test = dataframes
    print("df_head", df_ereno.head())
    print(df_ereno['class'].unique())

    df_ereno['split'] = None

    train_idx, val_idx = train_test_split(
        df_ereno.index, 
        test_size=0.20, 
        stratify=df_ereno['class'],
        random_state=42
    )

    df_ereno.loc[train_idx, 'split'] = 'train'
    df_ereno.loc[val_idx, 'split'] = 'val'

    df_ereno_val = df_ereno[df_ereno['split'] == 'val'].copy().reset_index(drop=True)
    df_ereno_train = df_ereno[df_ereno['split'] == 'train'].copy().reset_index(drop=True)
    df_ereno_test = df_ereno_test.copy().reset_index(drop=True)


    df_powerduck = pd.read_csv("./data/powerduck/powerduck-labeled_final.csv")
    df_powerduck = df_powerduck[df_powerduck['gooseTimeAllowedtoLive'] != 0]
    df_powerduck = df_powerduck.reset_index(drop=True)
    df_powerduck.drop(columns=['split'], inplace=True)
    df_powerduck['split'] = None

    train_idx, temp_idx = train_test_split(
        df_powerduck.index, 
        test_size=0.25, 
        stratify=df_powerduck['class'], 
        random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=0.5, 
        stratify=df_powerduck.loc[temp_idx, 'class'], 
        random_state=42
    )
    df_powerduck.loc[train_idx, 'split'] = 'train'
    df_powerduck.loc[val_idx, 'split'] = 'val'
    df_powerduck.loc[test_idx, 'split'] = 'test'

    df_powerduck_val = df_powerduck[df_powerduck['split'] == 'val'].copy().reset_index(drop=True)
    df_powerduck_test = df_powerduck[df_powerduck['split'] == 'test'].copy().reset_index(drop=True)
    df_powerduck_train = df_powerduck[df_powerduck['split'] == 'train'].copy().reset_index(drop=True)

    selection = features + ["class"]
    df_train = pd.concat([df_ereno_train[selection], df_powerduck_train[selection]], ignore_index=True)
    df_val = pd.concat([df_ereno_val[selection], df_powerduck_val[selection]], ignore_index=True)
    df_test = pd.concat([df_ereno_test[selection], df_powerduck_test[selection]], ignore_index=True)


dataframes = [df_train, df_val, df_test]

print(f"df_train DataFrame form: {df_train.shape}, Columns: {list(df_train.columns[:8])}")
print(f"df_val DataFrame form: {df_val.shape}, Columns: {list(df_val.columns[:8])}")
print(f"df_test DataFrame form: {df_test.shape}, Columns: {list(df_test.columns[:8])}")

# X och y
x_train = df_train[features].astype(float).copy()
y_train = (df_train["class"] != "normal").astype(int).copy().reset_index(drop=True)

x_test = df_test[features].astype(float).copy()
y_test = (df_test["class"] != "normal").astype(int).copy().reset_index(drop=True)

x_val = df_val[features].astype(float).copy()
y_val = (df_val["class"] != "normal").astype(int).copy().reset_index(drop=True)



print("'*************************Starting scaling and model building...\n")

# scale the data
scaler = PowerTransformer(method="yeo-johnson")
#scaler = RobustScaler()

X_scaled = scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(X_scaled, columns=features)

X_test_scaled = scaler.transform(x_test)
x_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

x_val_scaled = scaler.transform(x_val)
x_val_scaled = pd.DataFrame(x_val_scaled, columns=features)

joblib.dump(scaler, f"{output_dir}/{Dataset}_scaler_powertransformer.pkl")

print(f"************DataFrame form x_train_scaled: {x_train_scaled.shape}, labels: {y_train.shape}")
print(f"x_train_scaled df.head(): \n {x_train_scaled.head()}")





# ==========================================
# Define the model architecture and training loop
# ==========================================

def create_model(input_dim, architecture):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    dense_count = 1
    for layer in architecture:
        if isinstance(layer, int):  # if it's an integer, it's the number of units for a Dense layer
            model.add(Dense(layer, activation="relu", name=f"dense_{dense_count}"))
            dense_count += 1
        elif layer == "batch": # if it's the string "batch", it's a BatchNormalization layer
            model.add(BatchNormalization())
        elif isinstance(layer, float): # if it's a float, it's a dropout rate
            model.add(Dropout(layer))
            
    model.add(Dense(1, activation="sigmoid", name="output"))
    return model


'''
print("**************DummyClassifier baseline...")
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(x_train_scaled, y_train)
baseline_DummyClassifier_pred = dummy_clf.predict(x_test_scaled)
baseline_mcc = matthews_corrcoef(y_test, baseline_DummyClassifier_pred)
f1_score_baseline_DummyClassifier = f1_score(y_test, baseline_DummyClassifier_pred)
accuracy_baseline_DummyClassifier = dummy_clf.score(x_test_scaled, y_test)
recall_baseline_DummyClassifier = recall_score(y_test, baseline_DummyClassifier_pred)
precision_baseline_DummyClassifier = precision_score(y_test, baseline_DummyClassifier_pred)
print(f"Baseline DummyClassifier MCC: {baseline_mcc:.4f}, F1-Score: {f1_score_baseline_DummyClassifier:.4f}, Accuracy: {accuracy_baseline_DummyClassifier:.4f}, Recall: {recall_baseline_DummyClassifier:.4f}, Precision: {precision_baseline_DummyClassifier:.4f}")
'''

# ==========================================
# Building and training the model
# ==========================================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=Early_stop_patience, restore_best_weights=True, start_from_epoch=20),
    ReduceLROnPlateau(monitor="val_loss", factor=reduce_LR_factor, patience=reduce_LR_patience, min_lr=1e-6),
]


architecture_1 = [32, 18]
architecture_2 = [64, 32, 18]
architecture_3 = [64, "batch", 32, 0.2, 18]
architecture_4 = [128, 64, 32, 16]
architecture_5 = [128, "batch", 0.2, 64, 32, 0.2, 16]
architecture_6 = [128, 128, 64, 64, 32]
architecture_7 = [128, "batch", 0.2, 128, 64, "batch", 64, 0.2, 32]
architecture_8 = [128, 128, 64, 64, 32, 16]
architecture_9 = [128, "batch", 0.2, 128, 64, "batch", 64, 32, 0.2, 16]
architecture_10 = [256, 128, 256, 128, 128, 64, 32, 16, 8]
architecture_11 = [256, "batch", 128, 0.2, 256, "batch", 128, 0.2, 128,  64, 32, 16, "batch", 0.2, 8]


architectures = [architecture_1, architecture_2, architecture_3 , architecture_4, architecture_5, architecture_6, architecture_7, architecture_8, architecture_9, architecture_10, architecture_11]
architectures_without_batch = [arch for arch in architectures if "batch" not in str(arch)]
architectures_with_batch = [arch for arch in architectures if "batch" in str(arch)]

print(f"***************************Dataset: {Dataset} - will train {len(architectures)} models with different architectures...")
print(f"Architectures: {architectures}")
print(f"features: {features}")
# To save the result output
results_list = []
save_threshold = []

for i, arch in enumerate(architectures, 1):
    print(f" \n **********************************Training model {i} with architecture: {arch}...")

    Start_time = pd.Timestamp.now() # Start time for training the model, to calculate total time taken at the end of this architecture

    base_model = create_model(len(features), architecture=arch)
    base_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    print(f"learning_rate:{learning_rate}, batch_size:{batch_size}")
    base_model.fit(
        x_train_scaled,
        y_train,
        epochs=epochs,
        validation_data=(x_val_scaled, y_val),
        class_weight=static_weight,         # For ereno 1:10 work well, need to test for powerduck
        verbose=2,              # Set to 2 for epoch-level logging, 1 for batch-level logging, and 0 for no logging
        callbacks=callbacks,
    )

    base_model.save(f"{output_dir}/{Dataset}_model_original_architecture_{i}.keras")

    print(f"***************{Dataset} model architecture{i} saved.")
    print("*************** Calculating y_pred_probs...")
    y_pred_probs = base_model.predict(x_test_scaled, verbose=0)

    # *************************

    # y_pred = (y_pred_probs > 0.5).astype(int)

    # Test different thresholds to find the one that gives the best MCC
    thresholds = np.linspace(0.1, 0.9, 81)
    mcc_scores = [
        matthews_corrcoef(y_test, (y_pred_probs > t).astype(int)) for t in thresholds
    ]

    # Find the best threshold based on MCC
    best_threshold = thresholds[np.argmax(mcc_scores)]
    save_threshold.append({
            "Dataset":      Dataset,
            "Architecture": i,
            "filnamn": f"{Dataset}_model_original_architecture_{i}",
            "threshold": best_threshold
    })

    # Apply the best threshold to get the final predictions
    y_pred = (y_pred_probs > best_threshold).astype(int)

    # *************************
    # Calculate and print the classification report and MCC
    # Calculate MCC separately (not included in the standard report)
    ANN_mcc = matthews_corrcoef(y_test, y_pred)


    orig_labels = df_test["class"].values   # pull once, not per row
    y_test_arr = np.array(y_test).ravel()   # (n,1) → (n,)
    y_pred_arr = np.array(y_pred).ravel()   # (n,1) → (n,)


    tp_mask = (y_test_arr == 1) & (y_pred_arr == 1)
    tn_mask = (y_test_arr == 0) & (y_pred_arr == 0)
    fp_mask = (y_test_arr == 0) & (y_pred_arr == 1)
    fn_mask = (y_test_arr == 1) & (y_pred_arr == 0)

    conf_df = pd.DataFrame({
        "label": orig_labels,
        "TP": tp_mask.astype(int),
        "TN": tn_mask.astype(int),
        "FP": fp_mask.astype(int),
        "FN": fn_mask.astype(int),
    })
    type_counts_df = conf_df.groupby("label")[["TP", "TN", "FP", "FN"]].sum()

    print(f"\n{'='*55}")
    print(f"  Dataset: {Dataset}  |  Architecture: {i}")
    print(f"{'='*55}")

    print("\n--- Binary Confusion Matrix (Normal=0, Attack=1) ---")
    cm = confusion_matrix(y_test_arr, y_pred_arr)
    tn, fp, fn, tp = cm.ravel()
    print(f"  TP: {tp:>7}   FP: {fp:>7}")
    print(f"  FN: {fn:>7}   TN: {tn:>7}")

    print("\n--- Binary Classification Report ---")
    print(classification_report(y_test_arr, y_pred_arr,
                                target_names=["Normal", "Attack"],
                                digits=4))
    print(f"  Global MCC (threshold={best_threshold:.4f}): {ANN_mcc:.4f}")

    print("\n--- Per Attack-Type Breakdown ---")
    print(f"{'Type':<22} {'TP':>7} {'TN':>7} {'FP':>7} {'FN':>7}  {'Metric':>9}")
    print("-" * 60)

    total_samples = len(y_test_arr)
    for label, row in type_counts_df.sort_index().iterrows():
        tp_, tn_, fp_, fn_ = int(row.TP), int(row.TN), int(row.FP), int(row.FN)
        total_positive = tp_ + fn_

        if total_positive > 0:                          # Attack class → show Recall
            metric_val  = tp_ / total_positive
            metric_name = "Recall"
        else:                                           # Normal class → show TNR
            metric_val  = tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0
            metric_name = "TNR"


        correct = tp_ + (total_samples - tp_ - fp_ - fn_)   # TP + true negatives globally
        acc_    = correct / total_samples

        total = tp_ + tn_ + fp_ + fn_
        acc_  = (tp_ + tn_) / total if total > 0 else 0

        print(f"  {label:<20} {tp_:>7} {tn_:>7} {fp_:>7} {fn_:>7}"
            f"  {metric_name}={metric_val:.1%}  Archi-MCC={ANN_mcc:.3f}  Acc={acc_:.3f}")

        results_list.append({
            "Dataset":      Dataset,
            "Architecture": i,
            "Class":        label,
            "TP": tp_, "TN": tn_, "FP": fp_, "FN": fn_,
            "Recall":   metric_val if total_positive > 0 else None,
            "TNR":      metric_val if total_positive == 0 else None,
            "Archi-MCC":      ANN_mcc,
            "Accuracy": acc_,
        })

    # Architecture model summary 
    print(f"\n  Model layers : {len(base_model.layers)}")
    print(f"  Parameters   : {base_model.count_params():,}")
    diff_min = (pd.Timestamp.now() - Start_time).total_seconds() / 60
    print(f"  Time elapsed : {diff_min:.2f} min")
    print("=" * 55)

# Save results of all architectures
df_results     = pd.DataFrame(results_list)
thresholds_file = pd.DataFrame(save_threshold)

df_results.to_csv(     f"{output_dir}/results_{Dataset}.csv",    index=False)
thresholds_file.to_csv(f"{output_dir}/thresholds_{Dataset}.csv", index=False)
print(f"\n  Saved {output_dir}/results_{Dataset}.csv")
print(f"  Saved {output_dir}/thresholds_{Dataset}.csv")
