import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    matthews_corrcoef)
import os
import joblib
import warnings
from pathlib import Path
from operator import le
import time
from shared_var import features_goose, features_9, features_13


# TensorFlow use Keras 2
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# Disable oneDNN optimizations to avoid potential issues on some platforms
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Disable TensorFlow-loggar 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable warnings from libraries, some scikit-learn
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

Dataset = "Ereno"
method = None
features = None

print("=" * 55)
print(" This script only for the ereno dataset")
print(" You need to have the models that you want to test in this path: \n")
print("./data/test_model/{Dataset}/features_{len(features) \n")
print("=" * 55)

while not features:
    choice = input(
        "How many features you want to use? \n 1 for 6 features, 2 for 9 , 3 for 13 features "
    )
    if choice == "1":
        features = features_goose
    elif choice == "2":        
        features = features_9
    elif choice == "3":
        features = features_13
    else:
        print("Invalid choice, please try again.")

train_classes = ['high_StNum', 'injection', 'inverse_replay', 'masquerade_fake_fault', 'masquerade_fake_normal', 'poisoned_high_rate', 'random_replay']

binary_threshold = float(0.84)
filname = "Ereno_architecture_1_f6"
filname_binary = f"{filname}_binary"
filname_encoded = f"{filname}_encoded"

file_encoder = f"{Dataset}_encoder"


test_dir = f"./data/test_model/{Dataset}/features_{len(features)}"
scaler_binary = joblib.load(f"{test_dir}/{Dataset}_scaler_powertransformer_binary.pkl")
scaler_encoded = joblib.load(f"{test_dir}/{Dataset}_scaler_powertransformer_encoded.pkl")


folder_structure = {
    "./data/test_model": ["BOTH", "Ereno", "PowerDuck"],
}

def setup_project_structure(base_structure):
    for main_folder, sub_folders in base_structure.items():
        for sub in sub_folders:
            folder_path = Path(main_folder) / sub
            folder_path.mkdir(parents=True, exist_ok=True)

setup_project_structure(folder_structure)


# --- preparing - Only the test dataset ---
print(f"\n Preparing test dataset for: {Dataset}...")
print("=" * 55)
if Dataset == "Ereno":
    
    target_names = ['random_replay', 'inverse_replay', 'masquerade_fake_fault', 'masquerade_fake_normal', 'injection', 'high_StNum', 'poisoned_high_rate', 'normal']
    mapping_dict = {name: index for index, name in enumerate(target_names)}

    df_ereno_test = pd.read_csv("./data/Ereno/test.csv")

    if "id" in df_ereno_test.columns:
         new_cols = df_ereno_test.columns[1:]
         temp_df = df_ereno_test.iloc[:, :-1]
         temp_df.columns = new_cols
         df_ereno_test = temp_df
    
    df_ereno_test['attack_tag'] = df_ereno_test['class'].map(mapping_dict)    

    x_test_only_features = df_ereno_test[features].astype(float).copy()
    y_test_encoded = df_ereno_test[df_ereno_test['class'] != 'normal']

    y_test_binary = (df_ereno_test["class"] != "normal").astype(int).copy().reset_index(drop=True)



def compute_metrics(y_true, y_pred):
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
        "True_Positives": TP,
        "True_Negatives": TN,
        "False_Negatives": FN,
        "False_Positives": FP,
    }
    print("metrics_dict_binary:")
    print(
        f"\n--- Encoded Modell Metrics ---\n"
        f"Accuracy:  {accuracy:.4f} | MCC:       {mcc:.4f}\n"
        f"F1-Score:  {f1:.4f}       | Recall:    {recall:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Confusion Matrix -> TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}\n"
        f"----------------------"
    )
    return metrics_dict

def _clone(model):
    new_model = tf.keras.models.clone_model(model)
    new_model.set_weights(model.get_weights())
    return new_model

print("Start loading the models")
model_binary = f"{test_dir}/{filname_binary}.keras"
model_encoded = f"{test_dir}/{filname_encoded}.keras"
base_model_binary = load_model(model_binary)
base_model_encoded = load_model(model_encoded, compile=False)

print("\n All models been loaded, start scale the dataset.")
print("=" * 55)

x_test_scaled_binary = scaler_binary.transform(x_test_only_features)
x_test_scaled_encoded = scaler_encoded.transform(x_test_only_features)

print("\n Data from binary model:")
print("=" * 55)

start_time_binary = time.perf_counter()

y_probs_binary = base_model_binary.predict(x_test_scaled_binary, verbose=0).ravel()
y_pred_binary = (y_probs_binary >= binary_threshold).astype(int)
print("=" * 55)
print("\n Compute metrics for binary model: \n")
print("=" * 55)

end_time_binary = time.perf_counter()
binary_exakt_tid = (end_time_binary - start_time_binary)
binary_responstid_per_raw_ms = (binary_exakt_tid / len(y_pred_binary)) * 1000

print("=" * 55)
print(f"Time for the {Dataset}:{binary_exakt_tid} Sec, Time per raw: {binary_responstid_per_raw_ms} ms.\n")
print("=" * 55)

metrics_dict_binary = compute_metrics(y_test_binary, y_pred_binary)


##############   STEG 2 ENCODED MODEL
#Filter the predicted attacks from binary-model
mask = (y_pred_binary == 1)

df_suspicious = df_ereno_test[mask]
y_test_filtered_numbers_from_binary = df_suspicious['attack_tag']


suspicious_traffic = x_test_scaled_encoded[mask]

print("=" * 55)
print("\n Data from encoded model: \n")
print("=" * 55)

######
# Solution to print 8*8 metrics of the result of encoded model


full_y_pred = np.full(len(df_ereno_test), 7)
y_test_full_true = df_ereno_test['attack_tag'].values

if suspicious_traffic.shape[0] > 0:

    start_time_encoded = time.perf_counter()

    print(f"Found {len(suspicious_traffic)} suspicious packets. Analyzing with secondary model...")
    attack_type_preds = base_model_encoded.predict(suspicious_traffic)
    y_pred_encoded = np.argmax(attack_type_preds, axis=1)

    end_time_encoded = time.perf_counter()
    encoded_exakt_tid_ms = (end_time_encoded - start_time_encoded) * 1000
    encoded_responstid_per_raw = (encoded_exakt_tid_ms / len(y_pred_encoded))

    print(f"Time for the {Dataset}:{encoded_exakt_tid_ms}, Time per raw: {encoded_responstid_per_raw}")
    
    full_y_pred[mask] = y_pred_encoded
    report_dict = classification_report(y_test_full_true, full_y_pred, target_names=target_names, output_dict=True)
    
    print(report_dict)
    conf_matrix = confusion_matrix(y_test_full_true, full_y_pred)
    print("\n--- Confusion Matrix ---")
    print(conf_matrix)

    print("compute_metrics:")
    compute_metrics(y_test_full_true, full_y_pred)

else:
    print("No attacks in this dataset.")

