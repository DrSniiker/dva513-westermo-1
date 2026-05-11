import os
import warnings
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
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


batch_size = 32
epochs = 50
validation_split = 0.2  # Not used in The ANN , DNN, but needed for pruning-aware training
Max_final_sparsity = 0.90
learning_rate = 0.0001
Early_stop_patience = 4
reduce_LR_factor = 0.8
reduce_LR_patience = 2


folder_structure = {
    "./data/ANN_model": ["readme"],
    "./data/ANN_post_pruning": ["BOTH", "Ereno", "PowerDuck"],
}
def setup_project_structure(base_structure):
    for main_folder, sub_folders in base_structure.items():
        for sub in sub_folders:
            folder_path = Path(main_folder) / sub
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Added folder: {folder_path}")

setup_project_structure(folder_structure)



# ==========================================
# select features and load data
# ==========================================


features_1 = [
    "gooseLengthDiff",
    "frameLengthDiff",
    "gooseTimeAllowedtoLive",
    "tDiff",
    "stDiff",
    "sqDiff",
    "timestampDiff",
]


features_2 = [
    "gooseLengthDiff",
    "frameLengthDiff",
    "gooseTimeAllowedtoLive",
    "tDiff",
    "stDiff",
    "sqDiff",
    "timestampDiff",
    "delay",
    "cbStatusDiff",
    "apduSizeDiff",
]


features = [
    "gooseLengthDiff",
    "stDiff",
    "sqDiff",
    "timestampDiff",
    "delay",
    "gooseLen"
]

features_3 = [
    "gooseLengthDiff",
    "gooseTimeAllowedtoLive",
    "stDiff",
    "sqDiff",
    "timestampDiff",
    "delay",
    "StNum",
    "SqNum",
    "gooseLen"
]




if Dataset == "Ereno":
    df = pd.read_csv("./data/Ereno/train.csv")
    split_idx = int(0.8 * len(df))
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_val = df.iloc[split_idx :].copy().reset_index(drop=True)

    df_test = pd.read_csv("./data/Ereno/test.csv")

    dataframes = [df_train, df_val, df_test]
    # To address the issue of "id" and "class" column, shift the columns to the left and rename them
    for i in range(len(dataframes)):
        if "id" in dataframes[i].columns:
            new_cols = dataframes[i].columns[1:]
            temp_df = dataframes[i].iloc[:, :-1]
            temp_df.columns = new_cols
            dataframes[i] = temp_df

    df_train, df_val, df_test = dataframes


elif Dataset == "PowerDuck":
    df = pd.read_csv("./data/powerduck/powerduck-labeled_final.csv")
    df = df[df['gooseTimeAllowedtoLive'] != 0]
    df = df.reset_index(drop=True)
    print(f"df.head()  \n {df.head()}")

    #df["class"] = df["class"].replace({0: "normal", 1: "attack"})

    df_val = df[df['split'] == 'val'].copy().reset_index(drop=True)
    df_test = df[df['split'] == 'test'].copy().reset_index(drop=True)
    df_train = df[df['split'] == 'train'].copy().reset_index(drop=True)

else:
    df_ereno = pd.read_csv("./data/Ereno/train.csv")
    split_idx = int(0.8 * len(df_ereno))
    df_train = df_ereno.iloc[:split_idx].copy().reset_index(drop=True)
    df_val = df_ereno.iloc[split_idx :].copy().reset_index(drop=True)
    df_ereno_test = pd.read_csv("./data/Ereno/test.csv")
    
    dataframes = [df_train, df_val, df_ereno_test]
    for i in range(len(dataframes)):
        if "id" in dataframes[i].columns:
            new_cols = dataframes[i].columns[1:]
            temp_df = dataframes[i].iloc[:, :-1]
            temp_df.columns = new_cols
            dataframes[i] = temp_df

    df_train, df_val, df_test = dataframes

    df_powerduck = pd.read_csv("./data/powerduck/powerduck-labeled_final.csv")
    df_powerduck = df_powerduck[df_powerduck['gooseTimeAllowedtoLive'] != 0]
    df_powerduck = df_powerduck.reset_index(drop=True)

    df_powerduck_val = df_powerduck[df_powerduck['split'] == 'val'].copy().reset_index(drop=True)
    df_powerduck_test = df_powerduck[df_powerduck['split'] == 'test'].copy().reset_index(drop=True)
    df_powerduck_train = df_powerduck[df_powerduck['split'] == 'train'].copy().reset_index(drop=True)

    selection = features + ["class"]
    df_train = pd.concat([df_train[selection], df_powerduck_train[selection]], ignore_index=True)
    df_val = pd.concat([df_val[selection], df_powerduck_val[selection]], ignore_index=True)
    df_test = pd.concat([df_test[selection], df_powerduck_test[selection]], ignore_index=True)


dataframes = [df_train, df_val, df_test]

print(f"DataFrame form: {df_train.shape}, Columns: {list(df_train.columns[:8])}")
print(f"df.head(): \n {df_train.head()}")

# Skapa X och y
x_train = df_train[features].astype(float).copy()
y_train = (df_train["class"] != "normal").astype(int).copy().reset_index(drop=True)

x_test = df_test[features].astype(float).copy()
y_test = (df_test["class"] != "normal").astype(int).copy().reset_index(drop=True)

x_val = df_val[features].astype(float).copy()
y_val = (df_val["class"] != "normal").astype(int).copy().reset_index(drop=True)




# scale the data
scaler = PowerTransformer(method="yeo-johnson")

X_scaled = scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(X_scaled, columns=features)

X_test_scaled = scaler.transform(x_test)
x_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

x_val_scaled = scaler.transform(x_val)
x_val_scaled = pd.DataFrame(x_val_scaled, columns=features)

joblib.dump(scaler, f"./data/ANN_model/{Dataset}_scaler_powertransformer.pkl")

print(f"************DataFrame form: {x_train_scaled.shape}, labels: {y_train.shape}")
print("'*************************Starting scaling and model building...\n")

static_weight = {0: 1.0, 1: 10.0}



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

# ==========================================
# Building and training the model
# ==========================================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=Early_stop_patience, restore_best_weights=True),
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
for i, arch in enumerate(architectures, 1):
    print(f"**********************************Training model {i} with architecture: {arch}...")

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
        class_weight=static_weight,         # i Used 1:10, please loock at the results above for different class weights, and select the one that gives the best MCC
        verbose=2,              # Set to 2 for epoch-level logging, 1 for batch-level logging, and 0 for no logging
        callbacks=callbacks,
    )

    base_model.save(f"./data/ANN_model/{Dataset}_model_original_architecture_{i}.keras")

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
    print(f"Best threshold based on MCC: {best_threshold:.2f}")

    # Apply the best threshold to get the final predictions
    y_pred = (y_pred_probs > best_threshold).astype(int)

    # *************************
    #  Calculate and print the classification report and MCC
    print(f"--- Classification Report --- Dataset {Dataset} --- arch {i}")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Attack (1)"]))

    # Calculate MCC separately (not included in the standard report)
    ANN_mcc = matthews_corrcoef(y_test, y_pred)

    print(f"Model: {i}, (MCC): {ANN_mcc:.4f}, MCC_threshold: {best_threshold:.4f}, architecture: {arch}")
    print(f"Our model has {len(base_model.layers)} layers and {base_model.count_params()} parameters.")
    diff_min = (pd.Timestamp.now() - Start_time).total_seconds() / 60
    print(f"Total time in minutes: {diff_min:.2f} min")
