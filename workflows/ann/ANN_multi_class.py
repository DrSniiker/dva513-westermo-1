from operator import le
import os
from tokenize import group
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import tensorflow as tf

import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import random

from shared_var import features_goose, features_9, features_13

# TensorFlow use Keras 2
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# Disable oneDNN optimizations to avoid potential issues on some platforms
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Disable TensorFlow-loggar 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable warnings from libraries, some scikit-learn
warnings.filterwarnings('ignore')

os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
tf.config.experimental.enable_op_determinism()

# ==========================================
# Preparing the folder structure and global variables
# ==========================================
USE_SMOTE = False
Dataset = None
DONT_USE_NORMAL_IN_TRAIN = False
# Ereno or PowerDuck , else to use the BOTH datasets
features = None


while not Dataset:
    choice = input(
        "Which data do you want to use? (2 for Ereno): "
    )
    if choice == "2":
        Dataset = "Ereno"
        features = None
    else:
        print("Invalid choice, please try again.")

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

choice = input("Do you want to use SMOTE for oversampling the minority classes and undersampling the majority classes? (y/n): ")
if choice.lower() == "y":
    USE_SMOTE = True
    print("SMOTE will be used for oversampling the minority classes.")
else:
    print("SMOTE will NOT be used. The original class distribution will be maintained.")

if not USE_SMOTE:
    compute_class_weight_boolean = False
    choice = input("Do you want to compute class weights (y)? or use hardcoded values (n): ")
    if choice.lower() == "y":
        compute_class_weight_boolean = True
    else:
        print("Class weights will NOT be computed. The original class distribution will be maintained.")

choice = input("Do you want to train with normal/attcks (y) or (Works with Ereno only ) only with attacks? (y/n): ")
if choice.lower() == "y":
    DONT_USE_NORMAL_IN_TRAIN = False
else:
    DONT_USE_NORMAL_IN_TRAIN = True



batch_size = 32  # 32, 256 or 2048
epochs = 100
learning_rate = 0.001
Early_stop_patience = 10
reduce_LR_factor = 0.75
reduce_LR_patience = 4
gamma_focal_loss = 3.0


folder_structure = {
    f"./data/ANN_multiclass/features_{len(features)}": ["BOTH", "Ereno", "PowerDuck"],
}
output_dir = f"./data/ANN_multiclass//features_{len(features)}/{Dataset}/"

def setup_project_structure(base_structure):
    for main_folder, sub_folders in base_structure.items():
        for sub in sub_folders:
            folder_path = Path(main_folder) / sub
            folder_path.mkdir(parents=True, exist_ok=True)

setup_project_structure(folder_structure)



# ==========================================
# select features and load data
# ==========================================



le = LabelEncoder()

if Dataset == "Ereno":
    ANN_static_weight = {
    0: 1,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 1
    }
    df_ereno = pd.read_csv("./data/Ereno/train.csv")
    df_ereno_test = pd.read_csv("./data/Ereno/test.csv")

    dataframes = [df_ereno, df_ereno_test]
    # To address the issue of "id" and "class" column, shift the columns to the left and rename them
    for i in range(len(dataframes)):
        if "id" in dataframes[i].columns:
            new_cols = dataframes[i].columns[1:]
            temp_df = dataframes[i].iloc[:, :-1]
            temp_df.columns = new_cols
            dataframes[i] = temp_df

    df_ereno, df_ereno_test = dataframes
    
    print(df_ereno['class'].unique())

    if DONT_USE_NORMAL_IN_TRAIN:
        target_only_attacks = ['random_replay', 'inverse_replay', 'masquerade_fake_fault', 'masquerade_fake_normal', 'injection', 'high_StNum', 'poisoned_high_rate']
        df_ereno = df_ereno[df_ereno['class'] != 'normal']
        df_ereno_test = df_ereno_test[df_ereno_test['class'] != 'normal']

    mapping_dict_attacks = {name: index for index, name in enumerate(target_only_attacks)}

    """    le.fit(all_possible_classes)
    joblib.dump(le, f"{output_dir}/{Dataset}_encoder.pkl") 

    df_ereno['attack_tag'] = le.transform(df_ereno['class'])
    df_ereno_test['attack_tag'] = le.transform(df_ereno_test['class'])
    """
    df_ereno['attack_tag'] = df_ereno['class'].map(mapping_dict_attacks)
    df_ereno_test['attack_tag'] = df_ereno_test['class'].map(mapping_dict_attacks)
    df_ereno.drop(columns=['class'], inplace=True)

    df_ereno['split'] = None

    train_idx, val_idx = train_test_split(
        df_ereno.index, 
        test_size=0.20, 
        stratify=df_ereno['attack_tag'],
        random_state=42
    )

    df_ereno.loc[train_idx, 'split'] = 'train'
    df_ereno.loc[val_idx, 'split'] = 'val'


    df_val = df_ereno[df_ereno['split'] == 'val'].copy().reset_index(drop=True)
    df_train = df_ereno[df_ereno['split'] == 'train'].copy().reset_index(drop=True)
    df_test = df_ereno_test.copy().reset_index(drop=True)


elif Dataset == "PowerDuck":
    ANN_static_weight = {
    0: 1,
    1: 100,
    2: 10,
    3: 5000,
    4: 500}
    df = pd.read_csv("./data/powerduck/powerduck-labeled_final.csv")
    df = df[df['gooseTimeAllowedtoLive'] != 0]
    df = df.reset_index(drop=True)
    
    df.loc[df['attack_tag'] == '01-replay-opening-switch-isolated','attack_tag'] = '02-replay-opening-switch-w-context'

    print(f"df.head()  \n {df.head()}")
    print(df['class'].unique())
    print(df['attack_tag'].unique())
    print(df['attack_tag'].unique())

    le.fit(df['attack_tag'])
    df['attack_tag'] = le.transform(df['attack_tag'])

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
    ANN_static_weight = {
    0: 10,
    1: 100,
    2: 100,
    3: 1,
    4: 10}
    df_ereno = pd.read_csv("./data/Ereno/train.csv")
    df_ereno_test = pd.read_csv("./data/Ereno/test.csv")

    dataframes = [df_ereno, df_ereno_test]
    # To address the issue of "id" and "class" column, shift the columns to the left and rename them
    for i in range(len(dataframes)):
        if "id" in dataframes[i].columns:
            new_cols = dataframes[i].columns[1:]
            temp_df = dataframes[i].iloc[:, :-1]
            temp_df.columns = new_cols
            dataframes[i] = temp_df

    df_ereno, df_ereno_test = dataframes
    target_names = ['replay', 'masquerade', 'normal', 'injection', 'flood']

    mapping = {
        'random_replay': 'replay',
        'inverse_replay': 'replay',
        'masquerade_fake_fault': 'masquerade',
        'masquerade_fake_normal': 'masquerade',
        'normal': 'normal',
        'injection': 'injection',
        'high_StNum':'flood',
        'poisoned_high_rate':'flood'
    }

    print(df_ereno['class'].unique())

    df_ereno['attack_tag'] = df_ereno['class'].replace(mapping)
    le.fit(df_ereno['attack_tag'])

    df_ereno['attack_tag'] = le.transform(df_ereno['attack_tag'])
    df_ereno.drop(columns=['class'], inplace=True)

    df_ereno['split'] = None

    train_idx, temp_idx = train_test_split(
        df_ereno.index, 
        test_size=0.25, 
        stratify=df_ereno['attack_tag'], 
        random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=0.5, 
        stratify=df_ereno.loc[temp_idx, 'attack_tag'], 
        random_state=42
    )

    df_ereno.loc[train_idx, 'split'] = 'train'
    df_ereno.loc[val_idx, 'split'] = 'val'
    df_ereno.loc[test_idx, 'split'] = 'test'


    df_ereno_val = df_ereno[df_ereno['split'] == 'val'].copy().reset_index(drop=True)
    df_ereno_test = df_ereno[df_ereno['split'] == 'test'].copy().reset_index(drop=True)
    df_ereno_train = df_ereno[df_ereno['split'] == 'train'].copy().reset_index(drop=True)

    df = pd.read_csv("./data/powerduck/powerduck-labeled_final_75.csv")
    df = df[df['gooseTimeAllowedtoLive'] != 0]
    df = df.reset_index(drop=True)
    
    print(f"df.head()  \n {df.head()}")
    print(df['class'].unique())
    print(df['attack_tag'].unique())
    target_names = ['flood', 'injection', 'normal', 'replay', 'supply']
    mapping = {
        '01-replay-opening-switch-isolated': 'replay',
        '02-replay-opening-switch-w-context': 'replay',
        '03-replay-old-measurements': 'replay',
        '04-insert-fake-open-w-intermediate': 'injection',
        '05-insert-fake-open-only-end': 'injection',
        '06-insert-distort-meas-up-grad': 'injection',
        '07-insert-distort-meas-down-grad': 'injection',
        '08-insert-distort-meas-up-sharp': 'injection',
        '09-insert-distort-meas-down-sharp': 'injection',
        '10-sup-1-1-tbv0': 'supply',
        '11-sup-1-1-tbv1': 'supply',
        '12-sup-1-1-tbv2': 'supply',
        '13-sup-2': 'supply',
        '14-sup-1': 'supply',
        '15-flood-repeat': 'flood',
        '16-flood-bloat-repeat': 'flood'
    }
    df['attack_tag'] = df['attack_tag'].replace(mapping)
    print(df['attack_tag'].unique())

    le.fit(df['attack_tag'])
    df['attack_tag'] = le.transform(df['attack_tag'])

    #df["class"] = df["class"].replace({0: "normal", 1: "attack"})

    df.drop(columns=['split'], inplace=True)
    df['strat_key'] = df['source_file'].astype(str) + "_" + df['attack_tag'].astype(str)
    
    df['split'] = None
    train_idx, temp_idx = train_test_split(
        df.index, 
        test_size=0.25, 
        stratify=df['strat_key'], 
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=0.5, 
        stratify=df.loc[temp_idx, 'strat_key'], 
        random_state=42
    )

    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'

    df.drop(columns=['strat_key'], inplace=True)

    df_powerduck_val = df[df['split'] == 'val'].copy().reset_index(drop=True)
    df_powerduck_test = df[df['split'] == 'test'].copy().reset_index(drop=True)
    df_powerduck_train = df[df['split'] == 'train'].copy().reset_index(drop=True)


    selection = features + ["attack_tag"]
    df_train = pd.concat([df_ereno_train[selection], df_powerduck_train[selection]], ignore_index=True)
    df_val = pd.concat([df_ereno_val[selection], df_powerduck_val[selection]], ignore_index=True)
    df_test = pd.concat([df_ereno_test[selection], df_powerduck_test[selection]], ignore_index=True)


print("##################Finished loading data...")

print("start printing unique attack tags in each split...")
print("df_val['attack_tag'].unique():")
print(df_val['attack_tag'].unique())
print(df_val['attack_tag'].value_counts())

print("df_test['attack_tag'].unique():")
print(df_test['attack_tag'].unique())
print(df_test['attack_tag'].value_counts())

print("df_train['attack_tag'].unique():")
print(df_train['attack_tag'].unique())
print(df_train['attack_tag'].value_counts())

if USE_SMOTE:
    under = RandomUnderSampler(
    sampling_strategy={3: 200000},  # normal
    random_state=42)

    smote = SMOTE(random_state=42)

    pipeline = Pipeline([
        ('under', under),
        ('smote', smote)
    ])

    X_resampled, y_resampled = pipeline.fit_resample(df_train[features], df_train['attack_tag'])

    df_train = None
    df_train = pd.DataFrame(X_resampled, columns=features)
    df_train['attack_tag'] = y_resampled

   
    print("After applying SMOTE and undersampling:")
    print("df_train['attack_tag'].unique():")
    print(df_train['attack_tag'].unique())
    print("df_train" ,df_train['attack_tag'].value_counts())
    print("df_val" ,df_val['attack_tag'].value_counts())
    print("df_test" ,df_test['attack_tag'].value_counts())

    class_weights = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1
}

if not USE_SMOTE:
    if compute_class_weight_boolean:
        classes = np.unique(df_train['attack_tag'])

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=df_train['attack_tag']
        )

        class_weights = dict(zip(classes, weights))

        print(f"computed Both Class weights: {class_weights}")
    else:
        class_weights = ANN_static_weight
        print(f"Using hardcoded class weights: {class_weights}")


print("finished printing unique attack tags in each split...")
    

for i, class_name in enumerate(target_only_attacks):
    if i not in class_weights:
        print(f"Index {i} är {class_name}, weight: Not have.")
    else:
        print(f"Index {i} är {class_name}, weight: {class_weights[i]}")

#dataframes = [df_train, df_val, df_test]

print(f"DataFrame form: {df_train.shape}")
print(f"df.head(): \n {df_train.head()}")

# X och y
x_train = df_train[features].astype(float).copy()
y_train = df_train["attack_tag"]
y_train_binary = (y_train != 'normal').astype(int)

x_test = df_test[features].astype(float).copy()
y_test = df_test["attack_tag"]
y_test_binary = (y_test != 'normal').astype(int)

x_val = df_val[features].astype(float).copy()
y_val = df_val["attack_tag"] 
y_val_binary = (y_val != 'normal').astype(int)
         


# scale the data
scaler = PowerTransformer(method="yeo-johnson")

X_scaled = scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(X_scaled, columns=features)

x_test_scaled = scaler.transform(x_test)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=features)

x_val_scaled = scaler.transform(x_val)
x_val_scaled = pd.DataFrame(x_val_scaled, columns=features)

joblib.dump(scaler, f"{output_dir}/{Dataset}_scaler_powertransformer_encoded.pkl")

print(f"************DataFrame form: {x_train_scaled.shape}, labels: {y_train.shape}")
print("'*************************Starting scaling and model building...\n")




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
            
    model.add(Dense(len(target_only_attacks), activation="softmax", name="output"))
    return model

def sparse_focal_loss(gamma=2.0):

    def loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.int32)

        y_true_one_hot = tf.one_hot(
            y_true,
            depth=tf.shape(y_pred)[-1]
        )

        epsilon = 1e-7
        y_pred = tf.clip_by_value(
            y_pred,
            epsilon,
            1.0 - epsilon
        )

        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)

        focal_weight = tf.pow(
            1 - y_pred,
            gamma
        )

        focal_loss = focal_weight * cross_entropy

        return tf.reduce_sum(focal_loss, axis=-1)

    return loss

def sparse_weighted_focal_loss(gamma=2.0, class_weights=None):
    if class_weights is not None:
        weight_list = [class_weights[i] for i in sorted(class_weights.keys())]
        class_weights_tensor = tf.constant(weight_list, dtype=tf.float32)
    else:
        class_weights_tensor = None

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        pt = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1, keepdims=True)

        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - pt, gamma)
        focal_loss = focal_weight * cross_entropy

        if class_weights_tensor is not None:
            focal_loss = focal_loss * class_weights_tensor

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    return loss

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

#architectures = [architecture_1]

architectures = [architecture_1, architecture_2, architecture_3 , architecture_4, architecture_5, architecture_6, architecture_7, architecture_8, architecture_9, architecture_10, architecture_11]
architectures_without_batch = [arch for arch in architectures if "batch" not in str(arch)]
architectures_with_batch = [arch for arch in architectures if "batch" in str(arch)]

print(f"\n ***************************Dataset: {Dataset} - will train {len(architectures)} models with different architectures...")
print(f"features: {features}")

# To save the result output
results_list = []

for i, arch in enumerate(architectures, 1):
    print(f" \n \n **********************************Training model {i} with architecture: {arch}...")

    Start_time = pd.Timestamp.now() # Start time for training the model, to calculate total time taken at the end of this architecture

    base_model = create_model(len(features), architecture=arch )
    base_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        #loss="sparse_categorical_crossentropy",
        #loss=sparse_focal_loss(gamma=2.0),
        loss=sparse_weighted_focal_loss(gamma=gamma_focal_loss, class_weights=class_weights),
        metrics=["accuracy"],
    )
    print(f"Max epochs: {epochs}, learning_rate: {learning_rate}, batch_size: {batch_size}")
    print(f"EarlyStop epoch: {Early_stop_patience}, reduce_LR_patience: {reduce_LR_patience}, reduce_LR_factor: {reduce_LR_factor} ")
    print(f"Gamma gamma_focal_loss:{gamma_focal_loss}")
    base_model.fit(
        x_train_scaled,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val_scaled, y_val),
        class_weight=None,         # i Used 1:10, please loock at the results above for different class weights, and select the one that gives the best MCC
        verbose=2,              # Set to 2 for epoch-level logging, 1 for batch-level logging, and 0 for no logging
        callbacks=callbacks,
    )

    base_model.save(f"{output_dir}/original_architecture_{i}.keras")

    print(f"***************{Dataset} model architecture{i} saved.")
    print("*************** Calculating y_pred_probs...")
    y_pred_probs = base_model.predict(x_test_scaled, verbose=0)

    # Find the indx to attack label
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n--- DEBUG Label Comparison ---")
    print(f"y_test labels: {np.unique(y_test)}")

    # *************************
    #  Calculate and print the classification report and MCC

    print(f"--- Classification Report --- Dataset {Dataset} --- arch {i}")
    report_dict = classification_report(y_test, y_pred, target_names=target_only_attacks , output_dict=True)

    print(report_dict)
    cm = confusion_matrix(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    ANN_mcc = matthews_corrcoef(y_test, y_pred)

    print("\n--- Confusion Matrix ---")
    print(conf_matrix)
    print(f"-----------------")

    for idx, cn in enumerate(target_only_attacks):
        tp = conf_matrix[idx, idx]
        fp = conf_matrix[:, idx].sum() - tp
        fn = conf_matrix[idx, :].sum() - tp
        tn = conf_matrix.sum() - (tp + fp + fn)

        acc = (tp + tn) / conf_matrix.sum()

        class_metrics = report_dict.get(cn, {})
        
        row = {
            "Dataset": Dataset,
            "Architecture": i,
            "Class": cn,
            "Precision": class_metrics.get('precision', 0),
            "Recall": class_metrics.get('recall', 0),
            "F1-Score": class_metrics.get('f1-score', 0),
            "Accuracy": acc,
            "MCC": ANN_mcc
        }
        results_list.append(row)

    print(f"Model: {i}, (MCC): {ANN_mcc:.4f}, architecture: {arch}")
    print(f"(F1-Score_weighted): {report_dict['weighted avg']['f1-score']:.4f}, (F1-Score_macro): {report_dict['macro avg']['f1-score']:.4f}  \n (Accuracy): {report_dict['accuracy']:.4f}, (Recall): {report_dict['weighted avg']['recall']:.4f}, (Precision): {report_dict['weighted avg']['precision']:.4f}")
    print(f"Our model has: {len(base_model.layers)} layers and {base_model.count_params()} parameters.")
    diff_min = (pd.Timestamp.now() - Start_time).total_seconds() / 60
    print(f"Total time in minutes: {diff_min:.2f} min")



# Save the results
df_results = pd.DataFrame(results_list)
print("\n--- Detailed Metrics per Class ---")
print(df_results.to_string(index=False))
csv_filename = f"results_{Dataset}.csv"
df_results.to_csv(f"{output_dir}/{csv_filename}", index=False)
print(f"\nResults saved to {csv_filename}")

