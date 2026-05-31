import os
print(f"THIS COMPUTER HAS {os.cpu_count()} CPU cores: ")

os.environ["PYTHONHASHSEED"] = "42"

import time
from tkinter import Y
import warnings
import random
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from shared_var import features_goose, features_9, features_13


# pip install torch numpy pandas scikit-learn

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Preparing the folder structure and global variables
# ==========================================

features = None
Dataset = None
# Ereno or PowerDuck , else to use the BOTH datasets
pruning_mode = None

while not Dataset:
    choice = input(
        "Which data do you want to use? (1 for BOTH, 2 for Ereno, 3 for PowerDuck): "
    )

    if choice == "1":
        Dataset = "BOTH"
        features = features_goose
        print(" Using 6 features as defult.")

    elif choice == "2":
        Dataset = "Ereno"
    elif choice == "3":
        Dataset = "PowerDuck"
        features = features_goose
        print(" Using 6 features as defult.")
    else:
        print("Invalid choice, please try again.")

while not pruning_mode:
    choice = input(" \n Select pruning mode, [1] Unstructured,  [2] Structured: ")

    if choice == "1":
        pruning_mode = "unstructured"
        print(" Unstructured pruning selected (default)")
    elif choice == "2":
        pruning_mode = "structured"
        print(" Structured pruning selected")
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


# batch_size = 512, 1024, 2048 , 4096 ,8192
learning_rate = 0.001
Early_stop_patience = 10
reduce_LR_factor = 0.6
reduce_LR_patience = 4
phase1_max_epochs = 30
phase2_max_epochs = 100
mcc_target_global = 0.90

batch_size_pruning = 16384

# Global variabel for pruning aware training
batch_size_pruning_arr = [256, 512, 1024, 2048, 4096]

final_pruning_sparsity = 0.99


if Dataset == "Ereno":
    static_weight = {0: 1.0, 1: 10.0}
    #static_weight = {0: 1.0, 1: 13.75}
elif Dataset == "PowerDuck":
    static_weight = {0: 1.0, 1: 1.0}
else:
    static_weight = {0: 1.0, 1: 10.0}

print(f"Using static_weight: {static_weight}")
print (f"batch_size_pruning: {batch_size_pruning}")
print(f"mcc_target_global: {mcc_target_global}")
results_list = []

output_dir = f"./data/ANN_pruned_awaer_traning/2_phase_v3/features_{len(features)}/batch_size_pruning_{batch_size_pruning}/{Dataset}/"
print(output_dir)

folder_structure = {
    f"./data/ANN_pruned_awaer_traning/2_phase_v3/features_{len(features)}/batch_size_pruning_{batch_size_pruning}": [
        "BOTH",
        "Ereno",
        "PowerDuck",
    ],
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
def load_powerDuck_data():

    df = pd.read_csv("./data/powerduck/powerduck-labeled_final.csv")
    df = df[df["gooseTimeAllowedtoLive"] != 0]
    df = df.reset_index(drop=True)

    # Because we have only 2 record with this type.
    df.loc[df["attack_tag"] == "01-replay-opening-switch-isolated", "attack_tag"] = (
        "02-replay-opening-switch-w-context"
    )

    print(f"df.head()  \n {df.head()}")
    print("df class", df["class"].unique())
    print("df attack_tag", df["attack_tag"].unique())
    df["class"] = df["attack_tag"]

    df.drop(columns=["split"], inplace=True)
    df["split"] = None

    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.25, stratify=df["class"], random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=df.loc[temp_idx, "class"], random_state=42
    )
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    df_val = df[df["split"] == "val"].copy().reset_index(drop=True)
    df_test = df[df["split"] == "test"].copy().reset_index(drop=True)
    df_train = df[df["split"] == "train"].copy().reset_index(drop=True)

    return df_val, df_train, df_test

def load_ereno_data():
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

    print("df_head", df_ereno.head())
    print(df_ereno["class"].unique())

    df_ereno["split"] = None
    train_idx, val_idx = train_test_split(
        df_ereno.index, test_size=0.20, stratify=df_ereno["class"], random_state=42
    )

    df_ereno["split"] = None

    train_idx, val_idx = train_test_split(
        df_ereno.index, test_size=0.20, stratify=df_ereno["class"], random_state=42
    )

    df_ereno.loc[train_idx, "split"] = "train"
    df_ereno.loc[val_idx, "split"] = "val"

    df_val = df_ereno[df_ereno["split"] == "val"].copy().reset_index(drop=True)
    df_train = df_ereno[df_ereno["split"] == "train"].copy().reset_index(drop=True)
    df_test = df_ereno_test.copy().reset_index(drop=True)

    return df_val, df_train, df_test

if not Dataset:
    if Dataset == "Ereno":
        df_val, df_train, df_test = load_ereno_data

    elif Dataset == "PowerDuck":
        df_val, df_train, df_test = load_powerDuck_data

    elif Dataset == "PowerDuck":
        df_ereno_val, df_ereno_train, df_ereno_test = load_ereno_data
        df_powerduck_val, df_powerduck_test, df_powerduck_train = load_powerDuck_data

        selection = features + ["class"]
        df_train = pd.concat(
            [df_ereno_train[selection], df_powerduck_train[selection]], ignore_index=True
        )
        df_val = pd.concat(
            [df_ereno_val[selection], df_powerduck_val[selection]], ignore_index=True
        )
        df_test = pd.concat(
            [df_ereno_test[selection], df_powerduck_test[selection]], ignore_index=True
        )

    else:
        print("Error no selected Datraset")


dataframes = [df_train, df_val, df_test]


architecture_1 = [32, 16]
architecture_2 = [64, 32, 16]
architecture_3 = [64, "batch", 32, 0.2, 18]
architecture_4 = [128, 64, 32, 16]
architecture_5 = [128, "batch", 0.2, 64, 32, 0.2, 16]
architecture_6 = [128, 128, 64, 64, 32]
architecture_7 = [128, "batch", 0.2, 128, 64, "batch", 64, 0.2, 32]
architecture_8 = [128, 128, 64, 64, 32, 16]
architecture_9 = [128, "batch", 0.2, 128, 64, "batch", 64, 32, 0.2, 16]
architecture_10 = [256, 128, 256, 128, 128, 64, 32, 16, 8]
architecture_11 = [256, "batch" , 128 , 0.2, 256, "batch", 128, 0.2, 128, 64, 32, 16, "batch", 0.2, 8,]

architecture_extra1 = [32, 16]
architecture_extra2 = [64, 64, 32, 16]
architecture_extra3 = [64, 64, 64, 32]

extra1 = [8, 4]
extra2 = [16, 8, 4]
extra3 = [32, 16, 8, 4]

all_architectures = [architecture_1, architecture_2, architecture_3 , architecture_4, architecture_5, architecture_6, architecture_7, architecture_8, architecture_9, architecture_10, architecture_11]
NoBatch_architectures = [architecture_1, architecture_2, architecture_4, architecture_6, architecture_8, architecture_10]

not_best_architectures = [architecture_2, architecture_3 , architecture_4, architecture_5, architecture_7, architecture_9, architecture_10, architecture_11]
best_architectures = [architecture_1, architecture_6, architecture_8]

best_for_v3 = [architecture_1, architecture_4, architecture_6, architecture_10]
extra = [extra1, extra2,extra3]

architectures = extra

architectures_without_batch = [
    arch for arch in architectures if "batch" not in str(arch)
]
architectures_with_batch = [arch for arch in architectures if "batch" in str(arch)]


print(
    f"df_train DataFrame form: {df_train.shape}, Columns: {list(df_train.columns[:8])}"
)
print(f"df_val DataFrame form: {df_val.shape}, Columns: {list(df_val.columns[:8])}")
print(f"df_test DataFrame form: {df_test.shape}, Columns: {list(df_test.columns[:8])}")

# X och y
x_train = df_train[features].astype(float).copy()
y_train = (df_train["class"] != "normal").astype(int).copy().reset_index(drop=True)

x_test = df_test[features].astype(float).copy()
y_test = (df_test["class"] != "normal").astype(int).copy().reset_index(drop=True)

x_val = df_val[features].astype(float).copy()
y_val = (df_val["class"] != "normal").astype(int).copy().reset_index(drop=True)

print("df_test class. values:")
print(df_test["class"].values)




print("'*************************Starting scaling and model building...\n")

# scale the data
scaler = PowerTransformer(method="yeo-johnson")
# scaler = RobustScaler()

X_scaled = scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(X_scaled, columns=features)

X_test_scaled = scaler.transform(x_test)
x_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

x_val_scaled = scaler.transform(x_val)
x_val_scaled = pd.DataFrame(x_val_scaled, columns=features)

joblib.dump(scaler, f"{output_dir}/{Dataset}_scaler_powertransformer_binary.pkl")

print(
    f"************DataFrame form x_train_scaled: {x_train_scaled.shape}, labels: {y_train.shape}"
)
print(f"x_train_scaled df.head(): \n {x_train_scaled.head()}")


# ==========================================
# PyTorch model and loss-change pruning during training
# ==========================================



class BinaryANN(nn.Module):
    """Feed-forward ANN; output is a logit."""

    def __init__(self, input_dim, architecture):
        super().__init__()
        layers = []
        in_features = input_dim
        for layer in architecture:
            if isinstance(layer, int):
                layers.append(nn.Linear(in_features, layer))
                layers.append(nn.ReLU())
                in_features = layer
            elif layer == "batch":
                layers.append(nn.BatchNorm1d(in_features))
            elif isinstance(layer, float):
                layers.append(nn.Dropout(layer))
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)




def _linear_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]

def _init_prune_masks(model):
    return {
        m.weight: torch.ones_like(m.weight, dtype=torch.bool, device=m.weight.device)
        for m in _linear_layers(model)
    }

def _apply_prune_masks(model, prune_masks):
    with torch.no_grad():
        for m in _linear_layers(model):
            m.weight.mul_(prune_masks[m.weight])

def _mask_gradients(model, prune_masks):
    """Zero gradients for already-pruned weights so they stay dead."""
    for m in _linear_layers(model):
        if m.weight.grad is not None:
            m.weight.grad.mul_(prune_masks[m.weight])


def target_sparsity_for_epoch(epoch, total_epochs, final_sparsity, power=1):

    # use power 3 to use Polynomial ramp, at start training more at end pruning mest.
    if total_epochs <= 0:
        return 0.0
    progress = min(1.0, max(0.0, (epoch + 1) / total_epochs))
    return final_sparsity * (progress**power)


def get_pruning_detail(model, prune_masks) -> pd.DataFrame:
    """Returns per-layer pruning detail as a DataFrame."""
    linear_layers = _linear_layers(model)
    rows = []

    for li, m in enumerate(linear_layers):
        mask = prune_masks[m.weight].cpu()
        n_w = mask.numel()
        n_nz = int(mask.sum().item())
        n_z = n_w - n_nz
        dead_out = int((mask.sum(dim=1) == 0).sum().item())
        dead_in = int((mask.sum(dim=0) == 0).sum().item())
        total_out = mask.shape[0]
        total_in = mask.shape[1]

        rows.append(
            {
                "Layer": f"L{li + 1} ({total_in}→{total_out})",
                "Weights": n_w,
                "Nonzero": n_nz,
                "Zeroed": n_z,
                "Sparsity": f"{n_z / max(n_w, 1):.1%}",
                "Dead_out": f"{dead_out}/{total_out}",
                "Dead_in": f"{dead_in}/{total_in}",
            }
        )

    # totals row
    total_w = sum(r["Weights"] for r in rows)
    total_nz = sum(r["Nonzero"] for r in rows)
    total_z = total_w - total_nz
    rows.append(
        {
            "Layer": "TOTAL",
            "Weights": total_w,
            "Nonzero": total_nz,
            "Zeroed": total_z,
            "Sparsity": f"{total_z / max(total_w, 1):.1%}",
            "Dead_out": "—",
            "Dead_in": "—",
        }
    )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def prune_by_loss_change_structured(
    model, criterion, x_cal, y_cal, target_sparsity, prune_masks, original_neurons, min_neurons_layer=2
):
    """
    Structured global pruning — removes entire neurons (rows) with lowest
    Taylor scores: sum(|w * dL/dw|) per neuron.
    Physically rebuilds the model with smaller layers after pruning.
    Returns updated model and prune_masks.
    """
    if target_sparsity <= 0:
        return model, prune_masks

    model.eval()
    x_cal = x_cal.to(device)
    y_cal = y_cal.to(device)
    model.zero_grad(set_to_none=True)
    logits = model(x_cal)
    loss = criterion(logits, y_cal)
    loss.backward()
    _mask_gradients(model, prune_masks)

    linear_layers = _linear_layers(model)

    num_hidden_layers = len(linear_layers) - 1
    
    min_possible_neurons = min_neurons_layer * num_hidden_layers

    neuron_scores = []
    for li, m in enumerate(linear_layers[:-1]):
        if m.weight.grad is None:
            continue
        scores_per_neuron = (m.weight.data * m.weight.grad).abs().sum(dim=1)
        for ni, score in enumerate(scores_per_neuron):
            neuron_scores.append((li, ni, score.item()))

    if not neuron_scores:
        return model, prune_masks

    # how many neurons to prune 
    total_neurons = sum(m.weight.shape[0] for m in linear_layers[:-1])
    n_to_prune = max(0, int(round(target_sparsity * total_neurons)))
    n_to_prune = min(n_to_prune, len(neuron_scores))

    # how many neurons model have now
    current_neurons = sum(m.weight.shape[0] for m in linear_layers[:-1])
    
    target_neurons = max(
        min_possible_neurons, 
        int(round((1.0 - target_sparsity) * original_neurons))
    )
    n_to_prune = current_neurons - target_neurons
    
    if n_to_prune <= 0:
        return model, prune_masks

    #  select lowest scoring neurons 
    neuron_scores.sort(key=lambda x: x[2])
    pruned = neuron_scores[:n_to_prune]

    #  build set of neurons to remove per layer 
    remove = {}  # {layer_idx: set of neuron indices}
    for li, ni, _ in pruned:
        remove.setdefault(li, set()).add(ni)

    #  rebuild model with smaller layers 
    with torch.no_grad():
        new_layers = []
        prev_keep = None  # which input indices survive into next layer

        for li, m in enumerate(linear_layers):
            out_size = m.weight.shape[0]
            in_size = m.weight.shape[1]

            # which output neurons to KEEP in this layer
            keep_out = [i for i in range(out_size) if i not in remove.get(li, set())]
            if len(keep_out) < min_neurons_layer:
                scores_row = (m.weight.data * m.weight.grad).abs().sum(dim=1)
                keep_out = torch.topk(scores_row, min(min_neurons_layer, out_size), largest=True).indices.tolist()

            # which input indices survive (= keep_out from previous layer)
            if prev_keep is not None:
                keep_in = prev_keep
            else:
                keep_in = list(range(in_size))

            # slice weight and bias
            w_new = m.weight.data[keep_out, :][:, keep_in]
            b_new = m.bias.data[keep_out] if m.bias is not None else None

            new_linear = nn.Linear(
                len(keep_in), len(keep_out), bias=(b_new is not None)
            )
            new_linear.weight.data = w_new
            if b_new is not None:
                new_linear.bias.data = b_new

            new_layers.append(new_linear)

            # output layer — keep_out is ALL outputs (never prune output layer)
            if li == len(linear_layers) - 1:
                prev_keep = None
            else:
                prev_keep = keep_out

        #  rebuild Sequential with same activations 
        li = 0
        rebuilt = []
        for module in model.net:
            if isinstance(module, nn.Linear):
                rebuilt.append(new_layers[li])
                li += 1
            else:
                rebuilt.append(module)  # ReLU, Dropout, BatchNorm unchanged

        model.net = nn.Sequential(*rebuilt).to(device)

    #  reinitialize masks for new shapes 
    new_masks = _init_prune_masks(model)

    print(
        f"  [Structured] Removed {n_to_prune}/{total_neurons} neurons  "
        f"New arch: {[m.weight.shape[0] for m in _linear_layers(model)]}"
    )

    return model, new_masks


def prune_by_loss_change_unstructured(
    model, criterion, x_cal, y_cal, target_sparsity, prune_masks
):
    """
    Unstructured global pruning via Taylor loss-change scores: |w · dL/dw|.

    1: exact topk index selection — never prunes more than n_to_prune
        regardless of score ties.
    2: caller re-samples x_cal/y_cal every call (see training loop).
    3: model.eval() — does not corrupt BatchNorm running statistics.
    4: _mask_gradients called before scoring — dead weights stay silent.
    """
    if target_sparsity <= 0:
        return

    model.eval()
    x_cal = x_cal.to(device)
    y_cal = y_cal.to(device)

    model.zero_grad(set_to_none=True)
    logits = model(x_cal)
    loss = criterion(logits, y_cal)
    loss.backward()

    #  zero dead-weight gradients before reading scores
    _mask_gradients(model, prune_masks)

    # Collect flat score/position tensors for every active weight
    all_scores = []
    all_layer_idx = []
    all_pos = []

    linear_layers = _linear_layers(model)
    for li, m in enumerate(linear_layers):
        w = m.weight
        if w.grad is None:
            continue
        scores = (w.data * w.grad).abs()  # Taylor score
        active_pos = prune_masks[w].nonzero(as_tuple=False)  # shape [N, 2]
        if active_pos.numel() == 0:
            continue
        all_scores.append(scores[active_pos[:, 0], active_pos[:, 1]])
        all_layer_idx.append(
            torch.full((active_pos.shape[0],), li, dtype=torch.long, device=w.device)
        )
        all_pos.append(active_pos)

    if not all_scores:
        return

    cat_scores = torch.cat(all_scores)
    cat_layer_idx = torch.cat(all_layer_idx)
    cat_pos = torch.cat(all_pos, dim=0)

    n_total = sum(m.weight.numel() for m in linear_layers)
    n_pruned = sum((~prune_masks[m.weight]).sum().item() for m in linear_layers)
    n_to_prune = max(0, int(round(target_sparsity * n_total)) - n_pruned)
    n_to_prune = min(n_to_prune, cat_scores.numel())

    if n_to_prune <= 0:
        return

    # topk gives exact indices; no tie ambiguity
    _, topk_idx = torch.topk(cat_scores, n_to_prune, largest=False, sorted=False)

    with torch.no_grad():
        for idx in topk_idx:
            li = cat_layer_idx[idx].item()
            r = cat_pos[idx, 0].item()
            c = cat_pos[idx, 1].item()
            w = linear_layers[li].weight
            prune_masks[w][r, c] = False
            w[r, c] = 0.0

# 
# Training loop
# 


def train_with_loss_change_pruning(
    model,
    x_train_t,
    y_train_t,
    x_val_t,
    y_val_t,
    x_test_t,
    y_test_t,
    *,
    batch_size,
    learning_rate,
    pos_weight,
    early_stop_patience,
    phase1_max_epochs,
    phase2_max_epochs,
    reduce_lr_factor,
    reduce_lr_patience,
    start_prune_epoch=None,
    warmup_epochs=None,
    final_sparsity=0.5,
    architecture_number,
):

    print(f"phase1_max_epochs type: {type(phase1_max_epochs)} value: {phase1_max_epochs}")
    print(f"phase2_max_epochs type: {type(phase2_max_epochs)} value: {phase2_max_epochs}")

    if start_prune_epoch is None:
        start_prune_epoch = max(1, phase2_max_epochs // 10)

    if warmup_epochs is None:
        warmup_epochs = max(10, phase2_max_epochs // 10)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_t, y_val_t),
        batch_size=batch_size*2,
        shuffle=False,
    )

    y_val_np = y_val_t.numpy()
    y_test_np = y_test_t.numpy()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=1e-6,
    )

    prune_masks = _init_prune_masks(model)

    best_state = None
    best_masks = None

    to_save_detail_rows = []
    sparsity_checkpoints = set(np.round(np.arange(0.00, final_pruning_sparsity, 0.025), 2))
    saved_checkpoints = set()

    val_loss = float("nan")
    original_neurons = sum(m.weight.shape[0] for m in _linear_layers(model)[:-1])
    min_neurons_layer = 2  # never prune a layer below this

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1 — pre-train until MCC 0.90 or best stable MCC over 30 epochs
    # ══════════════════════════════════════════════════════════════════
    print("  Phase 1: Pre-training...")

    mcc_target          = mcc_target_global
    phase1_patience     = Early_stop_patience    # epochs without MCC improvement before giving up
    best_phase1_mcc     = -1.0
    best_phase1_state   = None
    phase1_no_improve   = 0

    for epoch in range(phase1_max_epochs):

        #  training step 
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        #  validation ─
        model.eval()
        val_loss_sum   = 0.0
        n_val          = 0
        all_val_logits = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss_sum += criterion(logits, yb).item() * len(yb)
                n_val        += len(yb)
                all_val_logits.append(logits.cpu())
        val_loss  = val_loss_sum / max(n_val, 1)
        val_probs = torch.sigmoid(torch.cat(all_val_logits)).numpy()
        scheduler.step(val_loss)

        #  compute val MCC 
        thresholds  = np.linspace(0.1, 0.9, 81)
        val_mcc_scores = [
            matthews_corrcoef(y_val_np, (val_probs > t).astype(int))
            for t in thresholds]
        
        best_threshold_val = thresholds[np.argmax(val_mcc_scores)]
        current_mcc = val_mcc_scores[np.argmax(val_mcc_scores)]


        if current_mcc > best_phase1_mcc:
            best_phase1_mcc   = current_mcc
            best_phase1_state = deepcopy(model.state_dict())
            phase1_no_improve = 0
        else:
            phase1_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  [Phase 1] Epoch {epoch+1}  val_loss={val_loss:.4f}  "
                f"MCC={current_mcc:.4f}  best={best_phase1_mcc:.4f} "
                f"threshold={best_threshold_val:.2f}"
            )

        #  stop conditions 
        if current_mcc >= mcc_target:
            print(f"  Phase 1 done — MCC {current_mcc:.4f} >= {mcc_target} "
                f"at epoch {epoch+1}")
            break

        if phase1_no_improve >= phase1_patience:
            print(f"  Phase 1 done — best MCC {best_phase1_mcc:.4f} "
                f"(no improvement for {phase1_patience} epochs) "
                f"at epoch {epoch+1}")
            break

    #  restore best phase 1 state before pruning ─
    if best_phase1_state is not None:
        model.load_state_dict(best_phase1_state)
        print(f"  Restored best Phase 1 state  MCC={best_phase1_mcc:.4f}")

    #  reset optimizer and scheduler for phase 2 ─
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=1e-6,
    )
    prune_masks = _init_prune_masks(model)


    # ══════════════════════════════════════════════════════════════════
    # PHASE 2 — pruning loop (existing loop, unchanged)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n  Phase 2: Pruning — {phase2_max_epochs} epochs remaining\n")
    original_neurons  = sum(m.weight.shape[0] for m in _linear_layers(model)[:-1])
    start_prune_epoch = 0   # start pruning from 0 in phase 2
    val_loss = float("nan")

    for epoch in range(phase2_max_epochs):
        t0 = time.perf_counter()   
        # training step
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            _mask_gradients(model, prune_masks)
            optimizer.step()
            _apply_prune_masks(model, prune_masks)

        t1 = time.perf_counter()
            # validation
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        all_val_logits = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss_sum += criterion(logits, yb).item() * len(yb)
                n_val += len(yb)
                all_val_logits.append(logits.cpu())
        val_loss = val_loss_sum / max(n_val, 1)
        val_probs = torch.sigmoid(torch.cat(all_val_logits)).numpy()
        scheduler.step(val_loss)

        t2 = time.perf_counter()

        # pruning step
        sparsity = target_sparsity_for_epoch(epoch, phase2_max_epochs, final_sparsity)
        if epoch >= start_prune_epoch and sparsity > 0:
            # fresh calibration sample every pruning call
            cal_idx = torch.randperm(len(x_train_t))[:batch_size]
            x_cal = x_train_t[cal_idx]
            y_cal = y_train_t[cal_idx]

            if pruning_mode == "structured":
                current_neurons = sum(
                    m.weight.shape[0] for m in _linear_layers(model)[:-1]
                )
                min_possible = min_neurons_layer * (len(_linear_layers(model)) - 1)
                if current_neurons > min_possible:
                    old_arch = [m.weight.shape[0] for m in _linear_layers(model)]
                    model, prune_masks = prune_by_loss_change_structured(
                        model,
                        criterion,
                        x_cal,
                        y_cal,
                        sparsity,
                        prune_masks,
                        original_neurons,
                        min_neurons_layer=min_neurons_layer,
                    )

                    new_arch = [m.weight.shape[0] for m in _linear_layers(model)]

                    if old_arch != new_arch:
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            elif pruning_mode == "unstructured":
                prune_by_loss_change_unstructured(
                    model, criterion, x_cal, y_cal, sparsity, prune_masks
                )
                _apply_prune_masks(model, prune_masks)

            else:
                print(" Error, nor pruning method selected!!!")
                
                
            t3 = time.perf_counter()   

            # actual sparsity
            if pruning_mode == "structured":
                current_neurons = sum(
                    m.weight.shape[0] for m in _linear_layers(model)[:-1]
                )
                actual_sparsity = round(
                    1 - current_neurons / max(original_neurons, 1), 2
                )
            else:
                actual_sparsity = round(
                    sum((~m).sum().item() for m in prune_masks.values())
                    / sum(m.numel() for m in prune_masks.values()),
                    2,
                )
            # sparsity checkpoints
            for ckpt in sorted(sparsity_checkpoints - saved_checkpoints):
                if actual_sparsity >= ckpt:
                    with torch.no_grad():
                        test_probs = (
                            torch.sigmoid(model(x_test_t.to(device))).cpu().numpy()
                        )

                    # honect threshold on val set
                    thresholds = np.linspace(0.1, 0.9, 81)
                    best_threshold_val = thresholds[
                        np.argmax(
                            [
                                matthews_corrcoef(y_val_np, (val_probs > t).astype(int))
                                for t in thresholds
                            ]
                        )
                    ]

                    y_pred_arr = (test_probs > best_threshold_val).astype(int)
                    y_test_arr = y_test_np

                    cm = confusion_matrix(y_test_arr, y_pred_arr)
                    tn, fp, fn, tp = cm.ravel()
                    ANN_mcc = matthews_corrcoef(y_test_arr, y_pred_arr)
                    acc = (tp + tn) / (tp + tn + fp + fn)

                    print(f"  TP: {tp:>7}   FP: {fp:>7}")
                    print(f"  FN: {fn:>7}   TN: {tn:>7}")
                    print(
                        f"  MCC: {ANN_mcc:.4f}   Acc: {acc:.4f}   "
                        f"Threshold: {best_threshold_val:.2f}"
                    )
                    detail_df = get_pruning_detail(model, prune_masks)
                    for _, row in detail_df.iterrows():
                        to_save_detail_rows.append(
                            {
                                "Sparsity_target": ckpt,
                                "Sparsity_actual": actual_sparsity,
                                "Epoch": epoch + 1,
                                **row.to_dict(),
                            }
                        )

                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)  # = sensitivity
                    specificity = tn / max(tn + fp, 1)
                    f1 = 2 * tp / max(2 * tp + fp + fn, 1)

                    results_list.append(
                        {
                            "Dataset": Dataset,
                            "Architecture": i,
                            "Epoch": epoch + 1,
                            "Sparsity_target": ckpt,
                            "Sparsity_actual": actual_sparsity,
                            "Parameters_total": count_parameters(model),

                            "Threshold": best_threshold_val,
                            "TP": tp,
                            "TN": tn,
                            "FP": fp,
                            "FN": fn,
                            "Honest-MCC": ANN_mcc,
                            "Accuracy": round(acc, 4),
                            "Precision": round(precision, 4),
                            "Recall": round(recall, 4),
                            "Specificity": round(specificity, 4),
                            "F1": round(f1, 4),
                            "Val_loss": round(val_loss, 4),
                        }
                    )

                    torch.save(
                        {
                            "state_dict": deepcopy(model.state_dict()),
                            "architecture": arch,
                            "features": features,
                            "prune_masks": [
                                prune_masks[m.weight].cpu()
                                for m in _linear_layers(model)
                            ],
                            "final_sparsity": ckpt,
                            "actual_sparsity": actual_sparsity,
                            "epoch": epoch,
                            "threshold": best_threshold_val,
                            "mcc": ANN_mcc,
                        },
                        f"{output_dir}/{Dataset}_Architecture{i}_sparsity_{ckpt:.2f}.pt",
                    )
                    saved_checkpoints.add(ckpt)
                    print(
                        f"  Checkpoint saved — target {ckpt:.0%}  "
                        f"actual {actual_sparsity:.2%}  epoch {epoch + 1}"
                    )

        t4 = time.perf_counter()   
        
        if epoch == 0:
            print(f"  --- TIMING EPOCH 0 ---")
            print(f"  Training:   {t1-t0:.2f}s  ({len(train_loader)} batches, {(t1-t0)/len(train_loader)*1000:.1f}ms/batch)")
            print(f"  Validation: {t2-t1:.2f}s  ({len(val_loader)} batches)")
            print(f"  Pruning:    {t3-t2:.2f}s")
            print(f"  Statistik and save:    {t4-t3:.2f}s")
            print(f"  Total:      {t4-t0:.2f}s/epoch → ~{(t3-t0)*phase2_max_epochs/60:.1f}min total")

        # early stopping : stop when 90% sparsity is reached
        if pruning_mode == "structured":
            current_neurons_es = sum(m.weight.shape[0] for m in _linear_layers(model)[:-1])
            actual_sparsity_es = 1 - current_neurons_es / max(original_neurons, 1)
        else:
            n_pruned_es        = sum((~m).sum().item() for m in prune_masks.values())
            n_weights_es       = sum(  m.numel()       for m in prune_masks.values())
            actual_sparsity_es = n_pruned_es / max(n_weights_es, 1)

        if actual_sparsity_es >= final_pruning_sparsity:
            print(f"  Target {final_pruning_sparsity:.0%} sparsity reached at epoch {epoch + 1} — stopping.")
            break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
            f"  Epoch {epoch+1}/{phase2_max_epochs}  val_loss={val_loss:.4f}  "
            f"sparsity_target={sparsity:.2%}  actual={actual_sparsity_es:.2%}"
        )

    if to_save_detail_rows:
        detail_df_full = pd.DataFrame(to_save_detail_rows)
        detail_df_full.to_csv(
            f"{output_dir}/{Dataset}_Architecture{i}_pruning_detail.csv", index=False
        )
        print(f" Detail saved {Dataset}_Architecture{i}_pruning_detail.csv")

    #  restore best checkpoint 
    if best_state is not None:
        model.load_state_dict(best_state)
        if best_masks is not None:
            for k, v in prune_masks.items():
                v.copy_(best_masks[k])
        _apply_prune_masks(model, prune_masks)

    return model, prune_masks



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_nonzero_weights(model):
    return sum((p != 0).sum().item() for m in _linear_layers(model) for p in [m.weight])


# Tensors for training (float32)
X_train_t = torch.tensor(x_train_scaled.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
X_val_t = torch.tensor(x_val_scaled.values, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32)
X_test_t = torch.tensor(x_test_scaled.values, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

pos_weight = torch.tensor(
    [static_weight[1] / static_weight[0]], dtype=torch.float32, device=device
)


# ==========================================
# Building and training the model (loss-change pruning during training)
# ==========================================

print(
    f"\n ***************************Dataset: {Dataset} - will train {len(architectures)} models with different architectures..."
)
print(f"Device: {device}")
print(f"Architectures: {architectures}")
print(f"features: {features}")
print(
    f"Pruning: loss-change (Taylor |w*grad|), final_sparsity={final_pruning_sparsity}, "
    f"Phase_1 epochs={phase1_max_epochs}, Phase_2 epochs={phase2_max_epochs}, batch_size={batch_size_pruning}"
)

# To save the result output
results_list_end = []
save_threshold = []

for i, arch in enumerate(architectures, 1):
    print(
        f" \n **********************************Training model {i} with architecture: {arch}..."
    )

    Start_time = pd.Timestamp.now()

    model = BinaryANN(len(features), architecture=arch).to(device)
    #model = torch.compile(model) # krashar :?
    print(
        f" \n Phase_1 epochs={phase1_max_epochs}, Phase_2 epochs={phase2_max_epochs},  learning_rate: {learning_rate}, "
        f"batch_size: {batch_size_pruning}"
    )

    print(
        f"EarlyStop patience: {Early_stop_patience}, reduce_LR_patience: {reduce_LR_patience}, "
        f"reduce_LR_factor: {reduce_LR_factor} \n "
    )

    model, prune_masks = train_with_loss_change_pruning(
        model,
        X_train_t,
        y_train_t,
        X_val_t,
        y_val_t,
        X_test_t,
        y_test_t,
        phase1_max_epochs = phase1_max_epochs,
        phase2_max_epochs = phase2_max_epochs,
        batch_size=batch_size_pruning,
        learning_rate=learning_rate,
        pos_weight=pos_weight,
        early_stop_patience=Early_stop_patience,
        reduce_lr_factor=reduce_LR_factor,
        reduce_lr_patience=reduce_LR_patience,
        start_prune_epoch=0,
        final_sparsity=final_pruning_sparsity,
        architecture_number=i,
    )
    print(f"***************{Dataset} model architecture {i} done.")

    # Architecture model summary
    n_weights = sum(m.weight.numel() for m in _linear_layers(model))
    n_nonzero = count_nonzero_weights(model)
    print(f"\n  Model modules: {len(list(model.modules()))}")
    print(f"  Parameters   : {count_parameters(model):,}")
    print(
        f"  Weight sparsity: {1 - n_nonzero / max(n_weights, 1):.1%} ({n_nonzero:,}/{n_weights:,} nonzero)"
    )
    diff_min = (pd.Timestamp.now() - Start_time).total_seconds() / 60
    print(f"  Time elapsed : {diff_min:.2f} min")
    print("=" * 55)

results_df = pd.DataFrame(results_list)
results_df.to_csv(f"{output_dir}/{Dataset}_sparsity_comparison.csv", index=False)
print(f"  Saved {Dataset}_sparsity_comparison.csv")
