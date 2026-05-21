"""
Latent-space visualization for a trained ERENO ANN checkpoint.

Loads a `.pt` produced by `train_ereno_stream.py --save-model`,
streams the test CSV through the same encoder, gathers stratified rows
per class, computes penultimate-layer activations, and saves a 2-D PCA
(or t-SNE) scatter coloured by class.

Example:
  python src/modeling/embed_viz.py \\
    --checkpoint reports/stream_full_focal_nocw_seed42.pt \\
    --test-csv outputs/test_real.csv \\
    --rows-per-class 3000 \\
    --method pca \\
    --output-png reports/embed_pca.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modeling.train_ereno_ann import ErenoANN  # noqa: E402
from modeling.train_ereno_stream import (  # noqa: E402
    StreamEncoderState,
    _encode_chunk,
)


def _state_from_ckpt(enc: dict, test_csv: Path) -> StreamEncoderState:
    numeric_cols = list(enc["numeric_cols"])
    obj_cols = list(enc["obj_cols"])
    cat_maps = {k: dict(v) for k, v in enc["cat_maps"].items()}
    label_to_id = {str(k): int(v) for k, v in enc["label_to_id"].items()}
    labels = sorted(label_to_id, key=lambda k: label_to_id[k])
    target_col = enc["target_col"]
    resolved = enc.get("resolved_target")
    if not resolved:
        head = pd.read_csv(test_csv, nrows=200)
        if target_col in head.columns and head[target_col].astype(str).str.strip().notna().any():
            resolved = target_col
        elif "delay" in head.columns and head["delay"].astype(str).isin(set(labels)).any():
            resolved = "delay"
        else:
            resolved = target_col
    binary_mode = bool(enc.get("binary_mode", False))
    return StreamEncoderState(
        target_col=target_col,
        resolved_target=resolved,
        feature_cols=numeric_cols + obj_cols,
        numeric_cols=numeric_cols,
        obj_cols=obj_cols,
        dropped_numeric=[],
        mean=np.asarray(enc["mean"], dtype=np.float32),
        std=np.asarray(enc["std"], dtype=np.float32),
        cat_maps=cat_maps,
        labels=labels,
        label_to_id=label_to_id,
        num_classes=len(labels),
        binary_mode=binary_mode,
        multiclass_label_to_id=enc.get("multiclass_label_to_id"),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2-D embedding visualization for a saved ERENO ANN.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt file from --save-model.")
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--chunk-rows", type=int, default=200_000)
    p.add_argument("--rows-per-class", type=int, default=2000, help="Cap on rows sampled per class.")
    p.add_argument("--method", choices=["pca", "tsne"], default="pca")
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-png", type=Path, default=Path("reports/embed_pca.png"))
    p.add_argument(
        "--highlight-classes",
        type=str,
        default="inverse_replay,masquerade_fake_fault,masquerade_fake_normal,normal,poisoned_high_rate",
        help="Classes to emphasize; non-listed classes are drawn smaller and lighter.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = _state_from_ckpt(ckpt["encoder"], args.test_csv)
    ga_mask = np.asarray(ckpt.get("ga_feature_mask", []), dtype=bool)
    if ga_mask.size == 0:
        ga_mask = np.ones(int(ckpt["input_dim"]), dtype=bool)
    hidden_dims = ckpt.get("hidden_dims", [128, 64])
    model = ErenoANN(
        input_dim=int(ckpt["input_dim"]),
        num_classes=int(ckpt["num_classes"]),
        hidden_dims=hidden_dims,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(
        f"[viz] loaded checkpoint labels={state.labels} hidden_dims={hidden_dims} "
        f"ga_features={int(ga_mask.sum())}/{ga_mask.size}",
        flush=True,
    )

    cap = int(args.rows_per_class)
    pools_x: dict[int, list[np.ndarray]] = {i: [] for i in range(state.num_classes)}
    counts: dict[int, int] = {i: 0 for i in range(state.num_classes)}
    reader = pd.read_csv(args.test_csv, chunksize=args.chunk_rows)
    for chunk in reader:
        x_np, y_np = _encode_chunk(chunk, state)
        if len(x_np) == 0:
            continue
        x_np = x_np[:, ga_mask]
        for cid in np.unique(y_np):
            cid_int = int(cid)
            mask = y_np == cid_int
            sub = x_np[mask]
            counts[cid_int] += int(mask.sum())
            need = cap - len(pools_x[cid_int])
            if need <= 0:
                continue
            if len(sub) > need:
                pick = rng.choice(len(sub), size=need, replace=False)
                sub = sub[pick]
            pools_x[cid_int].extend(list(sub))
        if all(len(pools_x[i]) >= cap for i in range(state.num_classes)):
            break

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for cid, rows in pools_x.items():
        for r in rows:
            xs.append(r)
            ys.append(cid)
    if not xs:
        raise RuntimeError("No rows accumulated for visualization (test CSV empty or labels unmapped).")
    x = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    seen_summary = ", ".join(
        f"{state.labels[i]}={len(pools_x[i])}/{counts[i]}" for i in range(state.num_classes)
    )
    print(f"[viz] sampled rows={len(x)} per-class<= {cap} (taken/seen): {seen_summary}", flush=True)

    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        emb = model.features(x_t).cpu().numpy()

    if args.method == "pca":
        proj = PCA(n_components=2, random_state=args.seed).fit_transform(emb)
        title = "PCA of penultimate-layer activations"
    else:
        from sklearn.manifold import TSNE  # local import: heavy

        tsne = TSNE(
            n_components=2,
            perplexity=args.tsne_perplexity,
            random_state=args.seed,
            init="pca",
            learning_rate="auto",
        )
        proj = tsne.fit_transform(emb)
        title = f"t-SNE of penultimate-layer activations (perplexity={args.tsne_perplexity})"

    highlight = {s.strip() for s in (args.highlight_classes or "").split(",") if s.strip()}
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=140)
    for cid in range(state.num_classes):
        mask = y == cid
        if not mask.any():
            continue
        cls = state.labels[cid]
        is_hi = cls in highlight
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=8 if is_hi else 4,
            alpha=0.7 if is_hi else 0.25,
            color=cmap(cid % 10),
            label=f"{cls} (n={int(mask.sum())})",
            linewidths=0,
        )
    ax.set_title(f"{title}\ncheckpoint={args.checkpoint.name}")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(fontsize=8, loc="best")
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_png)
    plt.close(fig)
    print(f"[viz] wrote {args.output_png}", flush=True)


if __name__ == "__main__":
    main()
