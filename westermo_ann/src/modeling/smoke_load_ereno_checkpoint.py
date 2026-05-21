"""Load a checkpoint written by train_ereno_ann.py (--save-model) and run a quick forward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modeling.train_ereno_ann import ErenoANN


def main() -> None:
    p = argparse.ArgumentParser(description="Load ERENO ANN .pt checkpoint and smoke-test forward.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--batch", type=int, default=8, help="Synthetic batch size for smoke forward.")
    args = p.parse_args()
    ck = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    input_dim = int(ck["input_dim"])
    num_classes = int(ck["num_classes"])
    labels = ck.get("labels", [])
    print(f"[load] checkpoint={args.checkpoint}")
    print(f"[load] input_dim={input_dim} num_classes={num_classes} labels={labels[:12]}{'...' if len(labels) > 12 else ''}")
    model = ErenoANN(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(sd)
    model.eval()
    x = torch.randn(args.batch, input_dim)
    with torch.no_grad():
        logits = model(x)
    print(f"[run] forward logits shape={tuple(logits.shape)} mean={float(logits.mean()):.4f} std={float(logits.std()):.4f}")
    print("[run] smoke OK")


if __name__ == "__main__":
    main()
