"""
data_plotter/main.py
--------------------
Interactive CLI tool that reads CSV files from ./inputs/ and generates
matplotlib plots saved to ./outputs/.
"""

import os
import sys
import glob
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # never open a window
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ────────────────────────────────────────────────────────────────────

INPUTS_DIR  = Path("inputs")
OUTPUTS_DIR = Path("outputs")

# ── Helpers ──────────────────────────────────────────────────────────────────

def clr(code: str, text: str) -> str:
    """ANSI colour helper."""
    codes = {"h": "\033[1m", "b": "\033[94m", "g": "\033[92m",
             "y": "\033[93m", "r": "\033[91m", "c": "\033[96m", "x": "\033[0m"}
    return f"{codes.get(code,'')}{text}{codes['x']}"


def hr(char: str = "─", width: int = 60) -> None:
    print(clr("b", char * width))


def pick(prompt: str, options: list, allow_back: bool = False) -> Optional[int]:
    """
    Numbered menu picker.  Returns 0-based index, or None if user picks 'back'.
    """
    for i, opt in enumerate(options, 1):
        print(f"  {clr('y', str(i))}) {opt}")
    if allow_back:
        print(f"  {clr('y', '0')}) ← Back")
    while True:
        raw = input(clr("h", f"\n{prompt} › ")).strip()
        if allow_back and raw == "0":
            return None
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        print(clr("r", "  Invalid choice – try again."))


def pick_column(prompt: str, columns: list[str], allow_back: bool = True) -> Optional[str]:
    """Pick a single column by number."""
    idx = pick(prompt, columns, allow_back=allow_back)
    return None if idx is None else columns[idx]


def pick_columns_multi(prompt: str, columns: list[str]) -> Optional[list[str]]:
    """Pick multiple columns (comma-separated numbers or ranges like 1,3,5-7)."""
    print(f"\n  {clr('c', 'Available columns:')}")
    for i, c in enumerate(columns, 1):
        print(f"    {clr('y', str(i))}) {c}")
    print(f"    {clr('y', '0')}) ← Back")
    while True:
        raw = input(clr("h", f"\n{prompt} (e.g. 1,3,5-7) › ")).strip()
        if raw == "0":
            return None
        selected = set()
        try:
            for part in raw.split(","):
                part = part.strip()
                if "-" in part:
                    a, b = part.split("-")
                    selected.update(range(int(a), int(b) + 1))
                else:
                    selected.add(int(part))
            chosen = [columns[i - 1] for i in sorted(selected) if 1 <= i <= len(columns)]
            if chosen:
                return chosen
        except Exception:
            pass
        print(clr("r", "  Invalid input – try again."))


def safe_output_path(stem: str, suffix: str = ".png") -> Path:
    """Return a unique path inside OUTPUTS_DIR."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    p = OUTPUTS_DIR / f"{stem}{suffix}"
    n = 1
    while p.exists():
        p = OUTPUTS_DIR / f"{stem}_{n}{suffix}"
        n += 1
    return p


def save_and_announce(fig: plt.Figure, stem: str) -> None:
    path = safe_output_path(stem)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(clr("g", f"\n  ✓ Plot saved → {path.resolve()}"))

# ── Styling ──────────────────────────────────────────────────────────────────

STYLES = ["default", "seaborn-v0_8-darkgrid", "seaborn-v0_8-whitegrid",
          "ggplot", "fivethirtyeight", "bmh"]

def apply_style(choice: int) -> None:
    try:
        plt.style.use(STYLES[choice])
    except Exception:
        plt.style.use("default")

# ── Plot builders ─────────────────────────────────────────────────────────────

def build_line(df: pd.DataFrame, cols: list[str]) -> None:
    """Line plot – user picks X and one-or-more Y columns."""
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        print(clr("r", "  Need at least 2 numeric columns for a line plot."))
        return

    x_col = pick_column("X axis", num_cols)
    if x_col is None:
        return
    y_candidates = [c for c in num_cols if c != x_col]
    y_cols = pick_columns_multi("Y axis column(s)", y_candidates)
    if not y_cols:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for y in y_cols:
        ax.plot(df[x_col], df[y], marker="o", markersize=3, label=y)
    ax.set_xlabel(x_col)
    ax.set_ylabel(", ".join(y_cols))
    ax.set_title("Line Plot")
    ax.legend()
    save_and_announce(fig, f"line_{'_'.join(y_cols)}_vs_{x_col}")


def build_scatter(df: pd.DataFrame, cols: list[str]) -> None:
    """Scatter plot – user picks X and Y."""
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        print(clr("r", "  Need at least 2 numeric columns for a scatter plot."))
        return

    x_col = pick_column("X axis", num_cols)
    if x_col is None:
        return
    y_col = pick_column("Y axis", [c for c in num_cols if c != x_col])
    if y_col is None:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[x_col], df[y_col], alpha=0.7, edgecolors="none")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Scatter: {y_col} vs {x_col}")
    save_and_announce(fig, f"scatter_{y_col}_vs_{x_col}")


def build_bar(df: pd.DataFrame, cols: list[str]) -> None:
    """Bar chart – categorical or numeric X, numeric Y."""
    x_col = pick_column("X axis (category or index)", cols)
    if x_col is None:
        return
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != x_col]
    if not num_cols:
        print(clr("r", "  No remaining numeric columns for Y axis."))
        return
    y_col = pick_column("Y axis (numeric)", num_cols)
    if y_col is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Bar Chart: {y_col} by {x_col}")
    plt.xticks(rotation=45, ha="right")
    save_and_announce(fig, f"bar_{y_col}_by_{x_col}")


def build_histogram(df: pd.DataFrame, cols: list[str]) -> None:
    """Histogram of one or more numeric columns."""
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        print(clr("r", "  No numeric columns found."))
        return
    chosen = pick_columns_multi("Column(s) to histogram", num_cols)
    if not chosen:
        return

    raw = input(clr("h", "  Number of bins [default 30] › ")).strip()
    bins = int(raw) if raw.isdigit() else 30

    fig, ax = plt.subplots(figsize=(9, 5))
    for c in chosen:
        ax.hist(df[c].dropna(), bins=bins, alpha=0.6, label=c)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram")
    ax.legend()
    save_and_announce(fig, f"histogram_{'_'.join(chosen)}")


def build_box(df: pd.DataFrame, cols: list[str]) -> None:
    """Box plot of selected numeric columns."""
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        print(clr("r", "  No numeric columns found."))
        return
    chosen = pick_columns_multi("Column(s) for box plot", num_cols)
    if not chosen:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(chosen) * 1.5), 6))
    df[chosen].boxplot(ax=ax, grid=False)
    ax.set_title("Box Plot")
    ax.set_ylabel("Value")
    save_and_announce(fig, f"boxplot_{'_'.join(chosen)}")


# ── Presets ───────────────────────────────────────────────────────────────────

def preset_all_numeric_lines(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Preset: plot every numeric column against the first column (assumed index/time).
    """
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        print(clr("r", "  Need at least 2 numeric columns."))
        return
    x_col = num_cols[0]
    y_cols = num_cols[1:]
    fig, ax = plt.subplots(figsize=(12, 5))
    for y in y_cols:
        ax.plot(df[x_col], df[y], marker="o", markersize=2, label=y)
    ax.set_xlabel(x_col)
    ax.set_title(f"All numeric columns vs {x_col}")
    ax.legend(fontsize=7)
    save_and_announce(fig, "preset_all_numeric_lines")


def preset_correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> None:
    """Preset: correlation matrix heatmap of all numeric columns."""
    try:
        import numpy as np
    except ImportError:
        print(clr("r", "  numpy required for heatmap."))
        return

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        print(clr("r", "  Need at least 2 numeric columns."))
        return

    corr = df[num_cols].corr()
    n = len(num_cols)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(corr.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(n)); ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(num_cols, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Correlation Heatmap")
    save_and_announce(fig, "preset_correlation_heatmap")


def preset_summary_subplots(df: pd.DataFrame, cols: list[str]) -> None:
    """Preset: one histogram subplot per numeric column."""
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        print(clr("r", "  No numeric columns found."))
        return
    n = len(num_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for idx, col in enumerate(num_cols):
        ax = axes[idx // ncols][idx % ncols]
        ax.hist(df[col].dropna(), bins=25, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
    # hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("Distribution Summary", fontsize=12, y=1.01)
    fig.tight_layout()
    save_and_announce(fig, "preset_summary_histograms")

# ── Column inspector ──────────────────────────────────────────────────────────

def show_columns(df: pd.DataFrame) -> None:
    hr()
    print(clr("h", f"  Columns ({len(df.columns)} total), {len(df):,} rows\n"))
    for i, col in enumerate(df.columns, 1):
        dtype  = str(df[col].dtype)
        nulls  = df[col].isna().sum()
        unique = df[col].nunique()
        if pd.api.types.is_numeric_dtype(df[col]):
            extra = (f"min={df[col].min():.4g}  "
                     f"max={df[col].max():.4g}  "
                     f"mean={df[col].mean():.4g}")
        else:
            top = df[col].value_counts().index[0] if unique else "—"
            extra = f"top='{top}'"
        print(f"  {clr('y', str(i).rjust(3))}) {clr('c', col.ljust(30))} "
              f"{dtype.ljust(12)} nulls={nulls}  unique={unique}  {extra}")
    hr()

# ── File loader ───────────────────────────────────────────────────────────────

def load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        print(clr("g", f"  Loaded {path.name}  ({len(df):,} rows × {len(df.columns)} cols)"))
        return df
    except Exception as e:
        print(clr("r", f"  Error reading {path}: {e}"))
        return None

# ── Main flow ─────────────────────────────────────────────────────────────────

PRESET_MENU = [
    ("All numeric columns – line chart",          preset_all_numeric_lines),
    ("Correlation heatmap",                        preset_correlation_heatmap),
    ("Distribution summary (histograms grid)",     preset_summary_subplots),
]

CUSTOM_MENU = [
    ("Line plot",      build_line),
    ("Scatter plot",   build_scatter),
    ("Bar chart",      build_bar),
    ("Histogram",      build_histogram),
    ("Box plot",       build_box),
]

def main() -> None:
    print()
    hr("═")
    print(clr("h", "  📊  Data Plotter"))
    hr("═")

    # ── 1. Choose CSV ──────────────────────────────────────────────────────
    csv_files = sorted(INPUTS_DIR.glob("*.csv"))
    if not csv_files:
        print(clr("r", f"  No CSV files found in ./{INPUTS_DIR}/"))
        sys.exit(1)

    print(clr("c", f"\n  CSV files in ./{INPUTS_DIR}/\n"))
    file_idx = pick("Select a file", [f.name for f in csv_files])
    df = load_csv(csv_files[file_idx])
    if df is None:
        sys.exit(1)

    # ── 2. Style ───────────────────────────────────────────────────────────
    print(clr("c", "\n  Plot style\n"))
    style_idx = pick("Choose a style", STYLES)
    apply_style(style_idx)

    # ── 3. Column filter ───────────────────────────────────────────────────
    all_cols = list(df.columns)

    while True:
        hr()
        print(clr("c", "  Main menu\n"))
        top_menu = [
            "Show all columns / statistics",
            "Preset plots",
            "Custom plot",
            "Choose a different CSV",
            "Quit",
        ]
        choice = pick("What would you like to do?", top_menu)

        # ── Show columns ───────────────────────────────────────────────────
        if choice == 0:
            show_columns(df)

        # ── Presets ────────────────────────────────────────────────────────
        elif choice == 1:
            while True:
                hr()
                print(clr("c", "  Preset plots\n"))
                p_idx = pick("Select preset", [p[0] for p in PRESET_MENU], allow_back=True)
                if p_idx is None:
                    break

                # Optional column subset
                print(clr("c", "\n  Use all columns or a subset?\n"))
                sub = pick("Scope", ["All columns", "Choose a subset"])
                if sub == 1:
                    chosen = pick_columns_multi("Select columns to include", all_cols)
                    if chosen is None:
                        continue
                    sub_df = df[chosen]
                else:
                    sub_df = df

                PRESET_MENU[p_idx][1](sub_df, list(sub_df.columns))

        # ── Custom ─────────────────────────────────────────────────────────
        elif choice == 2:
            while True:
                hr()
                print(clr("c", "  Custom plot\n"))
                c_idx = pick("Select plot type", [c[0] for c in CUSTOM_MENU], allow_back=True)
                if c_idx is None:
                    break

                # Optional column subset
                print(clr("c", "\n  Use all columns or a subset?\n"))
                sub = pick("Scope", ["All columns", "Choose a subset"])
                if sub == 1:
                    chosen = pick_columns_multi("Select columns to include", all_cols)
                    if chosen is None:
                        continue
                    sub_df = df[chosen]
                else:
                    sub_df = df

                show_columns(sub_df)
                CUSTOM_MENU[c_idx][1](sub_df, list(sub_df.columns))

        # ── Change CSV ─────────────────────────────────────────────────────
        elif choice == 3:
            csv_files = sorted(INPUTS_DIR.glob("*.csv"))
            if not csv_files:
                print(clr("r", "  No CSV files found."))
                continue
            print(clr("c", f"\n  CSV files in ./{INPUTS_DIR}/\n"))
            file_idx = pick("Select a file", [f.name for f in csv_files])
            new_df = load_csv(csv_files[file_idx])
            if new_df is not None:
                df = new_df
                all_cols = list(df.columns)

        # ── Quit ───────────────────────────────────────────────────────────
        elif choice == 4:
            print(clr("g", "\n  Bye!\n"))
            sys.exit(0)


if __name__ == "__main__":
    main()
