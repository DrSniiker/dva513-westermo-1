"""
data_plotter/main.py
--------------------
Interactive CLI tool that reads CSV files from ./inputs/ and generates
matplotlib plots saved to ./outputs/.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # never open a window
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────

INPUTS_DIR  = Path("inputs")
OUTPUTS_DIR = Path("outputs")

# ── Helpers ──────────────────────────────────────────────────────────────────

def clr(code: str, text: str) -> str:
    codes = {"h": "\033[1m", "b": "\033[94m", "g": "\033[92m",
             "y": "\033[93m", "r": "\033[91m", "c": "\033[96m", "x": "\033[0m"}
    return f"{codes.get(code,'')}{text}{codes['x']}"


def hr(char: str = "─", width: int = 60) -> None:
    print(clr("b", char * width))


def pick(prompt: str, options: list, allow_back: bool = False) -> Optional[int]:
    """Numbered menu – returns 0-based index, or None for back/cancel."""
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
    idx = pick(prompt, columns, allow_back=allow_back)
    return None if idx is None else columns[idx]


def pick_columns_multi(prompt: str, columns: list[str]) -> Optional[list[str]]:
    """Pick ≥1 columns by number, ranges, or comma-separated (e.g. 1,3,5-7)."""
    print(f"\n  {clr('c', 'Available columns:')}")
    for i, c in enumerate(columns, 1):
        print(f"    {clr('y', str(i))}) {c}")
    print(f"    {clr('y', '0')}) ← Back")
    while True:
        raw = input(clr("h", f"\n{prompt} (e.g. 1,3,5-7) › ")).strip()
        if raw == "0":
            return None
        selected: set[int] = set()
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


def pick_files_multi(prompt: str, files: list[Path]) -> Optional[list[Path]]:
    """Pick ≥1 files by number / range / comma-list."""
    names = [f.name for f in files]
    print(f"\n  {clr('c', 'Available files:')}")
    for i, n in enumerate(names, 1):
        print(f"    {clr('y', str(i))}) {n}")
    print(f"    {clr('y', '0')}) ← Back")
    while True:
        raw = input(clr("h", f"\n{prompt} (e.g. 1,3 or 1-4) › ")).strip()
        if raw == "0":
            return None
        selected: set[int] = set()
        try:
            for part in raw.split(","):
                part = part.strip()
                if "-" in part:
                    a, b = part.split("-")
                    selected.update(range(int(a), int(b) + 1))
                else:
                    selected.add(int(part))
            chosen = [files[i - 1] for i in sorted(selected) if 1 <= i <= len(files)]
            if chosen:
                return chosen
        except Exception:
            pass
        print(clr("r", "  Invalid input – try again."))


def safe_output_path(stem: str, suffix: str = ".png") -> Path:
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

# ── Styling ───────────────────────────────────────────────────────────────────

STYLES = ["default", "seaborn-v0_8-darkgrid", "seaborn-v0_8-whitegrid",
          "ggplot", "fivethirtyeight", "bmh"]

def apply_style(choice: int) -> None:
    try:
        plt.style.use(STYLES[choice])
    except Exception:
        plt.style.use("default")

# ── Group-by helpers ─────────────────────────────────────────────────────────

_MARKERS   = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+"]
_LINESTYLE = ["-", "--", "-.", ":"]


def auto_group_col(df: pd.DataFrame, exclude: list[str]) -> Optional[str]:
    """
    Silently detect the best column to split series by.
    Picks the non-numeric column with the fewest unique values (>=2, <=50).
    Returns None when no suitable column exists (all rows -> one series).
    """
    candidates = [
        c for c in df.columns
        if c not in exclude
        and not pd.api.types.is_numeric_dtype(df[c])
        and 2 <= df[c].nunique() <= 50
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: df[c].nunique())


def draw_series(ax: plt.Axes, x_data, y_data, label: str,
                series_idx: int, kind: str) -> None:
    """Draw one series onto ax with consistent cycling markers/linestyles."""
    marker    = _MARKERS[series_idx % len(_MARKERS)]
    linestyle = _LINESTYLE[series_idx % len(_LINESTYLE)]
    if kind == "line":
        order = x_data.argsort() if hasattr(x_data, "argsort") else range(len(x_data))
        ax.plot(x_data.iloc[order] if hasattr(x_data, "iloc") else x_data,
                y_data.iloc[order] if hasattr(y_data, "iloc") else y_data,
                marker=marker, markersize=4, linestyle=linestyle,
                linewidth=1.6, label=label)
    else:  # scatter
        ax.scatter(x_data, y_data, marker=marker, s=45,
                   alpha=0.75, edgecolors="none", label=label)


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

# ── Single-file plot builders ─────────────────────────────────────────────────

def build_line(df: pd.DataFrame, cols: list[str]) -> None:
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        print(clr("r", "  Need at least 2 numeric columns for a line plot.")); return
    x_col = pick_column("X axis", num_cols)
    if x_col is None: return
    y_cols = pick_columns_multi("Y axis column(s)", [c for c in num_cols if c != x_col])
    if not y_cols: return

    group_col = auto_group_col(df, exclude=[x_col] + y_cols)

    fig, ax = plt.subplots(figsize=(10, 5))
    series_idx = 0

    if group_col:
        groups = sorted(df[group_col].dropna().unique(), key=str)
        print(clr("g", f"\n  Splitting into {len(groups)} series by '{group_col}'"))
        for group_val in groups:
            mask   = df[group_col] == group_val
            label  = str(group_val)
            for y in y_cols:
                full_label = f"{label} – {y}" if len(y_cols) > 1 else label
                draw_series(ax, df.loc[mask, x_col], df.loc[mask, y],
                            full_label, series_idx, "line")
                series_idx += 1
        title = f"Line Plot  |  grouped by {group_col}"
    else:
        for y in y_cols:
            draw_series(ax, df[x_col], df[y], y, series_idx, "line")
            series_idx += 1
        title = "Line Plot"

    ax.set_xlabel(x_col); ax.set_ylabel(", ".join(y_cols))
    ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout()
    stem = f"line_{'_'.join(y_cols)}_vs_{x_col}" + (f"_by_{group_col}" if group_col else "")
    save_and_announce(fig, stem)


def build_scatter(df: pd.DataFrame, cols: list[str]) -> None:
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        print(clr("r", "  Need at least 2 numeric columns for a scatter plot.")); return
    x_col = pick_column("X axis", num_cols)
    if x_col is None: return
    y_col = pick_column("Y axis", [c for c in num_cols if c != x_col])
    if y_col is None: return

    group_col = auto_group_col(df, exclude=[x_col, y_col])

    fig, ax = plt.subplots(figsize=(8, 6))
    series_idx = 0

    if group_col:
        groups = sorted(df[group_col].dropna().unique(), key=str)
        print(clr("g", f"\n  Splitting into {len(groups)} series by '{group_col}'"))
        for group_val in groups:
            mask = df[group_col] == group_val
            draw_series(ax, df.loc[mask, x_col], df.loc[mask, y_col],
                        str(group_val), series_idx, "scatter")
            series_idx += 1
        title = f"Scatter: {y_col} vs {x_col}  |  grouped by {group_col}"
    else:
        draw_series(ax, df[x_col], df[y_col], y_col, 0, "scatter")
        title = f"Scatter: {y_col} vs {x_col}"

    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout()
    stem = f"scatter_{y_col}_vs_{x_col}" + (f"_by_{group_col}" if group_col else "")
    save_and_announce(fig, stem)


def build_bar(df: pd.DataFrame, cols: list[str]) -> None:
    x_col = pick_column("X axis (category or index)", cols)
    if x_col is None: return
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c != x_col]
    if not num_cols:
        print(clr("r", "  No remaining numeric columns for Y axis.")); return
    y_col = pick_column("Y axis (numeric)", num_cols)
    if y_col is None: return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(f"Bar Chart: {y_col} by {x_col}")
    plt.xticks(rotation=45, ha="right")
    save_and_announce(fig, f"bar_{y_col}_by_{x_col}")


def build_histogram(df: pd.DataFrame, cols: list[str]) -> None:
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        print(clr("r", "  No numeric columns found.")); return
    chosen = pick_columns_multi("Column(s) to histogram", num_cols)
    if not chosen: return
    raw = input(clr("h", "  Number of bins [default 30] › ")).strip()
    bins = int(raw) if raw.isdigit() else 30
    fig, ax = plt.subplots(figsize=(9, 5))
    for c in chosen:
        ax.hist(df[c].dropna(), bins=bins, alpha=0.6, label=c)
    ax.set_xlabel("Value"); ax.set_ylabel("Frequency")
    ax.set_title("Histogram"); ax.legend()
    save_and_announce(fig, f"histogram_{'_'.join(chosen)}")


def build_box(df: pd.DataFrame, cols: list[str]) -> None:
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        print(clr("r", "  No numeric columns found.")); return
    chosen = pick_columns_multi("Column(s) for box plot", num_cols)
    if not chosen: return
    fig, ax = plt.subplots(figsize=(max(6, len(chosen) * 1.5), 6))
    df[chosen].boxplot(ax=ax, grid=False)
    ax.set_title("Box Plot"); ax.set_ylabel("Value")
    save_and_announce(fig, f"boxplot_{'_'.join(chosen)}")

# ── Presets ───────────────────────────────────────────────────────────────────

def preset_all_numeric_lines(df: pd.DataFrame, cols: list[str]) -> None:
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        print(clr("r", "  Need at least 2 numeric columns.")); return
    x_col  = num_cols[0]
    y_cols = num_cols[1:]
    fig, ax = plt.subplots(figsize=(12, 5))
    for y in y_cols:
        ax.plot(df[x_col], df[y], marker="o", markersize=2, label=y)
    ax.set_xlabel(x_col); ax.set_title(f"All numeric columns vs {x_col}")
    ax.legend(fontsize=7)
    save_and_announce(fig, "preset_all_numeric_lines")


def preset_correlation_heatmap(df: pd.DataFrame, cols: list[str]) -> None:
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        print(clr("r", "  Need at least 2 numeric columns.")); return
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
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        print(clr("r", "  No numeric columns found.")); return
    n = len(num_cols); ncols = min(3, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    for idx, col in enumerate(num_cols):
        ax = axes[idx // ncols][idx % ncols]
        ax.hist(df[col].dropna(), bins=25, color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_title(col, fontsize=9); ax.set_xlabel("Value", fontsize=7); ax.set_ylabel("Count", fontsize=7)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("Distribution Summary", fontsize=12, y=1.01); fig.tight_layout()
    save_and_announce(fig, "preset_summary_histograms")

# ── Multi-file overlay ────────────────────────────────────────────────────────

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _ask_label(default: str) -> str:
    raw = input(clr("h", f"  Series label [{default}] › ")).strip()
    return raw if raw else default


def _ask_title(default: str) -> str:
    raw = input(clr("h", f"  Plot title [{default}] › ")).strip()
    return raw if raw else default


def build_multi_file_overlay(csv_files: list[Path]) -> None:
    """
    Overlay series from multiple CSV files onto one graph.

    Flow
    ────
    1. Pick files (multi-select).
    2. Load them; detect whether all share the same column schema.
    3. Shared schema  → ask for X and Y once, then optionally a group-by column
                        so each (file × group) becomes its own series.
       Mixed schemas  → ask per-file which columns map to X and Y.
    4. Name each series (default = filename stem, or "file – group_value").
    5. Pick plot type: line or scatter.
    6. Optional custom title.
    7. Save.
    """
    hr()
    print(clr("h", "  Multi-file overlay\n"))

    # ── 1. Pick files ──────────────────────────────────────────────────────
    selected_paths = pick_files_multi("Select files to overlay", csv_files)
    if not selected_paths:
        return
    if len(selected_paths) < 2:
        print(clr("y", "  Tip: select ≥2 files to actually overlay; continuing with 1."))

    # ── 2. Load ────────────────────────────────────────────────────────────
    entries: list[tuple[Path, pd.DataFrame]] = []
    for p in selected_paths:
        df = load_csv(p)
        if df is not None:
            entries.append((p, df))
    if not entries:
        print(clr("r", "  No files could be loaded.")); return

    # ── 3. Column mapping ──────────────────────────────────────────────────
    num_col_sets  = [set(_numeric_cols(df)) for _, df in entries]
    common_num    = sorted(set.intersection(*num_col_sets)) if num_col_sets else []
    shared_schema = bool(common_num)

    # series entries: {df, x, y, label}
    series: list[dict] = []

    if shared_schema:
        print(clr("g", f"\n  ✓ All files share {len(common_num)} numeric column(s) → "
                       "pick axes once.\n"))
        x_col = pick_column("X axis (shared)", common_num)
        if x_col is None: return
        y_candidates = [c for c in common_num if c != x_col]
        if not y_candidates:
            print(clr("r", "  No remaining numeric columns for Y axis.")); return
        y_col = pick_column("Y axis (shared)", y_candidates)
        if y_col is None: return

        # Group-by: first check if any file has a useful categorical column
        # Use columns from the first file as a representative sample
        first_df = entries[0][1]
        group_col = auto_group_col(first_df, exclude=[x_col, y_col])

        print(clr("c", "\n  Name each series (Enter = keep default):\n"))
        for path, df in entries:
            if group_col and group_col in df.columns:
                groups = sorted(df[group_col].dropna().unique(), key=str)
                for gval in groups:
                    mask    = df[group_col] == gval
                    default = f"{path.stem} – {gval}"
                    label   = _ask_label(default)
                    series.append({"df": df[mask].reset_index(drop=True),
                                   "x": x_col, "y": y_col, "label": label})
            else:
                label = _ask_label(path.stem)
                series.append({"df": df, "x": x_col, "y": y_col, "label": label})

    else:
        print(clr("y", "\n  Files have different columns → map axes per file.\n"))
        for path, df in entries:
            hr("·")
            print(clr("h", f"  {path.name}\n"))
            show_columns(df)
            nc = _numeric_cols(df)
            if len(nc) < 2:
                print(clr("r", f"  {path.name} has fewer than 2 numeric cols – skipping.")); continue
            x_col = pick_column(f"X axis for {path.name}", nc)
            if x_col is None: continue
            y_col = pick_column(f"Y axis for {path.name}", [c for c in nc if c != x_col])
            if y_col is None: continue

            # Per-file group-by
            group_col = auto_group_col(df, exclude=[x_col, y_col])
            if group_col:
                groups = sorted(df[group_col].dropna().unique(), key=str)
                for gval in groups:
                    mask    = df[group_col] == gval
                    default = f"{path.stem} – {gval}"
                    label   = _ask_label(default)
                    series.append({"df": df[mask].reset_index(drop=True),
                                   "x": x_col, "y": y_col, "label": label})
            else:
                label = _ask_label(path.stem)
                series.append({"df": df, "x": x_col, "y": y_col, "label": label})

    if not series:
        print(clr("r", "  No valid series defined.")); return

    # ── 4. Plot type ───────────────────────────────────────────────────────
    print(clr("c", "\n  Plot type\n"))
    plot_kind = "line" if pick("Choose", ["Line", "Scatter"]) == 0 else "scatter"

    # ── 5. Title ───────────────────────────────────────────────────────────
    x_names = list(dict.fromkeys(s["x"] for s in series))
    y_names = list(dict.fromkeys(s["y"] for s in series))
    default_title = f"{' / '.join(y_names)} vs {' / '.join(x_names)}"
    print()
    title = _ask_title(default_title)

    # ── 6. Draw ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, s in enumerate(series):
        draw_series(ax, s["df"][s["x"]], s["df"][s["y"]], s["label"], i, plot_kind)

    ax.set_xlabel(x_names[0] if len(x_names) == 1 else ", ".join(x_names))
    ax.set_ylabel(y_names[0] if len(y_names) == 1 else ", ".join(y_names))
    ax.set_title(title)
    ax.legend(framealpha=0.8, fontsize=8)
    fig.tight_layout()

    stem = f"overlay_{plot_kind}_{'_'.join(y_names)}_vs_{'_'.join(x_names)}"
    stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem)[:80]
    save_and_announce(fig, stem)

# ── Menus ─────────────────────────────────────────────────────────────────────

PRESET_MENU = [
    ("All numeric columns – line chart",      preset_all_numeric_lines),
    ("Correlation heatmap",                   preset_correlation_heatmap),
    ("Distribution summary (histogram grid)", preset_summary_subplots),
]

CUSTOM_MENU = [
    ("Line plot",    build_line),
    ("Scatter plot", build_scatter),
    ("Bar chart",    build_bar),
    ("Histogram",    build_histogram),
    ("Box plot",     build_box),
]

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    hr("═")
    print(clr("h", "  📊  Data Plotter"))
    hr("═")

    # ── Discover CSV files ─────────────────────────────────────────────────
    csv_files = sorted(INPUTS_DIR.glob("*.csv"))
    if not csv_files:
        print(clr("r", f"  No CSV files found in ./{INPUTS_DIR}/")); sys.exit(1)

    # ── Initial single-file selection ──────────────────────────────────────
    print(clr("c", f"\n  CSV files in ./{INPUTS_DIR}/\n"))
    file_idx = pick("Select a file to work with", [f.name for f in csv_files])
    df = load_csv(csv_files[file_idx])
    if df is None: sys.exit(1)

    # ── Style ──────────────────────────────────────────────────────────────
    print(clr("c", "\n  Plot style\n"))
    apply_style(pick("Choose a style", STYLES))

    all_cols = list(df.columns)

    # ── Main loop ──────────────────────────────────────────────────────────
    while True:
        hr()
        print(clr("c", "  Main menu\n"))
        top_menu = [
            "Show all columns / statistics",
            "Preset plots",
            "Custom plot",
            "Multi-file overlay plot  ← overlay series from different files",
            "Choose a different CSV",
            "Quit",
        ]
        choice = pick("What would you like to do?", top_menu)

        # Show columns
        if choice == 0:
            show_columns(df)

        # Presets
        elif choice == 1:
            while True:
                hr()
                print(clr("c", "  Preset plots\n"))
                p_idx = pick("Select preset", [p[0] for p in PRESET_MENU], allow_back=True)
                if p_idx is None: break
                print(clr("c", "\n  Use all columns or a subset?\n"))
                sub_df = df
                if pick("Scope", ["All columns", "Choose a subset"]) == 1:
                    chosen = pick_columns_multi("Select columns to include", all_cols)
                    if chosen is None: continue
                    sub_df = df[chosen]
                PRESET_MENU[p_idx][1](sub_df, list(sub_df.columns))

        # Custom single-file
        elif choice == 2:
            while True:
                hr()
                print(clr("c", "  Custom plot\n"))
                c_idx = pick("Select plot type", [c[0] for c in CUSTOM_MENU], allow_back=True)
                if c_idx is None: break
                print(clr("c", "\n  Use all columns or a subset?\n"))
                sub_df = df
                if pick("Scope", ["All columns", "Choose a subset"]) == 1:
                    chosen = pick_columns_multi("Select columns to include", all_cols)
                    if chosen is None: continue
                    sub_df = df[chosen]
                show_columns(sub_df)
                CUSTOM_MENU[c_idx][1](sub_df, list(sub_df.columns))

        # Multi-file overlay
        elif choice == 3:
            csv_files = sorted(INPUTS_DIR.glob("*.csv"))
            if not csv_files:
                print(clr("r", "  No CSV files found.")); continue
            build_multi_file_overlay(csv_files)

        # Change CSV
        elif choice == 4:
            csv_files = sorted(INPUTS_DIR.glob("*.csv"))
            if not csv_files:
                print(clr("r", "  No CSV files found.")); continue
            print(clr("c", f"\n  CSV files in ./{INPUTS_DIR}/\n"))
            file_idx = pick("Select a file", [f.name for f in csv_files])
            new_df = load_csv(csv_files[file_idx])
            if new_df is not None:
                df = new_df
                all_cols = list(df.columns)

        # Quit
        elif choice == 5:
            print(clr("g", "\n  Bye!\n")); sys.exit(0)


if __name__ == "__main__":
    main()
