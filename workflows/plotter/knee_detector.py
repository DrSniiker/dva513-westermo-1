import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
generated_outputs = [] # used for path print

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

# Only process files matching this pattern
FILE_PATTERN = "*sparsity_comparison.csv"

DATASET_NAME = None
# Example:
# DATASET_NAME = "Ereno"

SMOOTH_WINDOW = 5
POLYORDER = 2

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PROCESS EACH CSV
# ============================================================

csv_files = glob.glob(
    os.path.join(INPUT_DIR, FILE_PATTERN)
)

for csv_file in csv_files:

    print(f"\nProcessing: {csv_file}")

    # --------------------------------------------------------
    # LOAD CSV
    # --------------------------------------------------------

    df = pd.read_csv(csv_file)

    # --------------------------------------------------------
    # OPTIONAL DATASET FILTER
    # --------------------------------------------------------

    if DATASET_NAME is not None:

        df = df[
            df["Dataset"] == DATASET_NAME
        ].copy()

    # --------------------------------------------------------
    # FIND ARCHITECTURES
    # --------------------------------------------------------

    architectures = sorted(
        df["Architecture"].dropna().unique()
    )

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    summary_rows = []

    # --------------------------------------------------------
    # PROCESS EACH ARCHITECTURE
    # --------------------------------------------------------

    for arch in architectures:

        curve = df[
            df["Architecture"] == arch
        ].copy()

        curve = curve[
            ["Sparsity_actual", "Honest-MCC"]
        ].dropna()

        if len(curve) < 5:
            print(f"Skipping Arch {arch}: not enough points")
            continue

        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        curve = curve.sort_values(
            "Sparsity_actual"
        )

        x = curve["Sparsity_actual"] \
            .astype(float) \
            .values

        y = curve["Honest-MCC"] \
            .astype(float) \
            .values

        # ----------------------------------------------------
        # SMOOTH CURVE
        # ----------------------------------------------------

        window = SMOOTH_WINDOW

        if window >= len(y):
            window = len(y) - 1

        # must be odd
        if window % 2 == 0:
            window -= 1

        if window < 3:
            window = 3

        y_smooth = savgol_filter(
            y,
            window_length=window,
            polyorder=POLYORDER
        )

        # ----------------------------------------------------
        # KNEE DETECTION
        # ----------------------------------------------------

        p1 = np.array([
            x[0],
            y_smooth[0]
        ])

        p2 = np.array([
            x[-1],
            y_smooth[-1]
        ])

        distances = []

        for xi, yi in zip(x, y_smooth):

            p = np.array([xi, yi])

            dist = np.abs(
                np.cross(
                    p2 - p1,
                    p1 - p
                )
            ) / np.linalg.norm(p2 - p1)

            distances.append(dist)

        knee_idx = int(
            np.argmax(distances)
        )

        knee_x = x[knee_idx]
        knee_y = y_smooth[knee_idx]

        # ----------------------------------------------------
        # SAVE SUMMARY
        # ----------------------------------------------------

        summary_rows.append([
            arch,
            knee_x,
            knee_y
        ])

        print(
            f"Arch {arch}: "
            f"knee at "
            f"({knee_x:.3f}, {knee_y:.3f})"
        )

        # ----------------------------------------------------
        # PLOT SMOOTHED CURVE
        # ----------------------------------------------------

        ax.plot(
            x,
            y_smooth,
            linewidth=2,
            label=f"Arch {arch}"
        )

        # ----------------------------------------------------
        # PLOT KNEE POINT
        # ----------------------------------------------------

        ax.scatter(
            [knee_x],
            [knee_y],
            s=120
        )

        ax.annotate(
            f"A{arch}\n{knee_x:.2f}",
            (knee_x, knee_y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8
        )

    # --------------------------------------------------------
    # FINAL GRAPH FORMATTING
    # --------------------------------------------------------

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("MCC")

    title = os.path.basename(csv_file)
    title = title.replace(".csv", "")

    ax.set_title(
        f"{title} — Multi-Architecture Knee Comparison"
    )

    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE PNG
    # --------------------------------------------------------

    output_png = os.path.join(
        OUTPUT_DIR,
        f"{title}_knees.png"
    )

    plt.savefig(
        output_png,
        dpi=300
    )

    generated_outputs.append(output_png)

    plt.close()

    # --------------------------------------------------------
    # SAVE SUMMARY CSV
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "Architecture",
            "Knee_Sparsity",
            "Knee_MCC"
        ]
    )

    output_summary = os.path.join(
        OUTPUT_DIR,
        f"{title}_knees.csv"
    )

    summary_df.to_csv(
        output_summary,
        index=False
    )

    generated_outputs.append(output_summary)

    print(f"Saved graph: {output_png}")
    print(f"Saved summary: {output_summary}")

print("\nDone.")
print("\nGenerated files:")

for path in generated_outputs:
    print(path)
