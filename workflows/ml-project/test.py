#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generate x values
x = np.linspace(0, 10, 100)

# 2. Create a sine wave with noise
y = np.sin(x) + np.random.normal(0, 0.2, size=len(x))

# 3. Put into a DataFrame (just to use pandas)
df = pd.DataFrame({
    "x": x,
    "y": y
})

# 4. Compute rolling average (smoothing)
df["y_smooth"] = df["y"].rolling(window=5).mean()

# 5. Plot
plt.figure()
plt.plot(df["x"], df["y"], label="Noisy signal", alpha=0.5)
plt.plot(df["x"], df["y_smooth"], label="Smoothed signal", linewidth=2)

plt.title("Noisy Sine Wave with Smoothing")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.savefig("plot.png")
