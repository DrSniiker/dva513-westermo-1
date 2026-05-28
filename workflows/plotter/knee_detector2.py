import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
from kneed import KneeLocator

# --- Global Configurations ---
INPUT_DIR = 'inputs'
OUTPUT_DIR = 'outputs'

# Input file pattern search criteria
SPARSITY_FILE_PATTERN = '*sparsity_comparison.csv'
BENCHMARK_FILE_PATTERN = 'benchmark_results*.csv'

# Smoothing configuration
SMOOTHING_SIGMA = 1.0  # Increase to smooth more heavily, decrease for less smoothing
generated_outputs = [] # used for path print

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class DataPlotter:
    def __init__(self, inputs_dir):
        self.inputs_dir = inputs_dir
        self.sparsity_data = {}  
        self.benchmark_data = None
        self._load_and_process_data()

    def _load_and_process_data(self):
        # --- 1. Dynamically Load Sparsity Comparison Data ---
        sparsity_pattern = os.path.join(self.inputs_dir, SPARSITY_FILE_PATTERN)
        matched_sparsity_files = glob.glob(sparsity_pattern)
        
        if not matched_sparsity_files:
            raise FileNotFoundError(f"No files matching '{SPARSITY_FILE_PATTERN}' found in '{self.inputs_dir}/'")
            
        sparsity_csv = matched_sparsity_files[0]
        print(f"Loaded sparsity data from dynamically matched file: {os.path.basename(sparsity_csv)}")
        
        df_sparsity = pd.read_csv(sparsity_csv)
        
        # Separate data by Architecture into the struct dictionary
        for arch_id, group in df_sparsity.groupby('Architecture'):
            self.sparsity_data[arch_id] = group.copy()
            
        # --- 2. Dynamically Load and Parse Benchmark Results ---
        bench_pattern = os.path.join(self.inputs_dir, BENCHMARK_FILE_PATTERN)
        matched_bench_files = glob.glob(bench_pattern)
        
        if not matched_bench_files:
            raise FileNotFoundError(f"No files matching '{BENCHMARK_FILE_PATTERN}' found in '{self.inputs_dir}/'")
            
        benchmark_csv = matched_bench_files[0]
        print(f"Loaded benchmark data from dynamically matched file: {os.path.basename(benchmark_csv)}")
        
        df_bench = pd.read_csv(benchmark_csv)
        
        # Extract Architecture ID and Sparsity value from model names
        def parse_model_string(model_str):
            arch_match = re.search(r'Architecture(\d+)', model_str)
            sparsity_match = re.search(r'sparsity_([\d\.]+)\.pt', model_str)
            
            arch_id = int(arch_match.group(1)) if arch_match else None
            sparsity = float(sparsity_match.group(1)) if sparsity_match else None
            return pd.Series([arch_id, sparsity])
            
        df_bench[['extracted_architecture', 'extracted_sparsity']] = df_bench['model'].apply(parse_model_string)
        self.benchmark_data = df_bench

    def plot_sparsity_vs_mcc(self, save_filename='line_MCC_vs_Sparsity_1.png'):
        """Plots smoothed MCC vs Sparsity for all architectures and finds the knee point."""
        plt.figure(figsize=(12, 8)) 
        
        # Pre-defined offsets for each architecture to prevent overlapping text boxes
        label_offsets = {
            1: (-40, 45),   # Top-left
            2: (25, 45),    # Top-right
            3: (-60, -45),  # Bottom-left
            4: (25, -45),   # Bottom-right
            5: (-80, 15),   # Far-left
            6: (40, 15)     # Far-right
        }
        
        for arch_id, data in self.sparsity_data.items():
            data_sorted = data.sort_values(by='Sparsity_actual')
            
            x = data_sorted['Sparsity_actual'].values
            y = data_sorted['Honest-MCC'].values
            
            # 1. Apply Gaussian smoothing to the y-axis (MCC) data
            y_smoothed = scipy.ndimage.gaussian_filter1d(y, sigma=SMOOTHING_SIGMA)
            
            # 2. Plot using y_smoothed instead of raw y values
            line, = plt.plot(x, y_smoothed, marker='o', label=f'Architecture {arch_id}')
            
            # 3. Feed the smoothed y-values into the Kneedle algorithm
            try:
                kneedle = KneeLocator(
                    x, y_smoothed, 
                    curve='concave', 
                    direction='decreasing', 
                    S=1.0, 
                    online=True,           
                    interp_method='interp1d'
                )
                
                knee_x = kneedle.knee
                knee_y = kneedle.knee_y # This grabs the exact Y value on the smoothed curve
                
                if knee_x is not None and knee_y is not None:
                    # Mark the knee prominently directly on top of the smoothed curve
                    plt.plot(knee_x, knee_y, marker='*', markersize=15, color=line.get_color())
                    
                    offset = label_offsets.get(arch_id, (15, 15))
                    
                    plt.annotate(
                        f'Arch {arch_id}\nPruned: {knee_x * 100:.1f}%\nMCC: {knee_y:.3f}',
                        xy=(knee_x, knee_y),
                        xytext=offset,
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color=line.get_color(), lw=1.5, alpha=0.8),
                        fontsize=9,
                        color='black',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=line.get_color(), lw=2, alpha=0.9),
                        zorder=5 
                    )
            except Exception as e:
                print(f"Could not calculate knee for Architecture {arch_id}: {e}")
                     
        plt.title('MCC vs Sparsity')
        plt.xlabel('Sparsity')
        plt.ylabel('MCC')
        plt.legend(title="Architecture", loc="lower left")
        plt.grid(True)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

    def plot_latency_vs_sparsity(self, save_filename='latency_vs_sparsity_combined.png'):
        """Plots latency_ms vs extracted sparsity for each architecture."""
        plt.figure(figsize=(10, 6))
        
        for arch_id, group in self.benchmark_data.groupby('extracted_architecture'):
            group_sorted = group.sort_values(by='extracted_sparsity')
            
            # Plots the requested global latency column
            plt.plot(group_sorted['extracted_sparsity'], group_sorted['latency_ms'], 
                     marker='s', linestyle='--', label=f'Architecture {int(arch_id)}')

        plt.title('Latency vs Sparsity')
        plt.xlabel('Sparsity')
        plt.ylabel('Latency (ms)')
        plt.legend(title="Architecture")
        plt.grid(True)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    print("Initializing DataPlotter...")
    plotter = DataPlotter(inputs_dir=INPUT_DIR)
    
    # 1. Plot MCC vs Sparsity with pre-smoothing and Kneedle annotations
    plotter.plot_sparsity_vs_mcc(save_filename='line_MCC_vs_Sparsity_1.png')
    
    # 2. Plot Latency vs Sparsity using latency_ms
    plotter.plot_latency_vs_sparsity(save_filename='latency_vs_sparsity_combined.png')
    print("Done!")
