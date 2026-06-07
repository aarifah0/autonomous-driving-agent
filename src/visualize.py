import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_training_results(log_dir="logs", window_size=10):
    """
    Parses the Monitor CSV log from training and generates publication-quality figures for
    episodic rewards and lengths with variance shading.
    """
    monitor_path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_path):
        print(f"Error: {monitor_path} not found. Cannot plot results.")
        return
        
    print(f"Loading training logs from {monitor_path}...")
    
    # Read SB3 monitor CSV (skips the first row which is metadata)
    df = pd.read_csv(monitor_path, skiprows=1)
    
    if len(df) == 0:
        print("Warning: The monitor log file is empty. Perhaps the training is still running?")
        return
        
    # Calculate cumulative steps
    df['steps'] = df['l'].cumsum()
    
    # Calculate rolling metrics
    df['reward_smooth'] = df['r'].rolling(window=window_size, min_periods=1).mean()
    df['reward_std'] = df['r'].rolling(window=window_size, min_periods=1).std().fillna(0)
    df['length_smooth'] = df['l'].rolling(window=window_size, min_periods=1).mean()
    
    # Plot 1: Episode Reward Learning Curve
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Main line
    plt.plot(df['steps'], df['reward_smooth'], color='#1f77b4', linewidth=2, label=f'Rolling Mean (w={window_size})')
    # Raw scatter points (semi-transparent)
    plt.scatter(df['steps'], df['r'], color='#1f77b4', alpha=0.15, s=15, label='Raw Episode Reward')
    # Standard deviation shading
    plt.fill_between(
        df['steps'],
        df['reward_smooth'] - df['reward_std'],
        df['reward_smooth'] + df['reward_std'],
        color='#1f77b4',
        alpha=0.15,
        edgecolor='none',
        label='Variance (±1 SD)'
    )
    
    plt.title('Autonomous Agent Training Evolution: Episodic Reward', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Episodic Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    reward_plot_path = os.path.join(log_dir, "training_reward.png")
    plt.savefig(reward_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved reward plot to: {reward_plot_path}")
    
    # Plot 2: Episode Length Evolution
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(df['steps'], df['length_smooth'], color='#2ca02c', linewidth=2, label=f'Rolling Mean (w={window_size})')
    plt.scatter(df['steps'], df['l'], color='#2ca02c', alpha=0.15, s=15, label='Raw Episode Length')
    
    plt.title('Autonomous Agent Training Evolution: Episode Length', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Episode Duration (Steps)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    length_plot_path = os.path.join(log_dir, "training_length.png")
    plt.savefig(length_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved episode length plot to: {length_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize training results from monitor CSV")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory where monitor.csv is stored")
    parser.add_argument("--window", type=int, default=10, help="Smoothing window size")
    args = parser.parse_args()
    
    plot_training_results(args.log_dir, args.window)

if __name__ == "__main__":
    main()
