"""
Visualize TensorBoard CSV logs generated during training.
Plots custom metrics and training progress from the TensorBoard log directory.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Path to TensorBoard CSV logs (stable-baselines3 writes to logs/tensorboard/)
    log_dir = Path(__file__).parent.parent / "logs" / "tensorboard"
    
    # Find the most recent CSV file
    csv_files = list(log_dir.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV log files found in {log_dir}")
        print("Train the model first with: python src/train.py")
        return
    
    csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading log data from {csv_path}")
    
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Plot mean episode reward if available
    plt.figure(figsize=(10, 6))
    if "time/total_timesteps" in data.columns and "rollout/ep_rew_mean" in data.columns:
        plt.plot(data["time/total_timesteps"], data["rollout/ep_rew_mean"], 
                 label="Mean Episode Reward", linewidth=2, color="#1f77b4")
        plt.xlabel("Total Timesteps", fontsize=12)
        plt.ylabel("Mean Episode Reward", fontsize=12)
        plt.title("Training Progress: Episode Reward vs Timesteps", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        out_path = log_dir / "reward_vs_timesteps.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved reward plot to {out_path}")
    else:
        print("Required columns not found in log file.")
    
    # Plot custom metrics if present
    custom_metrics = [col for col in data.columns if col.startswith("custom_metrics/")]
    for metric in custom_metrics:
        plt.figure(figsize=(10, 6))
        metric_name = metric.split('/')[-1]
        if "time/total_timesteps" in data.columns:
            plt.plot(data["time/total_timesteps"], data[metric], 
                    linewidth=2, color="#2ca02c")
            plt.xlabel("Total Timesteps", fontsize=12)
        else:
            plt.plot(data[metric], linewidth=2, color="#2ca02c")
            plt.xlabel("Episode", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f"{metric_name} over Training", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        out_path = log_dir / f"{metric_name}_vs_timesteps.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved {metric_name} plot to {out_path}")
    
    print("\nTensorBoard visualization complete!")

if __name__ == "__main__":
    main()
