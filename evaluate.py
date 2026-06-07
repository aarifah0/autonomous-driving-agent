import os
import yaml
import argparse
import numpy as np
import glob
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from moviepy.editor import VideoFileClip

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment import make_highway_env
from agent import load_agent

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent on highway-env")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model/best_model.zip", help="Path to trained PPO model zip")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--record", action="store_true", help="Whether to record driving video")
    parser.add_argument("--video_dir", type=str, default="videos", help="Directory to save evaluation videos")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Check if we should fallback to final model if best model doesn't exist
    if not os.path.exists(args.model_path):
        fallback = "checkpoints/highway_ppo_final.zip"
        if os.path.exists(fallback):
            print(f"Model at '{args.model_path}' not found. Falling back to final model: '{fallback}'")
            args.model_path = fallback
        else:
            # Look for any checkpoint in the checkpoints folder
            checkpoints = glob.glob("checkpoints/highway_ppo_model_*.zip")
            if checkpoints:
                checkpoints.sort(key=os.path.getmtime)
                fallback = checkpoints[-1]
                print(f"Model at '{args.model_path}' not found. Falling back to latest checkpoint: '{fallback}'")
                args.model_path = fallback
            else:
                raise FileNotFoundError(f"No trained PPO agent checkpoint found at {args.model_path} or fallbacks.")
                
    # Create environment
    print(f"Creating evaluation environment...")
    render_mode = "rgb_array" if args.record else None
    env = make_highway_env(args.config, render_mode=render_mode)
    
    # Wrap for video recording if requested
    if args.record:
        os.makedirs(args.video_dir, exist_ok=True)
        # Record all episodes for evaluation visualization
        env = RecordVideo(
            env,
            video_folder=args.video_dir,
            episode_trigger=lambda e: True,
            name_prefix="eval-video"
        )
        print(f"Video recording enabled. Output folder: {args.video_dir}/")
        
    # Load model
    print(f"Loading trained agent from {args.model_path}...")
    model = load_agent(args.model_path, env=env)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    episode_speeds = []
    episode_collisions = []
    episode_lanes = []
    
    print(f"Starting evaluation for {args.episodes} episodes...")
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        ep_reward = 0.0
        steps = 0
        speeds = []
        collisions = []
        lanes = []
        
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            ep_reward += reward
            steps += 1
            
            # Extract tracking components
            if "custom_reward_components" in info:
                comp = info["custom_reward_components"]
                speeds.append(comp["speed"])
                collisions.append(comp["collision"])
                lanes.append(comp["lane"])
                
        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)
        
        # Calculate averages for this episode
        episode_speeds.append(sum(speeds) / len(speeds) if speeds else 0.0)
        episode_collisions.append(1.0 if any(c > 0.0 for c in collisions) else 0.0)
        episode_lanes.append(sum(lanes) / len(lanes) if lanes else 0.0)
        
        print(f"Episode {ep+1:02d}: Steps={steps:3d} | Reward={ep_reward:7.2f} | Avg Speed Ratio={episode_speeds[-1]:5.2%}"
              f" | Lane Discipline={episode_lanes[-1]:5.2%} | Crashed={bool(episode_collisions[-1])}")
              
    # Summary of metrics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_speed_ratio = np.mean(episode_speeds)
    crash_rate = np.mean(episode_collisions)
    survival_rate = 1.0 - crash_rate
    avg_lane_discipline = np.mean(episode_lanes)
    
    print("\n" + "="*50)
    print("                 EVALUATION REPORT")
    print("="*50)
    print(f"Total Episodes evaluated:  {args.episodes}")
    print(f"Average Episode Reward:   {avg_reward:7.2f} +/- {std_reward:.2f}")
    print(f"Average Episode Length:   {avg_length:7.1f} steps")
    print(f"Average Speed Ratio:       {avg_speed_ratio:7.2%}")
    print(f"Lane Discipline Accuracy:  {avg_lane_discipline:7.2%}")
    print(f"Survival Rate:            {survival_rate:7.2%}")
    print(f"Crash Rate:               {crash_rate:7.2%}")
    print("="*50)
    
    # Save metrics to YAML
    metrics = {
        "episodes": args.episodes,
        "avg_reward": float(avg_reward),
        "std_reward": float(std_reward),
        "avg_length": float(avg_length),
        "avg_speed_ratio": float(avg_speed_ratio),
        "lane_discipline_accuracy": float(avg_lane_discipline),
        "survival_rate": float(survival_rate),
        "crash_rate": float(crash_rate)
    }
    
    metrics_path = os.path.join(os.path.dirname(args.model_path), "evaluation_metrics.yaml")
    with open(metrics_path, "w") as f:
        yaml.safe_dump(metrics, f)
    print(f"Evaluation metrics written to {metrics_path}")
    
    # Close environment
    env.close()
    
    # Post-process video to GIF if recording was enabled
    if args.record:
        mp4_files = glob.glob(os.path.join(args.video_dir, "*.mp4"))
        if mp4_files:
            # Sort by modification time to get the latest video
            mp4_files.sort(key=os.path.getmtime)
            latest_mp4 = mp4_files[-1]
            gif_path = os.path.join(args.video_dir, "evaluation.gif")
            print(f"Converting evaluation video to GIF: {latest_mp4} -> {gif_path}")
            try:
                clip = VideoFileClip(latest_mp4)
                # Resize to width 480px and limit frame rate to 10fps for a lightweight, smooth GIF
                resized_clip = clip.resize(width=480)
                resized_clip.write_gif(gif_path, fps=10, opt="colors")
                clip.close()
                resized_clip.close()
                print(f"Evaluation GIF successfully saved to: {gif_path}")
            except Exception as e:
                print(f"Error converting MP4 to GIF: {e}")
        else:
            print("No video files found to convert to GIF.")

if __name__ == "__main__":
    main()
