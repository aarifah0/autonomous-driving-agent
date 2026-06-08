


import os
# Optimise CPU core usage for PyTorch on laptop CPUs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import yaml
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback

from environment import make_highway_env
from agent import create_ppo_agent, save_agent

# Adjust paths for config and checkpoint directories when running from src/
import sys
from pathlib import Path
if "src" in Path(__file__).parts:
    os.chdir(Path(__file__).parent.parent)

class CustomMetricCallback(BaseCallback):
    """
    Custom callback to extract reward components from steps and log averages to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.speeds = []
        self.collisions = []
        self.lanes = []

    def _on_step(self) -> bool:
        # Loop through info dicts for vectorized envs
        infos = self.locals.get("infos", [])
        for info in infos:
            if "custom_reward_components" in info:
                comp = info["custom_reward_components"]
                self.speeds.append(comp["speed"])
                self.collisions.append(comp["collision"])
                self.lanes.append(comp["lane"])
            
            # Check if episode ended (Monitor wrapper adds 'episode' key)
            if "episode" in info:
                avg_speed = sum(self.speeds) / len(self.speeds) if self.speeds else 0.0
                collision_occurred = 1.0 if any(c > 0.0 for c in self.collisions) else 0.0
                avg_lane = sum(self.lanes) / len(self.lanes) if self.lanes else 0.0
                
                # Record to Tensorboard / Logger
                self.logger.record("custom_metrics/avg_speed_ratio", avg_speed)
                self.logger.record("custom_metrics/collision_occurred", collision_occurred)
                self.logger.record("custom_metrics/avg_lane_discipline", avg_lane)
                
                # Reset episode buffers
                self.speeds = []
                self.collisions = []
                self.lanes = []
        return True

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on highway-env")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    training_config = config["training"]
    log_dir = training_config.get("log_dir", "logs")
    checkpoint_dir = training_config.get("checkpoint_dir", "checkpoints")
    total_timesteps = int(training_config.get("total_timesteps", 50000))
    checkpoint_freq = int(training_config.get("checkpoint_freq", 10000))
    eval_freq = int(training_config.get("eval_freq", 5000))
    eval_episodes = int(training_config.get("eval_episodes", 5))
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"TensorBoard log folder: {log_dir}/tensorboard/")
    print(f"Checkpoints saving folder: {checkpoint_dir}/")
    
    # Setup Vectorized Monitored Training Env
    def make_train_env():
        env = make_highway_env(args.config)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        return env
        
    train_env = DummyVecEnv([make_train_env])
    
    # Setup Vectorized Monitored Evaluation Env
    def make_eval_env():
        env = make_highway_env(args.config)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor_eval.csv"))
        return env
        
    eval_env = DummyVecEnv([make_eval_env])
    
    # Create PPO Agent
    print("Initializing PPO agent from hyperparameters...")
    model = create_ppo_agent(train_env, args.config)
    
    # Setup Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="highway_ppo_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(checkpoint_dir, "best_model"),
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False
    )
    
    custom_metric_callback = CustomMetricCallback()
    
    callback_list = CallbackList([checkpoint_callback, eval_callback, custom_metric_callback])
    
    # Begin training
    print(f"Starting training for {total_timesteps} timesteps on CPU...")
    model.learn(total_timesteps=total_timesteps, callback=callback_list)
    print("Training finished successfully!")
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "highway_ppo_final")
    save_agent(model, final_model_path)
    
    # Close environments
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
