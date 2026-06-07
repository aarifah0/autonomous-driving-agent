import gymnasium as gym
import highway_env
import yaml
import numpy as np

class CustomRewardWrapper(gym.Wrapper):
    """
    Custom wrapper to calculate a multi-objective reward function for highway-env.
    Balances:
    1. Speed: Encourage moving forward quickly.
    2. Collision avoidance: High penalty for crashing.
    3. Lane discipline: Reward staying centered in the current lane.
    """
    def __init__(self, env, wrapper_config):
        super().__init__(env)
        self.w_speed = wrapper_config.get("weight_speed", 1.5)
        self.w_collision = wrapper_config.get("weight_collision", 20.0)
        self.w_lane = wrapper_config.get("weight_lane", 0.8)
        self.v_min = wrapper_config.get("v_min", 0.0)
        self.v_max = wrapper_config.get("v_max", 30.0)
        
        # Keep track of rewards for logging / analysis
        self.episode_rewards = []
        self.current_episode_speed_rewards = []
        self.current_episode_collision_rewards = []
        self.current_episode_lane_rewards = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_episode_speed_rewards = []
        self.current_episode_collision_rewards = []
        self.current_episode_lane_rewards = []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        try:
            ego = self.env.unwrapped.vehicle
            road = self.env.unwrapped.road
            
            # 1. Speed Reward
            # Normalise speed to [0, 1] range and scale by weight
            v_ego = ego.speed
            speed_reward = (v_ego - self.v_min) / (self.v_max - self.v_min) if self.v_max > self.v_min else 0.0
            speed_reward = max(0.0, min(1.0, speed_reward))
            
            # 2. Collision Reward
            # Positive reward when driving safely, negative penalty when crashing
            is_crashed = info.get("crashed", False)
            collision_reward = -1.0 if is_crashed else 1.0
            
            # 3. Lane Discipline Reward
            # Reward staying centered in the lane. We compute lateral offset in the lane.
            lane_index = ego.lane_index
            lane = road.network.get_lane(lane_index)
            _, lateral = lane.local_coordinates(ego.position)
            lane_width = getattr(lane, "width", 4.0)
            
            # We use a quadratic penalty for being off-center
            lane_reward = 1.0 - (lateral / (lane_width / 2.0)) ** 2
            lane_reward = max(-1.0, min(1.0, lane_reward))
            
        except Exception as e:
            # Fallbacks in case ego vehicle or lane coordinates are not accessible
            speed_reward = 0.0
            collision_reward = -1.0 if info.get("crashed", False) else 1.0
            lane_reward = 0.0

        # Weighted custom reward calculation
        custom_reward = (
            self.w_speed * speed_reward +
            self.w_collision * collision_reward +
            self.w_lane * lane_reward
        )
        
        # Track raw component values for analysis
        self.current_episode_speed_rewards.append(speed_reward)
        self.current_episode_collision_rewards.append(collision_reward)
        self.current_episode_lane_rewards.append(lane_reward)
        
        # Store episode-wide components in info dictionary for callbacks / metrics
        info["custom_reward_components"] = {
            "speed": speed_reward,
            "collision": collision_reward,
            "lane": lane_reward,
            "raw_custom_reward": custom_reward
        }
        
        return obs, custom_reward, terminated, truncated, info

def make_highway_env(config_path="config.yaml", render_mode=None):
    """
    Creates and configures a highway-env instance wrapped with the custom reward calculator.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    env_id = config["env"]["id"]
    env_config = config["env"]["config"]
    wrapper_config = config["wrapper"]
    
    # Gymnasium expects render_mode during make
    if render_mode is not None:
        env = gym.make(env_id, render_mode=render_mode)
    else:
        env = gym.make(env_id)
        
    # Configure the unwrapped environment
    env.unwrapped.configure(env_config)
    env.reset()
    
    # Apply our custom reward wrapper
    wrapped_env = CustomRewardWrapper(env, wrapper_config)
    return wrapped_env
