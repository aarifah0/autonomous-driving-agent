import yaml
from stable_baselines3 import PPO

def create_ppo_agent(env, config_path="config.yaml"):
    """
    Parses PPO agent parameters from config.yaml and initializes a Stable-Baselines3 PPO agent.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    agent_config = config["agent"]
    training_config = config["training"]
    
    # Extract training-related directories
    log_dir = training_config.get("log_dir", "logs")
    tensorboard_log = f"{log_dir}/tensorboard/"
    
    # Extract hyperparameters
    policy = agent_config.get("policy", "MlpPolicy")
    learning_rate = float(agent_config.get("learning_rate", 3e-4))
    n_steps = int(agent_config.get("n_steps", 2048))
    batch_size = int(agent_config.get("batch_size", 64))
    n_epochs = int(agent_config.get("n_epochs", 10))
    gamma = float(agent_config.get("gamma", 0.99))
    gae_lambda = float(agent_config.get("gae_lambda", 0.95))
    clip_range = float(agent_config.get("clip_range", 0.2))
    ent_coef = float(agent_config.get("ent_coef", 0.0))
    verbose = int(agent_config.get("verbose", 1))
    
    # Net arch needs structure formatting for SB3
    policy_kwargs = agent_config.get("policy_kwargs", {})
    
    # Create SB3 PPO model
    model = PPO(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose
    )
    return model

def save_agent(model, path):
    """
    Saves the trained model to path.
    """
    model.save(path)
    print(f"Agent successfully saved to: {path}")

def load_agent(path, env=None):
    """
    Loads a trained agent from a zip checkpoint.
    """
    model = PPO.load(path, env=env)
    print(f"Agent successfully loaded from: {path}")
    return model
