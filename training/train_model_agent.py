import yaml
from env.radiology_env import RadiologyEnv, ModelEnvWrapper
from agents.model_selection_agent import ModelSelectionAgent

def train_model_independent(config_path: str, timesteps: int = 20000):
    base_env = RadiologyEnv(config_path)
    env = ModelEnvWrapper(base_env)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    agent = ModelSelectionAgent(env, config)
    print("Starting independent training for Model Selection Agent...")
    agent.train(total_timesteps=timesteps)
    agent.save("model_selection_agent_independent")
    print("Training complete.")

if __name__ == "__main__":
    train_model_independent("configs/default_config.yaml")
