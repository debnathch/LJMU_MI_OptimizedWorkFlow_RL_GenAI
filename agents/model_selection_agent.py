from stable_baselines3 import PPO
import os

class ModelSelectionAgent:
    """
    Agent 1: Model Selection Agent.
    Chooses between Lightweight and Heavyweight CNN.
    """
    def __init__(self, env, config: dict):
        self.env = env
        self.config = config['agents']['model_selection_agent']
        self.model_save_path = config['logging']['model_save_path']

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            ent_coef=self.config['ent_coef'],
            verbose=1,
            tensorboard_log=config['logging']['tensorboard_log']
        )

    def train(self, total_timesteps: int):
        self.model.learn(total_timesteps=total_timesteps, tb_log_name="ModelSelectionAgent")

    def save(self, name="model_selection_agent"):
        os.makedirs(self.model_save_path, exist_ok=True)
        self.model.save(os.path.join(self.model_save_path, name))

    def load(self, path):
        self.model = PPO.load(path, env=self.env)
