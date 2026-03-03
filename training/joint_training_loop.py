import yaml
import torch
import numpy as np
from env.radiology_env import RadiologyEnv
from agents.workflow_agent import WorkflowAgent
from agents.model_selection_agent import ModelSelectionAgent

def joint_training(config_path: str, total_episodes: int = 500):
    """
    Custom joint training loop for sequential MARL.
    """
    env = RadiologyEnv(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    workflow_agent = WorkflowAgent(env, config)
    model_agent = ModelSelectionAgent(env, config)

    print(f"Starting Joint Training for {total_episodes} episodes...")

    for episode in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward_w = 0
        episode_reward_m = 0

        while not done:
            # Current agent turn determined by info
            if env.current_agent == env.WORKFLOW_AGENT:
                action, _ = workflow_agent.model.predict(obs, deterministic=False)
                new_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward_w += reward
            else:
                action, _ = model_agent.model.predict(obs, deterministic=False)
                new_obs, reward, terminated, truncated, info = env.step(action)
                episode_reward_m += reward
            
            obs = new_obs
            done = terminated or truncated

        # For SB3, we can't easily do partial updates without buffers.
        # A simpler 'joint' approach is alternating training:
        # 1. Train Workflow for 1000 steps with Model fixed
        # 2. Train Model for 1000 steps with Workflow fixed
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}: Workflow Reward: {episode_reward_w:.2f}, Model Reward: {episode_reward_m:.2f}")

    print("Joint training logic would typically involve alternating .learn() calls.")
    # In practice for research code:
    print("Performing alternating learning phases...")
    workflow_agent.train(total_timesteps=10000)
    model_agent.train(total_timesteps=10000)
    
    workflow_agent.save("workflow_agent_joint")
    model_agent.save("model_selection_agent_joint")
    print("Joint Training Complete.")

if __name__ == "__main__":
    joint_training("configs/default_config.yaml")
