import yaml
import torch
import numpy as np
from env.radiology_env import RadiologyEnv, WorkflowEnvWrapper, ModelEnvWrapper
from agents.workflow_agent import WorkflowAgent
from agents.model_selection_agent import ModelSelectionAgent
from evaluation.metrics import RadiologyMetrics
import os

def run_evaluation(config_path: str, workflow_model_path: str, model_selection_path: str, episodes: int = 10):
    """
    Evaluates trained agents and logs metrics.
    """
    base_env = RadiologyEnv(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Wrap for SB3 compatibility
    workflow_env = WorkflowEnvWrapper(base_env)
    model_env = ModelEnvWrapper(base_env)

    # Load agents
    workflow_agent = WorkflowAgent(workflow_env, config)
    workflow_agent.load(workflow_model_path)
    
    model_agent = ModelSelectionAgent(model_env, config)
    model_agent.load(model_selection_path)


    metrics = RadiologyMetrics(config['logging']['csv_log'])
    print(f"Starting evaluation for {episodes} episodes...")

    for ep in range(episodes):
        obs, _ = base_env.reset()
        done = False
        ep_w_reward = 0
        ep_m_reward = 0
        total_wait = 0
        urgent_wait = 0
        urgent_count = 0
        
        while not done:
            if base_env.current_agent == base_env.WORKFLOW_AGENT:
                action, _ = workflow_agent.model.predict(obs, deterministic=True)
                new_obs, reward, terminated, truncated, info = base_env.step(action)
                ep_w_reward += reward
            else:
                action, _ = model_agent.model.predict(obs, deterministic=True)
                # Tracking metrics
                if base_env.selected_case:
                    total_wait += base_env.selected_case.wait_time
                    if base_env.selected_case.is_urgent:
                        urgent_wait += base_env.selected_case.wait_time
                        urgent_count += 1
                
                new_obs, reward, terminated, truncated, info = base_env.step(action)
                ep_m_reward += reward
            
            obs = new_obs
            done = terminated or truncated

        # Log episode metrics
        metrics.log_step({
            "episode": ep,
            "workflow_reward": ep_w_reward,
            "model_reward": ep_m_reward,
            "avg_turnaround_time": total_wait / base_env.max_steps,
            "gpu_utilization": base_env.current_gpu_load,
            "urgent_delay": urgent_wait / max(1, urgent_count)
        })


    df = metrics.save_to_csv("evaluation_results.csv")
    metrics.plot_results(df, "evaluation")
    print("Evaluation complete. Results saved to logs/csv/")

if __name__ == "__main__":
    # Example usage (assuming models exist)
    # run_evaluation("configs/default_config.yaml", "checkpoints/workflow_agent_joint", "checkpoints/model_selection_agent_joint")
    pass
