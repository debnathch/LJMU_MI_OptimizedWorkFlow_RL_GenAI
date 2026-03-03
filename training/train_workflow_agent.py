import yaml
from env.radiology_env import RadiologyEnv, WorkflowEnvWrapper
from agents.workflow_agent import WorkflowAgent

def train_workflow_independent(config_path: str, timesteps: int = 20000):
    base_env = RadiologyEnv(config_path)
    env = WorkflowEnvWrapper(base_env)
    # During independent training of workflow agent, 
    # the environment will use a random or default action for the model agent 
    # if it's not specifically handled.
    # In our RadiologyEnv, it switches control.
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    agent = WorkflowAgent(env, config)
    print("Starting independent training for Workflow Agent...")
    agent.train(total_timesteps=timesteps)
    agent.save("workflow_agent_independent")
    print("Training complete.")

if __name__ == "__main__":
    train_workflow_independent("configs/default_config.yaml")
