from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml
import os
import numpy as np
import torch
from env.radiology_env import RadiologyEnv, WorkflowEnvWrapper, ModelEnvWrapper
from agents.workflow_agent import WorkflowAgent
from agents.model_selection_agent import ModelSelectionAgent
from evaluation.metrics import RadiologyMetrics
from typing import List, Optional

app = FastAPI(title="Radiology MARL System API")

# Global state for simulation
class SimulationState:
    def __init__(self):
        self.env = None
        self.workflow_agent = None
        self.model_agent = None
        self.config = None
        self.is_initialized = False

    def initialize(self, config_path: str = "configs/default_config.yaml"):
        if not os.path.exists(config_path):
             return False
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.env = RadiologyEnv(config_path)
        
        workflow_env = WorkflowEnvWrapper(self.env)
        model_env = ModelEnvWrapper(self.env)
        
        self.workflow_agent = WorkflowAgent(workflow_env, self.config)
        self.model_agent = ModelSelectionAgent(model_env, self.config)
        
        # Load default checkpoints if they exist
        wf_path = "checkpoints/workflow_agent_joint"
        ms_path = "checkpoints/model_selection_agent_joint"
        
        if os.path.exists(wf_path + ".zip"):
            self.workflow_agent.load(wf_path)
        elif os.path.exists("checkpoints/workflow_agent_independent.zip"):
            self.workflow_agent.load("checkpoints/workflow_agent_independent")
            
        if os.path.exists(ms_path + ".zip"):
            self.model_agent.load(ms_path)
        elif os.path.exists("checkpoints/model_selection_agent_independent.zip"):
            self.model_agent.load("checkpoints/model_selection_agent_independent")
            
        self.is_initialized = True
        return True

sim = SimulationState()

class RunRequest(BaseModel):
    episodes: int = 5
    config_path: str = "configs/default_config.yaml"

class StepRequest(BaseModel):
    agent_type: str # "workflow" or "model"
    action: List[int]

@app.on_event("startup")
async def startup_event():
    sim.initialize()

@app.get("/health")
def health_check():
    return {"status": "healthy", "initialized": sim.is_initialized}

@app.get("/config")
def get_config():
    if not sim.is_initialized:
        raise HTTPException(status_code=400, detail="Simulation not initialized")
    return sim.config

@app.post("/simulation/run")
def run_simulation(req: RunRequest):
    if not sim.is_initialized:
        if not sim.initialize(req.config_path):
            raise HTTPException(status_code=400, detail="Failed to initialize simulation")

    results = []
    for ep in range(req.episodes):
        obs, _ = sim.env.reset()
        done = False
        ep_metrics = {
            "episode": ep,
            "workflow_reward": 0,
            "model_reward": 0,
            "total_wait": 0,
            "urgent_count": 0,
            "steps": 0
        }
        
        while not done:
            if sim.env.current_agent == sim.env.WORKFLOW_AGENT:
                action, _ = sim.workflow_agent.model.predict(obs, deterministic=True)
                new_obs, reward, terminated, truncated, info = sim.env.step(action)
                ep_metrics["workflow_reward"] += float(reward)
            else:
                action, _ = sim.model_agent.model.predict(obs, deterministic=True)
                if sim.env.selected_case:
                    ep_metrics["total_wait"] += float(sim.env.selected_case.wait_time)
                    if sim.env.selected_case.is_urgent:
                        ep_metrics["urgent_count"] += 1
                
                new_obs, reward, terminated, truncated, info = sim.env.step(action)
                ep_metrics["model_reward"] += float(reward)
            
            obs = new_obs
            done = terminated or truncated
            ep_metrics["steps"] += 1

        results.append(ep_metrics)
    
    return {"status": "success", "results": results}

@app.post("/simulation/reset")
def reset_simulation():
    if not sim.is_initialized:
        sim.initialize()
    obs, info = sim.env.reset()
    return {
        "observation": obs.tolist(),
        "current_agent": "workflow" if sim.env.current_agent == sim.env.WORKFLOW_AGENT else "model"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
