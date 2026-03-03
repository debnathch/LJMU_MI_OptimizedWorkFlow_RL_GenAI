import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
from env.queue_simulator import QueueSimulator
from env.reward_engine import RewardEngine

class RadiologyEnv(gym.Env):
    """
    Base Sequential Multi-Agent Environment for Radiology Workflow Optimization.
    Manages the shared state and simulation.
    """
    def __init__(self, config_path: str):
        super(RadiologyEnv, self).__init__()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_radiologists = self.config['environment']['num_radiologists']
        self.max_queue_size = self.config['environment']['max_queue_size']
        self.gpu_capacity = self.config['environment']['gpu_capacity']
        self.max_steps = self.config['environment']['max_steps']

        # Components
        self.simulator = QueueSimulator(self.config)
        self.reward_engine = RewardEngine(self.config)

        # Agent IDs
        self.WORKFLOW_AGENT = 2
        self.MODEL_AGENT = 1
        self.current_agent = self.WORKFLOW_AGENT

        # Action Spaces
        self.action_space_workflow = spaces.MultiDiscrete([self.max_queue_size, self.num_radiologists])
        self.action_space_model = spaces.Discrete(2)

        # Observation Spaces
        # Workflow State: queue_length, urgent_cases_count, average_wait_time, radiologist_load (vec), gpu_utilization
        self.observation_space_workflow = spaces.Box(
            low=0, high=1000, shape=(4 + self.num_radiologists,), dtype=np.float32
        )
        self.observation_space_model = spaces.Box(
            low=0, high=1000, shape=(4,), dtype=np.float32
        )

        # State Variables
        self.current_step = 0
        self.radiologist_loads = np.zeros(self.num_radiologists)
        self.current_gpu_load = 0.0
        self.last_latency = 0.0
        self.selected_case = None
        self.selected_radiologist = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.simulator.reset()
        self.current_step = 0
        self.radiologist_loads = np.zeros(self.num_radiologists)
        self.current_gpu_load = 0.0
        self.last_latency = 0.0
        self.current_agent = self.WORKFLOW_AGENT
        self.selected_case = None

        # Initial queue simulation
        self.simulator.step(self.current_step)
        
        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_agent == self.WORKFLOW_AGENT:
            q_state = self.simulator.get_state()
            obs = np.array([
                q_state['queue_length'],
                q_state['urgent_cases_count'],
                q_state['average_wait_time'],
                *self.radiologist_loads,
                self.current_gpu_load
            ], dtype=np.float32)
            return obs
        else:
            q_state = self.simulator.get_state()
            queue_pressure = q_state['queue_length'] / self.max_queue_size
            case_urgency = 1.0 if (self.selected_case and self.selected_case.is_urgent) else 0.0
            obs = np.array([
                self.current_gpu_load,
                case_urgency,
                queue_pressure,
                self.last_latency
            ], dtype=np.float32)
            return obs

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if self.current_agent == self.WORKFLOW_AGENT:
            case_idx, radio_id = action
            q_len = len(self.simulator.queue)
            actual_case_idx = min(case_idx, q_len - 1) if q_len > 0 else -1
            
            if actual_case_idx != -1:
                self.selected_case = self.simulator.remove_case(actual_case_idx)
                self.selected_radiologist = radio_id
                reward = self.reward_engine.calculate_workflow_reward(
                    self.selected_case, self.radiologist_loads
                )
                self.current_agent = self.MODEL_AGENT
            else:
                reward = -1.0
                self._progress_time()
        else:
            model_type = "heavyweight_cnn" if action == 1 else "lightweight_cnn"
            model_cfg = self.config['simulation'][model_type]
            accuracy = model_cfg['accuracy']
            latency = model_cfg['latency']
            gpu_cost = model_cfg['gpu_cost']
            
            self.last_latency = latency
            self.current_gpu_load = min(self.gpu_capacity, self.current_gpu_load + gpu_cost)
            
            if self.selected_radiologist is not None:
                self.radiologist_loads[self.selected_radiologist] += 1
            
            reward = self.reward_engine.calculate_model_reward(
                accuracy, latency, gpu_cost, 
                self.selected_case.is_urgent if self.selected_case else False,
                self.selected_case.wait_time if self.selected_case else 0
            )
            
            self._progress_time()
            self.current_agent = self.WORKFLOW_AGENT
            self.selected_case = None

        self.current_step += 1
        if self.current_step >= self.max_steps:
            terminated = True

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _progress_time(self):
        self.current_gpu_load = max(0.0, self.current_gpu_load - 5.0)
        self.radiologist_loads = np.maximum(0.0, self.radiologist_loads - 0.1)
        self.simulator.step(self.current_step)

class WorkflowEnvWrapper(gym.Env):
    def __init__(self, base_env: RadiologyEnv):
        self.base_env = base_env
        self.action_space = base_env.action_space_workflow
        self.observation_space = base_env.observation_space_workflow

    def reset(self, **kwargs):
        obs, info = self.base_env.reset(**kwargs)
        while self.base_env.current_agent != self.base_env.WORKFLOW_AGENT:
            obs, _, term, trunc, _ = self.base_env.step(self.base_env.action_space_model.sample())
            if term or trunc: obs, info = self.base_env.reset()
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.base_env.step(action)
        # If it's the model agent's turn, we need to take a dummy action to get back to workflow
        while self.base_env.current_agent != self.base_env.WORKFLOW_AGENT and not (term or trunc):
            # For independent training, use a random model selection for the other agent
            obs, m_reward, term, trunc, info = self.base_env.step(self.base_env.action_space_model.sample())
        return obs, reward, term, trunc, info

class ModelEnvWrapper(gym.Env):
    def __init__(self, base_env: RadiologyEnv):
        self.base_env = base_env
        self.action_space = base_env.action_space_model
        self.observation_space = base_env.observation_space_model

    def reset(self, **kwargs):
        obs, info = self.base_env.reset(**kwargs)
        while self.base_env.current_agent != self.base_env.MODEL_AGENT:
            obs, _, term, trunc, _ = self.base_env.step(self.base_env.action_space_workflow.sample())
            if term or trunc: obs, info = self.base_env.reset()
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.base_env.step(action)
        while self.base_env.current_agent != self.base_env.MODEL_AGENT and not (term or trunc):
            obs, w_reward, term, trunc, info = self.base_env.step(self.base_env.action_space_workflow.sample())
        return obs, reward, term, trunc, info

