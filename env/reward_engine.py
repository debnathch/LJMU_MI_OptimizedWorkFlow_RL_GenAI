import numpy as np

class RewardEngine:
    """
    Calculates rewards for Workflow and Model Selection agents.
    """
    def __init__(self, config: dict):
        self.config = config
        self.w_cfg = config['rewards']['workflow']
        self.m_cfg = config['rewards']['model']
        self.sla_threshold = config['environment']['sla_threshold']

    def calculate_workflow_reward(self, case, radiologist_loads, sla_violation=False):
        """
        Agent 2 Reward:
        - Reduce urgent delay
        - Minimize turnaround time
        - Workload imbalance penalty
        - SLA violation penalty
        """
        reward = 0.0
        
        # Turnaround time (wait time) penalty
        if case:
            reward -= self.w_cfg['turnaround_weight'] * case.wait_time
            if case.is_urgent:
                reward -= self.w_cfg['urgent_weight'] * case.wait_time

        # Workload imbalance penalty (variance of radiologist loads)
        if len(radiologist_loads) > 1:
            imbalance = np.var(radiologist_loads)
            reward -= self.w_cfg['imbalance_penalty'] * imbalance

        # SLA violation penalty
        if sla_violation:
            reward -= self.w_cfg['sla_penalty']

        return reward

    def calculate_model_reward(self, accuracy, latency, gpu_cost, is_urgent, wait_time):
        """
        Agent 1 Reward:
        - Diagnostic accuracy
        - Inference latency
        - GPU cost
        - Urgent case delay penalty
        """
        reward = self.m_cfg['accuracy_weight'] * accuracy
        reward -= self.m_cfg['latency_penalty'] * latency
        reward -= self.m_cfg['gpu_penalty'] * gpu_cost

        # Double penalty for latency if urgent
        if is_urgent:
            reward -= self.m_cfg['urgent_penalty'] * latency

        return reward
