import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class RadiologyMetrics:
    """
    Calculates and stores KPIs for radiology workflow.
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.results = []

    def log_step(self, metrics_dict: dict):
        self.results.append(metrics_dict)

    def save_to_csv(self, filename: str):
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.log_dir, filename), index=False)
        return df

    def plot_results(self, df: pd.DataFrame, prefix: str):
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Turnaround Time
        plt.subplot(2, 2, 1)
        sns.lineplot(data=df, x=df.index, y='avg_turnaround_time')
        plt.title('Average Turnaround Time')
        
        # Plot 2: GPU Utilization
        plt.subplot(2, 2, 2)
        sns.lineplot(data=df, x=df.index, y='gpu_utilization')
        plt.title('GPU Utilization %')
        
        # Plot 3: Urgent Delay Redux
        plt.subplot(2, 2, 3)
        sns.lineplot(data=df, x=df.index, y='urgent_delay')
        plt.title('Urgent Case Delay')
        
        # Plot 4: Rewards
        plt.subplot(2, 2, 4)
        sns.lineplot(data=df, x=df.index, y='workflow_reward', label='Workflow')
        sns.lineplot(data=df, x=df.index, y='model_reward', label='Model')
        plt.title('Agent Rewards')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{prefix}_plots.png"))
        plt.close()
