# Radiology Workflow Optimization using Sequential MARL

A research-grade implementation of a Multi-Agent Reinforcement Learning (MARL) system for intelligent radiology pipeline optimization.

## Architecture Overview

The system simulates a radiology department where two PPO-based agents operate sequentially to optimize turnaround time, accuracy, and resource utilization.

```text
[ Patient Case Arrival ]
          |
          v
+-----------------------+      Obs: Queue Length, Urgency, WL, GPU
|  Agent 2: Workflow    | <---------------------------------------+
|  (PPO Policy)         |                                         |
+-----------------------+                                         |
          | Action: Select Case & Assign Radiologist              |
          v                                                       |
+-----------------------+      Obs: GPU Load, Urgency, Pressure   |
|  Agent 1: Model Agent | <---------------------------------------+
|  (PPO Policy)         |                                         |
+-----------------------+                                         |
          | Action: Light vs Heavy CNN                            |
          v                                                       |
+-----------------------+                                         |
|   Simulated Inference | ----> Reward Logic ---------------------+
| (Accuracy vs Latency) |
+-----------------------+
          |
          v
[ Report Generation Stub ]
```

## Project Structure

- `env/`: Radiology environment and queue simulator.
- `agents/`: PPO agent wrappers (Stable-Baselines3).
- `models/`: Simulated CNN models (PyTorch).
- `training/`: Independent and Joint training scripts.
- `evaluation/`: KPI tracking and visualization.
- `configs/`: YAML-based configuration management.

## Installation

1. Ensure Python 3.10+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

You can train agents independently or jointly.

**Train Workflow Agent (Independent):**
```bash
python main.py --mode train_workflow --steps 20000
```

**Train Model Agent (Independent):**
```bash
python main.py --mode train_model --steps 20000
```

**Joint Training:**
```bash
python main.py --mode train_joint --episodes 500
```

### Evaluation

Evaluate the trained agents and generate performance plots:
```bash
python main.py --mode evaluate --episodes 20
```
Results will be saved in `logs/csv/evaluation_plots.png`.

### Visualization

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir ./logs/tensorboard/
```

## Reward Design

- **Workflow Agent**: Penalized for wait times, urgent case delays, and radiologist workload imbalance.
- **Model Agent**: Rewarded for accuracy, penalized for high latency and excessive GPU cost.

## Reproducibility

Fixed random seeds are used throughout the system (default seed: 42 in `default_config.yaml`).
