import argparse
import yaml
import os
from training.train_workflow_agent import train_workflow_independent
from training.train_model_agent import train_model_independent
from training.joint_training_loop import joint_training
from evaluation.evaluate import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Radiology MARL Optimization System")
    parser.add_argument("--mode", type=str, choices=["train_workflow", "train_model", "train_joint", "evaluate"], 
                        required=True, help="Mode of operation")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--episodes", type=int, default=10, help="Evaluation episodes")
    
    args = parser.parse_args()

    if args.mode == "train_workflow":
        train_workflow_independent(args.config, args.steps)
    elif args.mode == "train_model":
        train_model_independent(args.config, args.steps)
    elif args.mode == "train_joint":
        joint_training(args.config, total_episodes=args.episodes)
    elif args.mode == "evaluate":
        # Using joint-trained models as default for evaluation
        wf_path = "checkpoints/workflow_agent_joint"
        ms_path = "checkpoints/model_selection_agent_joint"
        if not os.path.exists(wf_path + ".zip"):
            wf_path = "checkpoints/workflow_agent_independent"
        if not os.path.exists(ms_path + ".zip"):
            ms_path = "checkpoints/model_selection_agent_independent"
            
        run_evaluation(args.config, wf_path, ms_path, args.episodes)

if __name__ == "__main__":
    main()
