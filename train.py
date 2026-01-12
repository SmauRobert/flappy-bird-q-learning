import glob
import json
import os
import random

import cv2
import numpy as np
import torch
from src.agent import FlappyAgent
from src.environment import Environment
from src.utils import MetricLogger

# --- CONFIGURATION ---
PARAMS = {
    "seed": 221022,
    "episodes": 10000,
    "batch_size": 32,
    "lr": 1e-4,
    "gamma": 0.99,
    "epsilon_start": 0.90,
    "epsilon_end": 0.01,
    "epsilon_decay": 30000,
    "target_update": 2000,
    "memory_size": 100000,
    "stack_size": 4,
    "frame_skip": 4,
    "validate_every": 250,
    "validate_games": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "initial_weights": "checkpoints/exp_005/best_model_score_6.weights",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def setup_experiment(base_dir="checkpoints"):
    os.makedirs(base_dir, exist_ok=True)
    existing = glob.glob(os.path.join(base_dir, "exp_*"))
    ids = [int(e.split("_")[-1]) for e in existing if "_" in e]
    next_id = max(ids) + 1 if ids else 1
    path = os.path.join(base_dir, f"exp_{next_id:03d}")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "params.json"), "w") as f:
        json.dump(PARAMS, f, indent=4)
    print(f"Initialized: {path}")
    return path


def validate(agent, env, num_games):
    """
    Runs evaluation games with Epsilon=0 (Pure Exploitation).
    Returns the average score.
    """
    total_score = 0
    for _ in range(num_games):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, epsilon=0.0)
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            score = info.get("score", 0)
        total_score += score
    return total_score / num_games


def train():
    set_seed(PARAMS["seed"])
    exp_path = setup_experiment()
    logger = MetricLogger(exp_path)
    device = torch.device(PARAMS["device"])

    env = Environment(
        render_mode="rgb_array",
        stack_size=PARAMS["stack_size"],
        frame_skip=PARAMS["frame_skip"],
    )

    # Pass initial_weights to agent
    agent = FlappyAgent(
        input_shape=(PARAMS["stack_size"], 84, 84),
        num_actions=2,
        device=device,
        learning_rate=PARAMS["lr"],
        gamma=PARAMS["gamma"],
        buffer_size=PARAMS["memory_size"],
        initial_weights=PARAMS["initial_weights"],
    )

    print(f"Training on {device}... Press Ctrl+C to stop.")

    steps_done = 0
    best_score = 0
    best_avg_score = 0.0  # Track best validation average

    try:
        for episode in range(PARAMS["episodes"]):
            # --- 1. Validation Step ---
            if episode > 0 and episode % PARAMS["validate_every"] == 0:
                print(
                    f"\nRunning Validation ({PARAMS['validate_games']} games)...",
                    end="",
                )
                avg_score = validate(agent, env, PARAMS["validate_games"])
                print(f" Average Score: {avg_score:.2f}")

                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    print("NEW BEST AVERAGE! Saving model...")
                    torch.save(
                        agent.policy_net.state_dict(),
                        os.path.join(
                            exp_path, f"best_avg_score_{best_avg_score:.2f}.weights"
                        ),
                    )
            # -----------------------------------------

            state, _ = env.reset()
            done = False
            ep_reward = 0
            raw_score = 0

            epsilon = PARAMS["epsilon_start"]

            while not done:
                epsilon = PARAMS["epsilon_end"] + (
                    PARAMS["epsilon_start"] - PARAMS["epsilon_end"]
                ) * np.exp(-1.0 * steps_done / PARAMS["epsilon_decay"])

                action = agent.select_action(state, epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Reward Shaping
                r = 0.1
                if done:
                    r = -1.0

                # Check pipe passing
                curr_score = info.get("score", 0)
                if curr_score > raw_score:
                    r = 5.0  # High reward for pipe
                    raw_score = curr_score

                agent.cache(state, action, r, next_state, done)
                agent.learn(PARAMS["batch_size"])

                if steps_done % PARAMS["target_update"] == 0:
                    agent.sync_target()

                state = next_state
                ep_reward += r
                steps_done += 1

            # --- End of Episode ---
            logger.log(episode, raw_score, ep_reward, epsilon)

            # Check Record
            if raw_score >= best_score and raw_score > 0:
                best_score = raw_score
                torch.save(
                    agent.policy_net.state_dict(),
                    os.path.join(exp_path, f"best_model_score_{best_score}.weights"),
                )
                time_str = logger.get_time_stats()
                print(
                    f"\n[RECORD] Score: {best_score} | Time: {time_str} | Ep: {episode}"
                )

            # 3. Console Output
            print(
                f"Ep {episode:4d} | Score: {raw_score} | Reward: {ep_reward:5.1f} | Eps: {epsilon:.4f}",
                end="\r",
            )

            if episode % 10 == 0:
                time_str = logger.get_time_stats()
                print(
                    f"Ep {episode:4d} | Score: {raw_score} | Reward: {ep_reward:5.1f} | Eps: {epsilon:.4f} | Time: {time_str}"
                )
                logger.update_plot()

    except KeyboardInterrupt:
        print("\nSaving interrupted model...")
        torch.save(
            agent.policy_net.state_dict(), os.path.join(exp_path, "interrupted.weights")
        )
        logger.update_plot()

    finally:
        print("\nSaving final model state...")
        if "agent" in locals():
            torch.save(
                agent.policy_net.state_dict(),
                os.path.join(exp_path, "final_model.weights"),
            )
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    train()
