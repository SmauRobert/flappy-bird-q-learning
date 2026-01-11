import glob
import json
import os
import random
import time

import cv2
import numpy as np
import torch

from src.agent import FlappyAgent
from src.environment import Environment
from src.utils import MetricLogger

# --- CONFIGURATION ---
PARAMS = {
    "seed": 221022,
    "episodes": 5000,
    "batch_size": 32,
    "lr": 1e-4,
    "gamma": 0.99,
    "epsilon_start": 0.9,
    "epsilon_end": 0.01,
    "epsilon_decay": 20000,
    "target_update": 1000,
    "memory_size": 50000,
    "stack_size": 4,
    "frame_skip": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
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


def play_replay(frames, score):
    print(f"\n>>> ⭐ NEW RECORD: {score} <<<")
    # Show at least 1 second of replay or it flickers too fast
    if len(frames) < 30:
        time.sleep(0.5)

    for f in frames:
        if f is not None:
            cv2.imshow("Replay", cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR))
            cv2.waitKey(30)
    time.sleep(0.5)


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
    agent = FlappyAgent(
        input_shape=(PARAMS["stack_size"], 84, 84),
        num_actions=2,
        device=device,
        learning_rate=PARAMS["lr"],
        gamma=PARAMS["gamma"],
        buffer_size=PARAMS["memory_size"],
    )

    print(f"Training on {device}... Press Ctrl+C to stop.")

    steps_done = 0
    best_score = 0

    try:
        for episode in range(PARAMS["episodes"]):
            state, _ = env.reset()
            done = False
            ep_reward = 0
            raw_score = 0
            frames = []

            epsilon = PARAMS["epsilon_start"]
            render_live = episode % 50 == 0

            while not done:
                # Capture frame
                if (
                    render_live or raw_score > best_score
                ):  # Keep frames if potentially a record
                    frame = env.env.render()
                    frames.append(frame)
                    if render_live and frame is not None:
                        cv2.imshow(
                            "Live Training",
                            cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR),
                        )
                        cv2.waitKey(1)

                # Decay
                epsilon = PARAMS["epsilon_end"] + (
                    PARAMS["epsilon_start"] - PARAMS["epsilon_end"]
                ) * np.exp(-1.0 * steps_done / PARAMS["epsilon_decay"])

                # Action
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
                    r = 5.0
                    raw_score = curr_score
                    # LOGGING CHANGE: Only print if score > 0
                    if raw_score > 0:
                        print(f"  Pipe {raw_score} passed!")

                # Learn
                agent.cache(state, action, r, next_state, done)
                loss = agent.learn(PARAMS["batch_size"])
                if steps_done % PARAMS["target_update"] == 0:
                    agent.sync_target()

                state = next_state
                ep_reward += r
                steps_done += 1

            # --- End of Episode ---
            logger.log(episode, raw_score, ep_reward, epsilon)

            # Check Record
            if raw_score > best_score:
                best_score = raw_score
                # Save
                torch.save(
                    agent.policy_net.state_dict(),
                    os.path.join(exp_path, f"best_model_score_{best_score}.weights"),
                )

                # TIMING STATS
                time_str = logger.get_time_stats()
                print(
                    f"\n[RECORD] Score: {best_score} | Time: {time_str} | Episode: {episode}"
                )

                play_replay(frames, best_score)

            # Update Graph every 10 episodes to prevent lag
            if episode % 10 == 0:
                logger.update_plot()
                print(
                    f"Ep {episode} | Score: {raw_score} | Reward: {ep_reward:.2f} | Eps: {epsilon:.4f}"
                )

    except KeyboardInterrupt:
        print("\nSaving...")
        torch.save(
            agent.policy_net.state_dict(), os.path.join(exp_path, "interrupted.weights")
        )
        logger.update_plot()  # Final graph update

    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    train()
