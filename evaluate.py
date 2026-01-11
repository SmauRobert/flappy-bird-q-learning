import argparse
import time

import cv2
import numpy as np
import torch

from src.agent import FlappyAgent
from src.environment import Environment

# --- SETTINGS ---
FRAME_SKIP = 4
STACK_SIZE = 4
DEVICE = "cpu"


def evaluate(model_path, num_games):
    print("Initializing Environment...")
    env = Environment(
        render_mode="rgb_array", stack_size=STACK_SIZE, frame_skip=FRAME_SKIP
    )

    agent = FlappyAgent(
        input_shape=(STACK_SIZE, 84, 84), num_actions=2, device=torch.device(DEVICE)
    )

    print(f"Loading model from: {model_path}")
    try:
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.policy_net.eval()
    except FileNotFoundError:
        print("❌ Error: Model file not found.")
        return

    print(f"\n>>> STARTING EVALUATION ({num_games} Games) <<<")
    print("Commands:")
    print("  [TAB] - Toggle View (Human Color <-> AI Grayscale)")
    print("  [Q]   - Quit")

    scores = []
    view_mode = 0  # 0 = Human, 1 = AI

    for i in range(1, num_games + 1):
        seed = np.random.randint(0, 100000)
        state, info = env.reset(seed=seed)
        done = False
        score = 0

        while not done:
            # 1. Get the visualization frames
            raw_frame = env.env.render()  # Human view (Color)
            ai_frame = state[-1]  # AI view (Last frame of the stack, Grayscale)

            # 2. Render based on mode
            if raw_frame is not None:
                if view_mode == 0:
                    # HUMAN MODE: Show raw color
                    display = cv2.cvtColor(np.array(raw_frame), cv2.COLOR_RGB2BGR)
                    cv2.imshow(
                        "Evaluation",
                        cv2.resize(
                            display, (500, 500), interpolation=cv2.INTER_NEAREST
                        ),
                    )
                else:
                    # AI MODE: Show what the network actually sees
                    # It's 84x84 float (0-1), so we scale to 0-255 for display
                    display = (ai_frame * 255).astype(np.uint8)
                    display = cv2.resize(
                        display, (500, 500), interpolation=cv2.INTER_NEAREST
                    )
                    cv2.imshow("Evaluation", display)

                # Handle Keys
                key = cv2.waitKey(30)
                if key & 0xFF == ord("q"):
                    return
                elif key == 9:  # TAB key code
                    view_mode = 1 - view_mode
                    print(
                        f"Switched View to: {'AI (Grayscale)' if view_mode else 'Human (Color)'}"
                    )

            # 3. AI Action
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score = info.get("score", 0)

        scores.append(score)
        print(f"Game {i} | Seed: {seed} | Score: {score}")

    print(f"\n>>> EVALUATION FINISHED <<<")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Flappy Bird AI")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the .weights file"
    )
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")

    args = parser.parse_args()
    evaluate(args.model, args.games)
