import argparse
import time
from collections import deque

import cv2
import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# --- CONFIG (Must match training!) ---
IMAGE_SIZE = 84
STACK_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. The Wrapper (Must match training logic) ---
class FlappyBirdCNNWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.stack_size = STACK_SIZE
        self.frames = deque(maxlen=STACK_SIZE)

    def _process_frame(self, frame):
        # Convert to Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84
        resized = cv2.resize(
            gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
        )
        # Normalize (0.0 - 1.0)
        return resized.astype(np.float32) / 255.0

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            _, info = self.env.reset(seed=seed, **kwargs)
        else:
            _, info = self.env.reset(**kwargs)

        # Capture initial frame
        frame = self.env.render()
        processed_frame = self._process_frame(frame)

        # Fill stack with the first frame
        for _ in range(self.stack_size):
            self.frames.append(processed_frame)

        return np.array(self.frames), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)

        frame = self.env.render()
        processed_frame = self._process_frame(frame)
        self.frames.append(processed_frame)

        return np.array(self.frames), reward, terminated, truncated, info


# --- 2. The Network Architecture (Must match training!) ---
class DuelingCNNDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingCNNDQN, self).__init__()
        c, h, w = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc_input_dim = 7 * 7 * 64

        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512), nn.ReLU(), nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


# --- 3. Evaluation Logic ---
def evaluate(model_path, num_games, fps):
    print(f"🔧 Initializing Environment...")
    # render_mode must be 'rgb_array' for our wrapper to grab frames
    raw_env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    env = FlappyBirdCNNWrapper(raw_env)

    # Initialize Model
    action_dim = env.action_space.n
    input_shape = (STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE)
    model = DuelingCNNDQN(input_shape, action_dim).to(DEVICE)

    # Load Weights
    print(f"📂 Loading model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)

        # Handle both raw state_dict and checkpoint dictionary
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"\n>>> STARTING EVALUATION ({num_games} Games) <<<")
    print("Commands:")
    print("  [TAB] - Switch View (Human Color <-> AI Grayscale)")
    print("  [Q]   - Quit")

    scores = []
    view_mode = 0  # 0 = Human, 1 = AI

    for i in range(1, num_games + 1):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # -- UI Handling --
            raw_frame = env.env.render()

            if view_mode == 0:
                # Human View
                display_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                display_frame = cv2.resize(
                    display_frame, (500, 500), interpolation=cv2.INTER_NEAREST
                )
            else:
                # AI View (The last frame in the stack)
                # Denormalize: 0.0-1.0 -> 0-255
                ai_frame = state[-1] * 255
                ai_frame = ai_frame.astype(np.uint8)
                display_frame = cv2.resize(
                    ai_frame, (500, 500), interpolation=cv2.INTER_NEAREST
                )
                # Apply false color map for coolness/visibility
                display_frame = cv2.applyColorMap(display_frame, cv2.COLORMAP_VIRIDIS)

            # Add Info Text
            cv2.putText(
                display_frame,
                f"Game: {i}/{num_games}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                f"Mode: {'Human' if view_mode == 0 else 'AI Input'}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Flappy Bird AI", display_frame)

            # Key Listener
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                print("Quit requested.")
                env.close()
                cv2.destroyAllWindows()
                return
            elif key == 9:  # TAB
                view_mode = 1 - view_mode

            # -- AI Action --
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = model(state_t)
                action = q_values.argmax().item()

            # -- Step --
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Artificial delay for human visibility
            if fps > 0:
                time.sleep(1.0 / fps)

        scores.append(total_reward)
        print(f"Game {i} Finished | Score: {total_reward:.1f}")

    print("\n>>> RESULTS <<<")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="best_model.pth", help="Path to .pth file"
    )
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    parser.add_argument(
        "--fps", type=int, default=30, help="Limit FPS for watching (0 = unlimited)"
    )

    args = parser.parse_args()
    evaluate(args.model, args.games, args.fps)
