import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np


class MetricLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.log_file = os.path.join(save_dir, "log.csv")

        # Updated Headers
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Episode",
                    "Score",
                    "Reward",
                    "Epsilon",
                    "Loss",
                    "AvgQ",
                    "GradNorm",
                    "FlapProb",
                ]
            )

        self.episodes = []
        self.scores = []
        self.rewards = []
        self.epsilons = []
        self.losses = []
        self.avg_qs = []
        self.grad_norms = []
        self.flap_probs = []  # New

        self.start_time = time.time()
        self.record_time = time.time()

        plt.ion()
        # 3x2 Grid for maximum visibility
        self.fig, self.axs = plt.subplots(3, 2, figsize=(12, 10))
        self.fig.canvas.manager.set_window_title("Deep Q-Learning Dashboard")
        plt.tight_layout(pad=3.0)

    def log(self, episode, score, reward, epsilon, loss, avg_q, grad_norm, flap_prob):
        self.episodes.append(episode)
        self.scores.append(score)
        self.rewards.append(reward)
        self.epsilons.append(epsilon)
        self.losses.append(loss)
        self.avg_qs.append(avg_q)
        self.grad_norms.append(grad_norm)
        self.flap_probs.append(flap_prob)

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [episode, score, reward, epsilon, loss, avg_q, grad_norm, flap_prob]
            )

    def update_plot(self):
        x = self.episodes

        # 1. Scores (The Main Goal)
        ax = self.axs[0, 0]
        ax.clear()
        ax.plot(x, self.scores, "b-", alpha=0.3, label="Score")
        if len(self.scores) >= 50:
            avg = np.convolve(self.scores, np.ones(50) / 50, mode="valid")
            ax.plot(x[49:], avg, "r-", label="Avg Score")
        ax.set_title("Score")
        ax.legend()

        # 2. Loss (Stability Check)
        ax = self.axs[0, 1]
        ax.clear()
        ax.plot(x, self.losses, "m-", alpha=0.5)
        ax.set_title("Loss (Log Scale)")
        ax.set_yscale("log")

        # 3. Epsilon (Exploration Phase)
        ax = self.axs[1, 0]
        ax.clear()
        ax.plot(x, self.epsilons, "orange", label="Epsilon")
        ax.set_title("Epsilon (Exploration)")

        # 4. Action Distribution (Policy Health)
        ax = self.axs[1, 1]
        ax.clear()
        ax.plot(x, self.flap_probs, "c-", alpha=0.5)
        ax.set_title("Flap Probability (Action Dist)")
        ax.set_ylim(0, 1)  # 0% to 100%
        ax.axhline(
            y=0.1, color="k", linestyle="--", alpha=0.3
        )  # Reference line for "healthy" flapping

        # 5. Q-Values (Confidence)
        ax = self.axs[2, 0]
        ax.clear()
        ax.plot(x, self.avg_qs, "g-", alpha=0.5)
        ax.set_title("Avg Q-Value")

        # 6. Gradient Norm (Technical Health)
        ax = self.axs[2, 1]
        ax.clear()
        ax.plot(x, self.grad_norms, "k-", alpha=0.5)
        ax.set_title("Gradient Norm")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.savefig(os.path.join(self.save_dir, "training_graph.png"))

    def get_time_stats(self):
        now = time.time()
        total = int(now - self.start_time)
        return f"{total // 60}m {total % 60}s"
