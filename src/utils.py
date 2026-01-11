import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np


class MetricLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.log_file = os.path.join(save_dir, "log.csv")

        # Headers
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Score", "Reward", "Epsilon", "TimeElapsed"])

        self.episodes = []
        self.scores = []
        self.rewards = []
        self.epsilons = []

        # Timing
        self.start_time = time.time()
        self.record_time = time.time()  # Tracks time since last record

        # Live Plotting Setup
        plt.ion()  # Interactive mode on
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        self.fig.canvas.manager.set_window_title("Training Evolution")
        (self.line_scores,) = self.ax1.plot([], [], color="tab:blue", label="Score")
        (self.line_trend,) = self.ax1.plot(
            [], [], color="red", linestyle="--", label="Avg"
        )
        (self.line_reward,) = self.ax2.plot([], [], color="tab:green", label="Reward")

        self.ax3 = self.ax2.twinx()
        (self.line_epsilon,) = self.ax3.plot(
            [], [], color="tab:orange", linestyle=":", label="Epsilon"
        )

        self.ax1.set_ylabel("Pipes Passed")
        self.ax1.legend(loc="upper left")
        self.ax1.grid(True, alpha=0.3)

        self.ax2.set_ylabel("Reward", color="tab:green")
        self.ax3.set_ylabel("Epsilon", color="tab:orange")
        plt.tight_layout()

    def log(self, episode, score, reward, epsilon):
        # Update Data
        self.episodes.append(episode)
        self.scores.append(score)
        self.rewards.append(reward)
        self.epsilons.append(epsilon)

        elapsed = time.time() - self.start_time

        # Save to CSV
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, score, reward, epsilon, elapsed])

    def update_plot(self):
        """Refreshes the live window (Call this periodically, not every step)"""
        x = self.episodes

        # Update Score Line
        self.line_scores.set_data(x, self.scores)

        # Update Trend Line
        if len(self.scores) >= 50:
            window = 50
            trend = np.convolve(self.scores, np.ones(window) / window, mode="valid")
            self.line_trend.set_data(x[window - 1 :], trend)

        # Update Reward/Epsilon
        self.line_reward.set_data(x, self.rewards)
        self.line_epsilon.set_data(x, self.epsilons)

        # Rescale Axes
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Save static copy
        plt.savefig(os.path.join(self.save_dir, "training_graph.png"))

    def get_time_stats(self):
        """Returns string with total time and time since last record"""
        now = time.time()
        total_time = now - self.start_time
        since_last = now - self.record_time
        self.record_time = now  # Reset split timer

        return f"{int(total_time // 60)}m {int(total_time % 60)}s (Split: {int(since_last)}s)"
