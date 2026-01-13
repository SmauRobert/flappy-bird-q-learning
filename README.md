# Flappy Bird AI Agent (Deep Q-Learning)

![Project Banner](path/to/your/banner_image.png)
_([Tip: You can add a screenshot of your agent playing here. Syntax: `![Alt Text](url)`])_

## 📜 Assignment 5: Reinforcement Learning

**Team:** [Student Name 1], [Student Name 2]  
**Course:** Neural Networks / AI  
**Deadline:** Week 14

---

## 📖 Table of Contents

-
- [How to Run](#-how-to-run)
- [Architecture & Methodology](#-architecture--methodology)
- [Training Strategy](#-training-strategy)
- [Results & Performance](#-results--performance)
- [Challenges & Future Improvements](#-challenges--future-improvements)
- [File Structure](#-file-structure)

---

## 🤖 Introduction

This project implements a Reinforcement Learning (RL) agent capable of playing **Flappy Bird** purely from raw pixels. The agent uses a **Dueling Double Deep Q-Network (D3QN)** to estimate optimal actions (`Jump` or `Do Nothing`) based on visual input.

The goal was to maximize the score by stabilizing the learning process using advanced RL techniques like **Experience Replay** and **Target Networks**.

### Key Features

- **Input:** 4 stacked grayscale frames (84x84px).
- **Model:** Dueling DQN (Separates Value & Advantage streams).
- **Optimization:** Adam Optimizer with SmoothL1Loss.
- **Policy:** Epsilon-Greedy with compound decay.

---

## 🚀 How to Run

### 1. Prerequisites

Make sure you have the required libraries installed:

```bash
pip install torch torchvision numpy opencv-python gym
```

### 2. Training

To train the agent from scratch (or resume from a checkpoint):

```Bash
python train.py
```

### 3. Evaluation (Watch it Play)

To watch the best-performing model play:

```Bash
python main.py --eval --model_path checkpoints/best_model.pth
```

## 🧠 Architecture & Methodology

### 1. Data Preprocessing

Instead of using raw game coordinates, we train on pixels to mimic human perception.

- **Grayscale Conversion**: Reduces input complexity (3 channels -> 1 channel).
- **Resizing**: Downscales frames to `84x84` for faster processing.
- **Frame Stacking**: We stack the last **4 frames** to give the agent a sense of velocity and acceleration.

### 2. Neural Network (Dueling DQN)

We utilize a Dueling Architecture which splits the network into two streams after the convolutional layers:

- **Value Stream V(s)**: Estimates the value of being in the current state.
- **Advantage Stream A(s,a)**: Estimates the benefit of taking a specific action.

Q(s,a)=V(s)+(A(s,a)−∣A∣1​a′∑​A(s,a′))

**Layer Breakdown**:

- `Conv2d`: 32 filters, 8x8 kernel, stride 4.
- `Conv2d`: 64 filters, 4x4 kernel, stride 2.
- `Conv2d`: 64 filters, 3x3 kernel, stride 1.
- `Linear`: Splits into Value (1 output) and Advantage (2 outputs).

## ⚙️ Training Strategy

### Hyperparameters

| Parameter     | Value  |                         Description |
| :------------ | :----: | ----------------------------------: |
| Batch Size    |   32   | Number of samples per training step |
| Gamma         |  0.99  |  Discount factor for future rewards |
| Learning Rate |  1e-4  |            Adam Optimizer step size |
| Buffer Size   | 50,000 |          Experience Replay capacity |
| Epsilon Start |  1.0   |            Initial exploration rate |
| Epsilon End   |  0.01  |              Final exploration rate |

### Reward Shaping

To encourage faster convergence, we modified the default rewards:

- +0.1: For every frame survived.
- +1.0: For passing a pipe.
- -1.0: For dying.

## 📊 Results & Performance

After training for 5000 episodes, our agent achieved the following stats:

- Max Score: 230 pipes
- Average Score (Last 100): 45 pipes
- Training Time: ~4 hours on NVIDIA GTX 1650

### Training Progress

<!--(You can upload your plot image to the repo and link it here)

    Observation: The agent initially struggles (pure exploration) but begins to understand the physics around Episode 1500, leading to a spike in scores.-->

## ⚠️ Challenges & Future Improvements

### Challenges

- **Sparse Rewards**: The agent takes a long time to find the first pipe, making initial learning slow.
- **Catastrophic Forgetting**: Sometimes the agent "forgets" how to fly safely after learning to pass pipes. We mitigated this by increasing the Replay Buffer size.

### Future Ideas

- [ ] Implement Prioritized Experience Replay (PER) to focus on "hard" mistakes.
- [ ] Add Noisy Nets for better exploration parameters.

## 📂 File Structure

- agent.py: Contains the FlappyAgent class and Q-Learning logic.
- model.py: Defines the PyTorch DuelingDQN architecture.
- train.py: Main loop for training, logging, and saving checkpoints.
- utils.py: Helper functions for plotting and logging.
