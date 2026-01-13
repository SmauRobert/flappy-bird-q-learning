# Flappy Bird AI Agent (Dueling Double Deep Q-Learning)

## Assignment 5: Reinforcement Learning

**Team:** Hriscu Cosmin-Nicolas, Smău George-Robert
**Course:** Neural Networks
**Deadline:** Week 14

---

## Table of Contents

- [Architecture & Methodology](#-architecture--methodology)
- [Training Strategy](#-training-strategy)
- [Results & Performance](#-results--performance)
- [Challenges & Future Improvements](#-challenges--future-improvements)
- [File Structure](#-file-structure)

---

## Introduction

This project implements a Reinforcement Learning (RL) agent capable of playing **Flappy Bird** purely from raw pixels. The agent uses a **Dueling Double Deep Q-Network (D3QN)** to estimate optimal actions (`Jump` or `Do Nothing`) based on visual input.

The goal was to maximize the score by stabilizing the learning process using advanced RL techniques like **Experience Replay**, **Target Networks** and **Epsilon Greedy Exploration**

### Key Features

- **Input:** 4 stacked grayscale frames (84x84px).
- **Model:** Dueling DQN (Separates Value & Advantage streams).
- **Optimization:** Adam Optimizer with SmoothL1Loss.
- **Policy:** Epsilon-Greedy with compound decay.

---

### 1. Prerequisites

Make sure you have the required libraries installed:

```bash
pip install torch torchvision numpy opencv-python gym flappy-bird-gymnasium
```


## 🧠 Architecture & Methodology

### 1. Data Preprocessing

Instead of using raw game coordinates, we train on pixels to mimic human perception.

- **Grayscale Conversion**: Reduces input complexity (3 channels -> 1 channel).
- **Resizing**: Downscales frames to `84x84` for faster processing.
- **Normalization**: The pixel values are divided by `255.0` to bring them all into [0,1] 
- **Frame Stacking**: We stack the last **4 frames** to give the agent a sense of velocity and acceleration.

### 2. Neural Network (Dueling DQN)


**Layer Breakdown**:

- `Conv2d`: 32 filters, 8x8 kernel, stride 4.
Input: 4 channels of 84x84 
Output: 32 channels of 20x20
- `Conv2d`: 64 filters, 4x4 kernel, stride 2.
Input: 32 channels of 20x20
Output: 64 channels of 9x9
- `Conv2d`: 64 filters, 3x3 kernel, stride 1.
Input: 64 channels of 9x9
Output: 64 channels of 7x7
- `Flatten`: Turns the 3D output of the convolution layers into a 1D value.
- `Linear`: Splits into Value (1 output) and Advantage (2 outputs).

We utilize a Dueling Architecture which splits the network into two streams after the convolutional layers:

- **Value Stream V(s)**: Estimates the value of being in the current state.
- **Advantage Stream A(s,a)**: Estimates the benefit of taking a specific action.

$Q(s,a)=V(s)+\left(A(s,a)-\frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a')\right)$

### 3. Loss Function: SmoothL1Loss (Huber Loss)
Small errors -> MSE behaviour  
Large errors -> Liniar behaviour

### 4. Optimizer: ADAM (Adaptive Moment Estimation)
Optimization algorithm that combines ideas from both Momentum and RMSprop

## ⚙️ Training Strategy

### Hyperparameters

| Parameter         | Value  |                               Description |
| :---------------- | :----: | ----------------------------------------: |
| Batch Size        |   64   |       Number of samples per training step |
| Gamma             |  0.99  |        Discount factor for future rewards |
| Learning Rate     |  1e-4  |                  Adam Optimizer step size |
| Buffer Size       | 50,000 |                Experience Replay capacity |
| Epsilon Start     |  1.0   |                  Initial exploration rate |
| Epsilon End       |  0.01  |                    Final exploration rate |
| Epsilon Decay Rate|  0.998 | Rate we multiply epsilon with per episode |


### Q-Learning Algorithm

- `Replay Buffer`: breaks correlations that are present in recent transitions by training with a random sampled batch instead of the most recent transition
- `Target Network`: a second instance of the same model that is updated periodically and is used only to compute the target values in the Q-learning update
- `Epsilon-Greedy Exploration`: solves the exploration/exploitation problem by having the probability epsilon to make a random move at a given time


## 📊 Results & Performance

After training for 2400 episodes, our agent achieved the following stats:

- Max Score: 323 pipes

After training for 3500 episodes, the agent has achieved:

- Max Score: 757


## ⚠️ Challenges & Future Improvements

### Challenges

- **Sparse Rewards**: The agent takes a long time to find the first pipe, making initial learning slow.
- **Catastrophic Forgetting**: Sometimes the agent "forgets" how to fly safely after learning to pass pipes. We mitigated this by increasing the Replay Buffer size.



## 📂 File Structure

- flappy_bird.ipynb: Jupyter Notebook that contains the program
- checkpoints: folder that stores locally the models/graphs 
- README.md: markdown documentation
