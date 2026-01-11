from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import cv2 as cv
import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from typing_extensions import override

ImageStack = npt.NDArray[np.float32]
RawFrame = npt.NDArray[np.uint8]


class Environment(gym.Wrapper):
    def __init__(
        self, render_mode: str = "rgb_array", stack_size: int = 4, frame_skip: int = 4
    ):
        """
        Args:
            render_mode: The render mode for the internal environment.
            stack_size: Number of frames to stack for the observation (memory).
            frame_skip: Number of frames to repeat an action (physics speed).
        """
        env = gym.make("FlappyBird-v0", render_mode=render_mode)
        super().__init__(env)

        self.stack_size: int = stack_size
        self.frame_skip: int = frame_skip

        # Define the Observation Space
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.stack_size, 84, 84), dtype=np.float32
        )

        # Deque to store the last 'stack_size' frames
        self.frame_buffer: Deque[npt.NDArray[np.float32]] = deque(
            maxlen=self.stack_size
        )

    @override
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ImageStack, Dict[str, Any]]:
        """
        Resets the environment and fills the buffer with the first frame.
        """
        _, info = self.env.reset(seed=seed, options=options)

        # Capture initial frame
        frame: Optional[RawFrame] = self.env.render()  # type: ignore
        processed_frame: npt.NDArray[np.float32] = self._process_frame(frame)

        # Clear buffer and fill with the first frame duplicated
        self.frame_buffer.clear()
        for _ in range(self.stack_size):
            self.frame_buffer.append(processed_frame)

        return np.array(self.frame_buffer), info

    @override
    def step(self, action: int) -> Tuple[ImageStack, float, bool, bool, Dict[str, Any]]:
        """
        Steps the environment with Frame Skipping.
        """
        total_reward: float = 0.0
        terminated: bool = False
        truncated: bool = False
        info: Dict[str, Any] = {}

        # 1. Frame Skipping Loop (Repeat action)
        for _ in range(self.frame_skip):
            _, reward, term, trunc, info = self.env.step(action)

            # Accumulate reward (cast to float to ensure type safety)
            total_reward += float(reward)

            terminated = term
            truncated = trunc

            if terminated or truncated:
                break

        # 2. Capture New Frame
        frame: Optional[RawFrame] = self.env.render()  # type: ignore
        processed_frame: npt.NDArray[np.float32] = self._process_frame(frame)

        # 3. Update Buffer
        self.frame_buffer.append(processed_frame)

        # 4. Return Stacked Observation
        return np.array(self.frame_buffer), total_reward, terminated, truncated, info

    def _process_frame(self, frame: Optional[RawFrame]) -> npt.NDArray[np.float32]:
        """
        Converts (512, 288, 3) -> (84, 84) Grayscale Normalized
        """
        if frame is None:
            return np.zeros((84, 84), dtype=np.float32)

        # Grayscale
        grayscale = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        # Resize
        resized = cv.resize(grayscale, (84, 84), interpolation=cv.INTER_AREA)

        # Normalize to 0.0 - 1.0
        normalized = resized.astype(np.float32) / 255.0

        return normalized
