import gymnasium as gym
import metaworld
import cv2
import numpy as np
from collections import deque


class MetaWorldEnv:
    def __init__(self, task_name="reach-v3", img_size=224, render_mode="rgb_array", obs_horizon=2):
        self.img_size    = img_size
        self.obs_horizon = obs_horizon
        self.env = gym.make(
            "Meta-World/MT1",
            env_name=task_name,
            render_mode=render_mode,
            width=img_size,
            height=img_size,
        )
        # ring buffer — always keeps last obs_horizon frames
        self.frame_buffer = deque(maxlen=obs_horizon)

    def _process_image(self, img):
        if img is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return img.astype(np.uint8)

    def _get_stacked_image(self):
        # stack along channel dim → (H, W, 3*obs_horizon)
        return np.concatenate(list(self.frame_buffer), axis=-1)

    def reset(self):
        obs, info = self.env.reset()
        frame = self._process_image(self.env.render())
        # fill buffer with same frame on reset — no fake motion signal
        for _ in range(self.obs_horizon):
            self.frame_buffer.append(frame)
        return {"state": obs, "image": self._get_stacked_image()}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.frame_buffer.append(self._process_image(self.env.render()))
        return {"state": obs, "image": self._get_stacked_image()}, reward, done, info

    def close(self):
        self.env.close()