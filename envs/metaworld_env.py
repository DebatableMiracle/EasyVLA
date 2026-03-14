import gymnasium as gym
import cv2
import numpy as np
import metaworld

class MetaWorldEnv:
    def __init__(self, task_name="reach-v3", img_size=224):
        self.img_size = img_size
        self.env = gym.make(
            "Meta-World/MT1",
            env_name=task_name,
            render_mode="human",   
            width=img_size,           
            height=img_size,           
        )

    def _process_image(self, img):
        if img is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)  # INTER_AREA is best for downscaling
        return img.astype(np.uint8)

    def reset(self):
        obs, info = self.env.reset()
        return {"state": obs, "image": self._process_image(self.env.render())}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return {"state": obs, "image": self._process_image(self.env.render())}, reward, done, info

    def close(self):
        self.env.close()