import gymnasium as gym
import metaworld
import cv2
import numpy as np
from collections import deque


# recommended cameras per task — pass explicitly to override default
TASK_CAMERAS = {
    "reach-v3":                  "corner2",
    "push-v3":                   "corner2",
    "pick-place-v3":             "corner2",
    "door-open-v3":              "corner",
    "drawer-close-v3":           "corner",
    "drawer-open-v3":            "corner",
    "button-press-topdown-v3":   "corner3",
    "peg-insert-side-v3":        "corner2",
    "window-open-v3":            "corner",
    "window-close-v3":           "corner",
}


class MetaWorldEnv:
    def __init__(
        self,
        task_name   = "reach-v3",
        img_size    = 84,
        render_mode = "rgb_array",
        obs_horizon = 3,
        camera_name = None,         # None = MetaWorld default camera
    ):
        self.img_size    = img_size
        self.obs_horizon = obs_horizon
        self.camera_name = camera_name   # only used if explicitly passed

        self.env = gym.make(
            "Meta-World/MT1",
            env_name    = task_name,
            render_mode = render_mode,
            width       = img_size,
            height      = img_size,
        )

        self.frame_buffer = deque(maxlen=obs_horizon)

    def _process_image(self, img):
        if img is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        if img.shape[:2] != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)
        return img.astype(np.uint8)

    def _render(self):
        if self.camera_name is not None:
            # use explicitly requested camera
            try:
                img = self.env.unwrapped.mujoco_renderer.render(
                    render_mode = "rgb_array",
                    camera_name = self.camera_name,
                )
                return self._process_image(img)
            except Exception:
                pass  # fallthrough to default
        # default MetaWorld camera
        return self._process_image(self.env.render())

    def _get_stacked_image(self):
        return np.concatenate(list(self.frame_buffer), axis=-1)

    def reset(self):
        obs, info = self.env.reset()
        frame = self._render()
        for _ in range(self.obs_horizon):
            self.frame_buffer.append(frame)
        return {"state": obs, "image": self._get_stacked_image()}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.frame_buffer.append(self._render())
        return {"state": obs, "image": self._get_stacked_image()}, reward, done, info

    def close(self):
        self.env.close()