# utils/check_episode_lengths.py
import numpy as np
from envs.metaworld_env import MetaWorldEnv
from metaworld.policies import (
    SawyerReachV3Policy,
    SawyerPushV3Policy,
    SawyerDoorOpenV3Policy,
    SawyerDrawerCloseV3Policy,
    SawyerButtonPressTopdownV3Policy,
)

POLICY_MAP = {
    "reach-v3":                  SawyerReachV3Policy,
    "push-v3":                   SawyerPushV3Policy,
    "door-open-v3":              SawyerDoorOpenV3Policy,
    "drawer-close-v3":           SawyerDrawerCloseV3Policy,
    "button-press-topdown-v3":   SawyerButtonPressTopdownV3Policy,
}

EPISODES = 50

for task, policy_cls in POLICY_MAP.items():
    env    = MetaWorldEnv(task, img_size=84, obs_horizon=3)
    policy = policy_cls()
    steps_list = []
    success_count = 0

    for _ in range(EPISODES):
        obs  = env.reset()
        done = False
        step = 0
        while not done and step < 500:
            action = policy.get_action(obs["state"])
            obs, _, done, info = env.step(action)
            step += 1
            if int(info.get("success", 0)) == 1:
                done = True
                success_count += 1
        steps_list.append(step)

    steps = np.array(steps_list)
    print(f"{task:<35} mean: {steps.mean():>5.0f}  max: {steps.max():>4}  "
          f"min: {steps.min():>3}  success: {success_count}/{EPISODES}")
    env.close()