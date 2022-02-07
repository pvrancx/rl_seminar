from gym.envs.registration import register


register(
    id="MountainCarLong-v0",
    entry_point="gym.envs.classic_control:MountainCarEnv",
    max_episode_steps=5000,
    reward_threshold=-110.0,
)