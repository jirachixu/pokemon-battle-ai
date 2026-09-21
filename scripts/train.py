from src.env import VGCEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor
from torch.cuda import is_available
from poke_env.player import SimpleHeuristicsPlayer, RandomPlayer

with open("teams/sample_team.txt", "r") as f:
    team = f.read().strip()

opponent = SimpleHeuristicsPlayer(battle_format="gen9championsvgc2026regmc", team=team)
env = VGCEnv(opponent=opponent)
env.agent.update_team(team=team)

eval_opponent = SimpleHeuristicsPlayer(battle_format="gen9championsvgc2026regmc", team=team)
eval_env_raw = VGCEnv(opponent=eval_opponent)
eval_env_raw.agent.update_team(team=team)
eval_env = Monitor(eval_env_raw)

policy_kwargs = dict(
    net_arch=dict(pi=[512, 256], vf=[512, 256])
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000, save_path="./checkpoints/history/", 
    name_prefix="maskable_ppo_model"
)
eval_callback = MaskableEvalCallback(
    eval_env=eval_env,
    eval_freq=10000,
    n_eval_episodes=25,
    deterministic=True,
    best_model_save_path="./checkpoints/best_model"
)

model = MaskablePPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./logs/",
    device="cuda" if is_available() else "cpu"
)

# model = MaskablePPO.load(
#     "./checkpoints/best_model/best_model.zip",
#     device="cuda" if is_available() else "cpu",
#     env=env
# )

model.learn(total_timesteps=100000, callback=[checkpoint_callback, eval_callback])