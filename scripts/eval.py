from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from torch.cuda import is_available
from src.env import VGCEnv
from poke_env.player import SimpleHeuristicsPlayer
import numpy as np

with open("teams/sample_team.txt", "r") as f:
    team = f.read().strip()

opponent = SimpleHeuristicsPlayer(battle_format="gen9championsvgc2026regmc", team=team)
env_raw = VGCEnv(opponent=opponent)
env_raw.agent.update_team(team=team)
env = Monitor(env_raw)

model = MaskablePPO.load(
    "./checkpoints/best_model/best_model.zip",
    device="cuda" if is_available() else "cpu"
)

print("Evaluating the trained model against a SimpleHeuristicsPlayer opponent...")
ep_rewards, ep_lengths = evaluate_policy(
    model,
    env,
    n_eval_episodes=100,
    deterministic=True,
    return_episode_rewards=True
)

assert isinstance(ep_rewards, list) and isinstance(ep_lengths, list), "Expected lists for rewards and lengths"

mean_reward = float(np.mean(ep_rewards))
wins = sum(1 for r in ep_rewards if r > 0)
losses = sum(1 for r in ep_rewards if r < 0)
win_rate = wins / len(ep_rewards)
print(f"Record: {wins}W - {losses}L ({win_rate * 100:.1f}%)")
print(f"Mean Reward: {mean_reward:.2f}")
print(f"Average Turns: {float(np.mean(ep_lengths)):.1f}")