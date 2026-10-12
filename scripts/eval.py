"""
Evaluation script for Pokemon VGC AI.
Benchmarks the full Neural-Guided Lookahead AI (Neural Policy Prior + 2-Turn Nash Equilibrium Search)
against competitive baselines (SimpleHeuristicsPlayer, LookaheadPlayer, RandomPlayer).
"""

from __future__ import annotations
import os
import argparse
import uuid
import numpy as np
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer, RandomPlayer, Player

from src.env import VGCEnv
from src.simulation import SimulatedBattleState
from src.search import LookaheadSearch, LookaheadPlayer
from src.models import TeamPreviewModel

def evaluate_neural_lookahead(
    model_path: str = "./checkpoints/best_model/best_model.zip",
    preview_model_path: str = "./checkpoints/best_model/best_team_preview_model.pt",
    team_path: str = "teams/MC306.txt",
    n_eval_episodes: int = 50,
    search_depth: int = 2,
    search_candidates: int = 6,
    opponent_type: str = "simple_heuristics",
) -> tuple[float, float, float]:
    """
    Evaluates the trained neural-guided lookahead player against a specified opponent.
    Returns (win_rate, mean_reward, mean_turns).
    """
    with open(team_path, "r", encoding="utf-8") as file_handle:
        team_paste = file_handle.read().strip()

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading trained neural model from: {model_path} on {device_name}...")
    model = MaskablePPO.load(model_path, device=device_name)

    print(f"Loading trained team preview model from: {preview_model_path} on {device_name}...")
    preview_model = TeamPreviewModel(ability_dim=32, item_dim=32).to(device_name)
    preview_model.load_state_dict(torch.load(preview_model_path, map_location=device_name))
    preview_model.eval()

    opponent: Player
    if opponent_type == "lookahead":
        opponent = LookaheadPlayer(
            battle_format="gen9championsvgc2026regmc",
            team=team_paste,
            depth=2,
            max_candidates=4,
            accept_open_team_sheet=True,
            account_configuration=AccountConfiguration(f"Lookahead_{uuid.uuid4().hex[:6]}", None),
        )
    elif opponent_type == "random":
        opponent = RandomPlayer(
            battle_format="gen9championsvgc2026regmc",
            team=team_paste,
            accept_open_team_sheet=True,
            account_configuration=AccountConfiguration(f"Random_{uuid.uuid4().hex[:6]}", None),
        )
    else:
        opponent = SimpleHeuristicsPlayer(
            battle_format="gen9championsvgc2026regmc",
            team=team_paste,
            accept_open_team_sheet=True,
            account_configuration=AccountConfiguration(f"Heuristic_{uuid.uuid4().hex[:6]}", None),
        )

    raw_env = VGCEnv(opponent=opponent)
    raw_env.agent.update_team(team=team_paste)
    raw_env.agent.team_preview_model = preview_model

    searcher = LookaheadSearch(
        depth=search_depth,
        max_candidates=search_candidates,
    )

    print(f"\n=======================================================")
    print(f"  Evaluating Neural-Guided Lookahead AI")
    print(f"  Opponent: {opponent.__class__.__name__} ({opponent_type})")
    print(f"  Lookahead Depth: {search_depth} | Candidates: {search_candidates}")
    print(f"  Episodes: {n_eval_episodes} | Mode: Pure Deterministic Argmax")
    print(f"=======================================================\n")

    wins = 0
    losses = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for episode_index in range(1, n_eval_episodes + 1):
        observation, _ = raw_env.reset()
        is_terminated = False
        is_truncated = False
        cumulative_reward = 0.0
        turn_count = 0

        while not (is_terminated or is_truncated):
            current_battle = raw_env.current_battle
            current_action_mask = raw_env.action_masks()

            # 1. Human-like Intuition: Neural Network Policy Prior
            tensor_obs, _ = model.policy.obs_to_tensor(observation)
            with torch.no_grad():
                distribution = model.policy.get_distribution(tensor_obs, action_masks=current_action_mask)
                if not isinstance(distribution, MaskableCategoricalDistribution):
                    raise TypeError("Expected MaskableCategoricalDistribution from policy")
                categorical_distribution = distribution.distribution
                policy_probs_tensor: torch.Tensor = categorical_distribution.probs
                policy_probs = policy_probs_tensor.squeeze(0).cpu().numpy()

            # 2. Human-like Calculation: 2-Turn Nash Equilibrium Game Tree Search
            simulated_state = SimulatedBattleState.from_battle(current_battle, fill_unrevealed=True)
            search_result = searcher.search(simulated_state, policy_prior=policy_probs)

            # 3. Deterministic Execution: Pure Argmax of the Verified Best Line
            chosen_action_code = int(min(675, max(0, searcher.to_env_action(search_result.best_action, current_battle))))
            if not current_action_mask[chosen_action_code]:
                legal_indices = np.where(current_action_mask)[0]
                if len(legal_indices) > 0:
                    chosen_action_code = int(legal_indices[0])

            observation, reward, is_terminated, is_truncated, _ = raw_env.step(chosen_action_code)
            cumulative_reward += reward
            turn_count += 1

        is_win = cumulative_reward > 0.0
        if is_win:
            wins += 1
        else:
            losses += 1

        episode_rewards.append(cumulative_reward)
        episode_lengths.append(turn_count)
        running_win_rate = (wins / episode_index) * 100.0

        print(
            f"Battle {episode_index:3d}/{n_eval_episodes:3d} | "
            f"Result: {'WIN ' if is_win else 'LOSS'} | "
            f"Turns: {turn_count:2d} | "
            f"Reward: {cumulative_reward:+.2f} | "
            f"Running WinRate: {running_win_rate:5.1f}% ({wins}W - {losses}L)"
        )

    final_win_rate = (wins / max(1, n_eval_episodes)) * 100.0
    final_mean_reward = float(np.mean(episode_rewards))
    final_mean_turns = float(np.mean(episode_lengths))

    print(f"\n================ Final Evaluation Summary ================")
    print(f"Record: {wins}W - {losses}L ({final_win_rate:.1f}%)")
    print(f"Mean Reward: {final_mean_reward:.2f}")
    print(f"Average Turns: {final_mean_turns:.1f}")
    print(f"==========================================================\n")

    return final_win_rate, final_mean_reward, final_mean_turns

def main():
    parser = argparse.ArgumentParser(description="Evaluate Neural-Guided Lookahead AI on Pokemon VGC")
    parser.add_argument("--model", type=str, default="./checkpoints/best_model/best_model.zip", help="Path to trained model")
    parser.add_argument(
        "--preview_model",
        type=str,
        default="./checkpoints/best_model/best_team_preview_model.pt",
        help="Path to trained team preview model checkpoint"
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--depth", type=int, default=2, help="Lookahead search depth")
    parser.add_argument("--candidates", type=int, default=6, help="Candidate pool limit per slot")
    parser.add_argument(
        "--opponent",
        type=str,
        default="simple_heuristics",
        choices=["simple_heuristics", "lookahead", "random"],
        help="Opponent player type"
    )
    args = parser.parse_args()

    evaluate_neural_lookahead(
        model_path=args.model,
        preview_model_path=args.preview_model,
        n_eval_episodes=args.episodes,
        search_depth=args.depth,
        search_candidates=args.candidates,
        opponent_type=args.opponent,
    )

if __name__ == "__main__":
    main()