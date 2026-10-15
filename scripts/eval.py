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
from src.search import LookaheadSearch
from src.models import TeamPreviewModel
from src.players import LookaheadPlayer
from src.team_pool import ALL_ARCHETYPES, TeamInfo, classify_paste, extract_pokemon_names

def evaluate_neural_lookahead(
    model_path: str = "./checkpoints/best_model/best_model.zip",
    preview_model_path: str = "./checkpoints/best_model/best_team_preview_model.pt",
    team_path: str = "teams/MC646.txt",
    n_eval_episodes: int = 50,
    search_depth: int = 2,
    search_candidates: int = 8,
    opponent_type: str = "simple_heuristics",
    sample_teams: bool = True,
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
            max_candidates=8,
            accept_open_team_sheet=True,
            account_configuration=AccountConfiguration(f"Lookahead_{uuid.uuid4().hex[:6]}", None),
            team_preview_model=preview_model,
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

    raw_env = VGCEnv(opponent=opponent, sample_teams=sample_teams)
    raw_env.agent.update_team(team=team_paste)
    raw_env.agent.team_preview_model = preview_model

    fixed_team_info = None
    if not sample_teams:
        team_id = os.path.splitext(os.path.basename(team_path))[0]
        fixed_team_info = TeamInfo(
            team_id=team_id,
            file_path=team_path,
            paste=team_paste,
            pokemon_names=extract_pokemon_names(team_paste),
            archetypes=classify_paste(team_paste),
        )
        raw_env.agent_team = fixed_team_info
        raw_env.opponent_team = fixed_team_info

    searcher = LookaheadSearch(
        depth=search_depth,
        max_candidates=search_candidates,
    )

    print(f"\n=======================================================")
    print(f"  Evaluating Neural-Guided Lookahead AI")
    print(f"  Opponent: {opponent.__class__.__name__} ({opponent_type})")
    print(f"  Lookahead Depth: {search_depth} | Candidates: {search_candidates}")
    print(f"  Episodes: {n_eval_episodes} | Mode: Pure Deterministic Argmax")
    print(f"  Team Sampling: {'Multi-Team Stratified Pool' if sample_teams else f'Fixed Team ({team_path})'}")
    print(f"=======================================================\n")

    wins = 0
    losses = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    opponent_archetype_stats: dict[str, dict[str, int]] = {
        arch: {"wins": 0, "losses": 0, "total": 0} for arch in ALL_ARCHETYPES
    }
    agent_archetype_stats: dict[str, dict[str, int]] = {
        arch: {"wins": 0, "losses": 0, "total": 0} for arch in ALL_ARCHETYPES
    }

    for episode_index in range(1, n_eval_episodes + 1):
        observation, _ = raw_env.reset()
        if not sample_teams and raw_env.agent_team is None:
            raw_env.agent_team = fixed_team_info
            raw_env.opponent_team = fixed_team_info

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
                state_value_pred = float(model.policy.predict_values(tensor_obs).squeeze().item())

            top_nn_action = int(np.argmax(policy_probs))
            top_nn_prob = float(policy_probs[top_nn_action])

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

            agent_team, opponent_team = raw_env.agent_team, raw_env.opponent_team
            team_info = (
                f"Agent Team: {agent_team.team_id}, {agent_team.archetypes} | Opponent Team: {opponent_team.team_id}, {opponent_team.archetypes}"
                if agent_team and opponent_team
                else "Teams not sampled"
            )
            print(
                f"    [Turn {current_battle.turn:2d} (Step {turn_count:2d})] "
                f"Action: {chosen_action_code:3d} (NN Top: {top_nn_action:3d} p={top_nn_prob:.2f} V={state_value_pred:+.2f}) | "
                f"Step Reward: {reward:+.3f} | Total: {cumulative_reward:+.3f} | "
                f"{team_info}"
            )

        is_win = cumulative_reward > 0.0
        if is_win:
            wins += 1
        else:
            losses += 1

        episode_rewards.append(cumulative_reward)
        episode_lengths.append(turn_count)
        running_win_rate = (wins / episode_index) * 100.0

        agent_team, opponent_team = raw_env.agent_team, raw_env.opponent_team
        team_info = (
            f"Agent Team: {agent_team.team_id}, {agent_team.archetypes} | Opponent Team: {opponent_team.team_id}, {opponent_team.archetypes}"
            if agent_team and opponent_team
            else "Teams not sampled"
        )

        agent_archetypes = agent_team.archetypes if agent_team and agent_team.archetypes else {"other"}
        opponent_archetypes = opponent_team.archetypes if opponent_team and opponent_team.archetypes else {"other"}

        for arch in opponent_archetypes:
            opponent_archetype_stats.setdefault(arch, {"wins": 0, "losses": 0, "total": 0})
            opponent_archetype_stats[arch]["total"] += 1
            if is_win:
                opponent_archetype_stats[arch]["wins"] += 1
            else:
                opponent_archetype_stats[arch]["losses"] += 1

        for arch in agent_archetypes:
            agent_archetype_stats.setdefault(arch, {"wins": 0, "losses": 0, "total": 0})
            agent_archetype_stats[arch]["total"] += 1
            if is_win:
                agent_archetype_stats[arch]["wins"] += 1
            else:
                agent_archetype_stats[arch]["losses"] += 1

        print(
            f"Battle {episode_index:3d}/{n_eval_episodes:3d} | "
            f"Result: {'WIN ' if is_win else 'LOSS'} | "
            f"Turns: {turn_count:2d} | "
            f"Reward: {cumulative_reward:+.2f} | "
            f"Running WinRate: {running_win_rate:5.1f}% ({wins}W - {losses}L) | "
            f"{team_info}"
        )
        opp_strat_info = ", ".join(
            f"{a}: {opponent_archetype_stats[a]['wins']}/{opponent_archetype_stats[a]['total']} ({(opponent_archetype_stats[a]['wins']/opponent_archetype_stats[a]['total'])*100:.1f}%)"
            for a in sorted(opponent_archetypes)
        )
        print(f"    Running Stratified vs Opponent Archetypes: [{opp_strat_info}]")

    final_win_rate = (wins / max(1, n_eval_episodes)) * 100.0
    final_mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    final_mean_turns = float(np.mean(episode_lengths)) if episode_lengths else 0.0

    print(f"\n================ Final Evaluation Summary ================")
    print(f"Record: {wins}W - {losses}L ({final_win_rate:.1f}%)")
    print(f"Mean Reward: {final_mean_reward:+.2f}")
    print(f"Average Turns: {final_mean_turns:.1f}")
    print(f"----------------------------------------------------------")
    print(f"Stratified Win Rate by Opponent Archetype (Matchup vs):")
    for arch in ALL_ARCHETYPES:
        stats = opponent_archetype_stats.get(arch, {"wins": 0, "losses": 0, "total": 0})
        total = stats["total"]
        if total > 0:
            pct = (stats["wins"] / total) * 100.0
            print(f"  {arch:14s}: {stats['wins']:2d}/{total:2d} ({pct:5.1f}%) [ {stats['wins']:2d}W - {stats['losses']:2d}L ]")
        else:
            print(f"  {arch:14s}:  0/ 0 (  N/A) [  0W -  0L ]")
    print(f"----------------------------------------------------------")
    print(f"Stratified Win Rate by Agent Archetype (Playing as):")
    for arch in ALL_ARCHETYPES:
        stats = agent_archetype_stats.get(arch, {"wins": 0, "losses": 0, "total": 0})
        total = stats["total"]
        if total > 0:
            pct = (stats["wins"] / total) * 100.0
            print(f"  {arch:14s}: {stats['wins']:2d}/{total:2d} ({pct:5.1f}%) [ {stats['wins']:2d}W - {stats['losses']:2d}L ]")
        else:
            print(f"  {arch:14s}:  0/ 0 (  N/A) [  0W -  0L ]")
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
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--depth", type=int, default=2, help="Lookahead search depth")
    parser.add_argument("--candidates", type=int, default=8, help="Candidate pool limit per slot")
    parser.add_argument(
        "--opponent",
        type=str,
        default="lookahead",
        choices=["simple_heuristics", "lookahead", "random"],
        help="Opponent player type"
    )
    parser.add_argument("--team_path", type=str, default="teams/MC646.txt", help="Path to default team file if not sampling")
    parser.add_argument(
        "--sample_teams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample diverse competitive teams from team pool for multi-archetype evaluation"
    )
    args = parser.parse_args()

    evaluate_neural_lookahead(
        model_path=args.model,
        preview_model_path=args.preview_model,
        team_path=args.team_path,
        n_eval_episodes=args.episodes,
        search_depth=args.depth,
        search_candidates=args.candidates,
        opponent_type=args.opponent,
        sample_teams=args.sample_teams,
    )

if __name__ == "__main__":
    main()