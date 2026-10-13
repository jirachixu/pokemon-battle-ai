"""
Multi-turn lookahead search engine for competitive Doubles Pokemon battles (VGC / Champions).
Simulates 2-3 turns ahead using Simultaneous Matrix Game Solving (Nash Equilibrium via Regret Matching).
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Any, Sequence

from src.simulation import (
    SimulatedBattleState,
    SimulatedPokemon,
    SimulatedMove,
    SimulatedAction,
    step_simulation,
    clean_id,
)
import src.constants as constants
from src.evaluator import BattleStateEvaluator, CompetitiveHeuristicEvaluator
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder, PassBattleOrder
from poke_env.ps_client.account_configuration import AccountConfiguration
import uuid
from src.constants import (
    PROTECT_MOVES,
    STATUS_SUPPORT_MOVE_IDS,
    SELF_OR_FIELD_MOVES,
    ALLY_TARGET_MOVES,
)

def is_self_or_field_move(move: SimulatedMove) -> bool:
    """Checks if a move targets self, ally side, or the whole field."""
    if move.target in ("self", "all", "allies", "allySide", "allyTeam"):
        return True
    return move.id in SELF_OR_FIELD_MOVES

def is_ally_buff_move(move: SimulatedMove) -> bool:
    """Checks if a move targets an adjacent partner Pokemon."""
    if move.target in ("adjacentAlly", "adjacentAllyOrSelf"):
        return True
    return move.id in ALLY_TARGET_MOVES

def get_effective_base_power(move: SimulatedMove) -> float:
    """Returns effective base power for moves, properly estimating variable-power attacks."""
    if move.category == "status":
        return 0.0
    if move.base_power > 0:
        return move.base_power
    mid = move.id
    if mid in ("lowkick", "grassknot", "heavyslam", "heatcrash", "gyroball", "electroball"):
        return 80.0
    if mid == "foulplay":
        return 95.0
    if mid in ("seismictoss", "nightshade"):
        return 70.0  # Approx 50 flat HP damage proxy
    if mid in ("superfang", "ruination", "naturesmadness"):
        return 80.0  # 50% current HP proxy
    if mid == "finalgambit":
        return 120.0
    if mid in ("counter", "mirrorcoat"):
        return 80.0
    return 60.0

def is_protect_action(action: SimulatedAction) -> bool:
    """Checks if action is a Protect-type damage negation move."""
    return action.type == "move" and action.move_id is not None and action.move_id in constants.PROTECT_MOVES

def is_switch_action(action: SimulatedAction) -> bool:
    """Checks if action is a switch action."""
    return action.type == "switch"

def is_support_action(action: SimulatedAction) -> bool:
    """Identifies whether an action is a simulated support, status, healing, or setup move."""
    if action.type != "move" or not action.move_id:
        return False
    if action.target_idx == 2:  # Explicitly targeting ally (Pollen Puff healing, Coaching, Helping Hand)
        return True
    if action.move_id in constants.PROTECT_MOVES:
        return False
    return action.move_id in STATUS_SUPPORT_MOVE_IDS

def solve_zero_sum_matrix_game(
    payoff_matrix: np.ndarray,
    iterations: int = 250
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves a 2-player zero-sum normal-form game matrix (num_player1_actions x num_player2_actions)
    using Counterfactual Regret Matching (CFR).
    Returns (strategy_player1, strategy_player2, game_value).
    """
    num_player1_actions, num_player2_actions = payoff_matrix.shape
    if num_player1_actions == 1 and num_player2_actions == 1:
        return np.array([1.0]), np.array([1.0]), float(payoff_matrix[0, 0])
    if num_player1_actions == 1:
        # Player 1 has only 1 move; Player 2 picks the column that minimizes payoff
        best_column_index = int(np.argmin(payoff_matrix[0, :]))
        player2_strategy = np.zeros(num_player2_actions)
        player2_strategy[best_column_index] = 1.0
        return np.array([1.0]), player2_strategy, float(payoff_matrix[0, best_column_index])
    if num_player2_actions == 1:
        best_row_index = int(np.argmax(payoff_matrix[:, 0]))
        player1_strategy = np.zeros(num_player1_actions)
        player1_strategy[best_row_index] = 1.0
        return player1_strategy, np.array([1.0]), float(payoff_matrix[best_row_index, 0])

    regrets_player1 = np.zeros(num_player1_actions, dtype=np.float64)
    regrets_player2 = np.zeros(num_player2_actions, dtype=np.float64)
    strategy_sum_player1 = np.zeros(num_player1_actions, dtype=np.float64)
    strategy_sum_player2 = np.zeros(num_player2_actions, dtype=np.float64)

    for _ in range(iterations):
        # Player 1 positive regrets
        positive_regrets_player1 = np.maximum(regrets_player1, 0.0)
        sum_positive_regrets_player1 = positive_regrets_player1.sum()
        current_strategy_player1 = (
            positive_regrets_player1 / sum_positive_regrets_player1
            if sum_positive_regrets_player1 > 0
            else np.ones(num_player1_actions) / num_player1_actions
        )
        strategy_sum_player1 += current_strategy_player1

        # Player 2 positive regrets (minimizer)
        positive_regrets_player2 = np.maximum(regrets_player2, 0.0)
        sum_positive_regrets_player2 = positive_regrets_player2.sum()
        current_strategy_player2 = (
            positive_regrets_player2 / sum_positive_regrets_player2
            if sum_positive_regrets_player2 > 0
            else np.ones(num_player2_actions) / num_player2_actions
        )
        strategy_sum_player2 += current_strategy_player2

        # Expected payoffs
        expected_payoff_player1 = payoff_matrix @ current_strategy_player2
        regrets_player1 += expected_payoff_player1 - np.dot(current_strategy_player1, expected_payoff_player1)

        expected_payoff_player2 = -current_strategy_player1 @ payoff_matrix
        regrets_player2 += expected_payoff_player2 - np.dot(current_strategy_player2, expected_payoff_player2)

    average_strategy_player1 = strategy_sum_player1 / iterations
    average_strategy_player2 = strategy_sum_player2 / iterations
    game_value = float(average_strategy_player1 @ payoff_matrix @ average_strategy_player2)
    return average_strategy_player1, average_strategy_player2, game_value

class SearchResult:
    """Encapsulates the result of a multi-turn lookahead search."""
    __slots__ = (
        "best_action", "value", "action_distribution", "predicted_opponent_actions",
        "depth", "nodes_evaluated", "method"
    )

    best_action: tuple[SimulatedAction, SimulatedAction]
    value: float
    action_distribution: dict[tuple[SimulatedAction, SimulatedAction], float]
    predicted_opponent_actions: dict[tuple[SimulatedAction, SimulatedAction], float]
    depth: int
    nodes_evaluated: int
    method: str

    def __init__(
        self,
        best_action: tuple[SimulatedAction, SimulatedAction],
        value: float,
        action_distribution: dict[tuple[SimulatedAction, SimulatedAction], float],
        predicted_opponent_actions: dict[tuple[SimulatedAction, SimulatedAction], float],
        depth: int,
        nodes_evaluated: int,
        method: str = "nash",
    ) -> None:
        self.best_action = best_action
        self.value = value
        self.action_distribution = action_distribution
        self.predicted_opponent_actions = predicted_opponent_actions
        self.depth = depth
        self.nodes_evaluated = nodes_evaluated
        self.method = method

    def __repr__(self) -> str:
        return (
            f"SearchResult(method='{self.method}', depth={self.depth}, "
            f"value={self.value:.1f}, best_action={self.best_action}, "
            f"nodes={self.nodes_evaluated})"
        )

class LookaheadSearch:
    """
    Simulates 2-3 turns ahead in Doubles battles using game-theoretic Nash equilibrium
    solving via Counterfactual Regret Matching (CFR).
    """

    evaluator: BattleStateEvaluator
    depth: int
    method: str
    max_candidates: int
    nodes_evaluated: int

    def __init__(
        self,
        evaluator: Optional[BattleStateEvaluator] = None,
        depth: int = 2,
        max_candidates: int = 10,
        **kwargs: Any,
    ) -> None:
        self.evaluator = evaluator or CompetitiveHeuristicEvaluator()
        self.depth = depth
        self.method = "nash"
        self.max_candidates = max_candidates
        self.nodes_evaluated = 0

    def generate_candidate_actions(
        self,
        state: SimulatedBattleState,
        side: int,
        max_candidates: Optional[int] = None,
        policy_prior: Optional[np.ndarray] = None,
    ) -> list[tuple[SimulatedAction, SimulatedAction]]:
        """
        Generates high-priority competitive joint candidate actions for the given side in Doubles.
        Filters out redundant or strictly dominated actions to prevent combinatorial explosion.
        """
        candidate_limit = max_candidates or self.max_candidates
        return self._generate_doubles_candidates(state, side, candidate_limit, policy_prior=policy_prior)

    def _generate_doubles_candidates(
        self,
        state: SimulatedBattleState,
        side: int,
        candidate_limit: int,
        policy_prior: Optional[np.ndarray] = None,
    ) -> list[tuple[SimulatedAction, SimulatedAction]]:
        """Generates top joint candidate actions (slot 0, slot 1) for Doubles."""
        active_pokemon_list = state.side1_active if side == 1 else state.side2_active
        bench_pokemon_list = state.side1_bench if side == 1 else state.side2_bench
        opponent_pokemon_list = state.side2_active if side == 1 else state.side1_active

        slot0_candidates = self._get_slot_candidates(state, active_pokemon_list, bench_pokemon_list, opponent_pokemon_list, slot=0)
        slot1_candidates = self._get_slot_candidates(state, active_pokemon_list, bench_pokemon_list, opponent_pokemon_list, slot=1)

        joint_candidates: list[tuple[float, tuple[SimulatedAction, SimulatedAction]]] = []

        for slot0_score, action_slot0 in slot0_candidates:
            for slot1_score, action_slot1 in slot1_candidates:
                # Disallow both slots switching to the same bench Pokémon
                if action_slot0.type == "switch" and action_slot1.type == "switch" and action_slot0.switch_idx == action_slot1.switch_idx:
                    continue
                # Disallow both slots Mega Evolving simultaneously
                if action_slot0.mega and action_slot1.mega:
                    continue
                # If both slots are forced to switch and bench mons remain, do not permit double pass
                alive_bench_count = sum(1 for p in bench_pokemon_list if p and not p.fainted)
                if any(state.force_switch) and all(state.force_switch) and alive_bench_count > 0:
                    if action_slot0.type == "pass" and action_slot1.type == "pass":
                        continue

                joint_score = slot0_score + slot1_score

                # Protect + Partner play synergy
                is_protect_slot0 = is_protect_action(action_slot0)
                is_protect_slot1 = is_protect_action(action_slot1)
                if (is_protect_slot0 and not is_protect_slot1) or (is_protect_slot1 and not is_protect_slot0):
                    joint_score += 15.0  # Protect + partner attack/support synergy
                elif is_protect_slot0 and is_protect_slot1:
                    joint_score -= 40.0  # Double protect usually passive / risky

                # Redirection (Follow Me / Rage Powder) synergy
                is_redir_slot0 = action_slot0.type == "move" and action_slot0.move_id in ("followme", "ragepowder")
                is_redir_slot1 = action_slot1.type == "move" and action_slot1.move_id in ("followme", "ragepowder")
                if (is_redir_slot0 and not is_redir_slot1) or (is_redir_slot1 and not is_redir_slot0):
                    joint_score += 20.0  # Redirection + partner setup / attack synergy
                elif is_redir_slot0 and is_redir_slot1:
                    joint_score -= 80.0  # Double redirection is completely redundant

                # Wide Guard synergy
                is_wideguard_0 = action_slot0.type == "move" and action_slot0.move_id == "wideguard"
                is_wideguard_1 = action_slot1.type == "move" and action_slot1.move_id == "wideguard"
                if is_wideguard_0 and is_wideguard_1:
                    joint_score -= 80.0

                # Attacking synergy: Balance focus-firing vs split-attacking to ensure hedging lines exist
                if action_slot0.type == "move" and action_slot1.type == "move":
                    if action_slot0.target_idx in (0, 1) and action_slot1.target_idx in (0, 1):
                        if action_slot0.target_idx == action_slot1.target_idx:
                            joint_score += 15.0  # Moderate focus-fire bonus (not +60 which crowds out everything)
                        else:
                            joint_score += 15.0  # Split attack bonus (hedges against opponent Protect!)

                # Integrate learned neural network policy prior (AlphaZero guidance)
                if policy_prior is not None and side == 1:
                    raw_battle_ref = getattr(state, "_raw_battle_reference", None)
                    slot0_code = self._slot_action_to_int(action_slot0, slot_index=0, battle=raw_battle_ref)
                    slot1_code = self._slot_action_to_int(action_slot1, slot_index=1, battle=raw_battle_ref)
                    action_code = int(min(675, max(0, slot0_code * 26 + slot1_code)))
                    if 0 <= action_code < len(policy_prior):
                        joint_score += 25.0 * float(policy_prior[action_code])

                joint_candidates.append((joint_score, (action_slot0, action_slot1)))

        joint_candidates.sort(key=lambda joint_entry: joint_entry[0], reverse=True)

        if any(state.force_switch) or len(joint_candidates) <= candidate_limit:
            return [joint_entry[1] for joint_entry in joint_candidates[:candidate_limit]]

        # Multi-Archetype Stratified Candidate Selection:
        # Categorizes candidate pairs by tactical pattern (Focus Opp 0, Focus Opp 1, Split,
        # Protect Slot 0, Protect Slot 1, Support/Disruption, and Strategic Switches).
        # This mathematically guarantees that no line of play (especially attacking the other
        # opponent or defending) is ever squeezed out or blinded in the lookahead search tree.
        bucket_focus_opp0: list[tuple[SimulatedAction, SimulatedAction]] = []
        bucket_focus_opp1: list[tuple[SimulatedAction, SimulatedAction]] = []
        bucket_split: list[tuple[SimulatedAction, SimulatedAction]] = []
        bucket_protect0: list[tuple[SimulatedAction, SimulatedAction]] = []
        bucket_protect1: list[tuple[SimulatedAction, SimulatedAction]] = []
        bucket_support: list[tuple[SimulatedAction, SimulatedAction]] = []
        bucket_switch: list[tuple[SimulatedAction, SimulatedAction]] = []

        for _, action_pair in joint_candidates:
            act0, act1 = action_pair
            is_p0 = is_protect_action(act0)
            is_p1 = is_protect_action(act1)
            is_s0 = is_switch_action(act0)
            is_s1 = is_switch_action(act1)
            is_supp0 = is_support_action(act0)
            is_supp1 = is_support_action(act1)

            if is_p0 and not is_p1:
                bucket_protect0.append(action_pair)
            elif is_p1 and not is_p0:
                bucket_protect1.append(action_pair)
            elif is_s0 or is_s1:
                bucket_switch.append(action_pair)
            elif is_supp0 or is_supp1:
                bucket_support.append(action_pair)
            elif act0.type == "move" and act1.type == "move":
                # Both attacking: distinguish target orientation
                if act0.target_idx == 0 and act1.target_idx == 0:
                    bucket_focus_opp0.append(action_pair)
                elif act0.target_idx == 1 and act1.target_idx == 1:
                    bucket_focus_opp1.append(action_pair)
                else:
                    bucket_split.append(action_pair)

        selected_candidates: list[tuple[SimulatedAction, SimulatedAction]] = []
        selected_set: set[tuple[SimulatedAction, SimulatedAction]] = set()

        def add_candidate(pair: tuple[SimulatedAction, SimulatedAction]) -> None:
            if pair not in selected_set and len(selected_candidates) < candidate_limit:
                selected_candidates.append(pair)
                selected_set.add(pair)

        # Tactical quotas ensuring comprehensive matrix coverage across all strategic dimensions
        quota_focus0 = max(1, candidate_limit // 5)
        quota_focus1 = max(1, candidate_limit // 5)
        quota_split = max(1, candidate_limit // 5)
        quota_prot0 = 1 if candidate_limit >= 6 else 0
        quota_prot1 = 1 if candidate_limit >= 6 else 0
        quota_support = max(1, candidate_limit // 6) if candidate_limit >= 6 else 0
        quota_switch = 1 if candidate_limit >= 7 else 0

        for pair in bucket_focus_opp0[:quota_focus0]:
            add_candidate(pair)
        for pair in bucket_focus_opp1[:quota_focus1]:
            add_candidate(pair)
        for pair in bucket_split[:quota_split]:
            add_candidate(pair)
        for pair in bucket_protect0[:quota_prot0]:
            add_candidate(pair)
        for pair in bucket_protect1[:quota_prot1]:
            add_candidate(pair)
        for pair in bucket_support[:quota_support]:
            add_candidate(pair)
        for pair in bucket_switch[:quota_switch]:
            add_candidate(pair)

        # Backfill remaining quota slots with highest-scoring remaining actions overall
        for _, pair in joint_candidates:
            if len(selected_candidates) >= candidate_limit:
                break
            add_candidate(pair)

        return selected_candidates

    def _get_slot_candidates(
        self,
        state: SimulatedBattleState,
        active_pokemon_list: Sequence[Optional[SimulatedPokemon]],
        bench_pokemon_list: list[SimulatedPokemon],
        opponent_pokemon_list: Sequence[Optional[SimulatedPokemon]],
        slot: int
    ) -> list[tuple[float, SimulatedAction]]:
        active_pokemon = active_pokemon_list[slot] if slot < len(active_pokemon_list) else None

        # Check if the battle is currently in a force_switch phase
        is_force_switch_phase = any(state.force_switch)
        if is_force_switch_phase:
            if slot < len(state.force_switch) and not state.force_switch[slot]:
                # This slot is NOT forced to switch (alive partner), so it MUST pass
                return [(100.0, SimulatedAction("pass", slot=slot))]

            # This slot IS forced to switch: must choose an alive bench Pokemon
            available_switches: list[tuple[float, SimulatedAction]] = []
            for bench_index, bench_pokemon in enumerate(bench_pokemon_list):
                if bench_pokemon and not bench_pokemon.fainted:
                    switch_score = 100.0
                    for opponent_pokemon in opponent_pokemon_list:
                        if opponent_pokemon and not opponent_pokemon.fainted:
                            for opponent_type in opponent_pokemon.types:
                                if bench_pokemon.damage_multiplier(opponent_type) < 1.0:
                                    switch_score += 20.0
                    available_switches.append((switch_score, SimulatedAction("switch", slot=slot, switch_idx=bench_index)))

            if available_switches:
                available_switches.sort(key=lambda item: item[0], reverse=True)
                # If both slots are forced to switch, also permit pass so if only 1 bench mon is left, one can pass
                if all(state.force_switch):
                    available_switches.append((0.0, SimulatedAction("pass", slot=slot)))
                return available_switches
            return [(0.0, SimulatedAction("pass", slot=slot))]

        if not active_pokemon or active_pokemon.fainted:
            # Regular turn and slot is fainted: must pass
            return [(0.0, SimulatedAction("pass", slot=slot))]

        slot_candidates: list[tuple[float, SimulatedAction]] = []

        is_doubles = (state.battle_format == "doubles")
        partner = active_pokemon_list[1 - slot] if (is_doubles and len(active_pokemon_list) > 1) else None

        # Evaluate all moves in active Pokemon's moveset without dropping any move
        for move_id, simulated_move in active_pokemon.moves.items():
            # Check volatile restrictions on attacker
            if active_pokemon.taunt_turns > 0 and simulated_move.category == "status":
                continue  # Cannot select status moves while Taunted
            if active_pokemon.disable_turns > 0 and move_id == active_pokemon.disabled_move:
                continue  # Cannot select disabled move
            if active_pokemon.encore_turns > 0 and active_pokemon.encore_move and move_id != active_pokemon.encore_move:
                continue  # Locked into encored move
            if active_pokemon.torment and active_pokemon.last_move_used and move_id == active_pokemon.last_move_used:
                continue  # Cannot select same move consecutively under Torment

            # 1. Attacking moves (physical / special)
            if simulated_move.category != "status":
                eff_power = get_effective_base_power(simulated_move)
                stab = 1.5 if simulated_move.type in active_pokemon.types else 1.0
                is_spread = simulated_move.is_spread or simulated_move.target in ("allAdjacentFoes", "allAdjacent")

                if is_spread:
                    # Spread move hits both opponents simultaneously
                    total_dmg_frac = 0.0
                    alive_opp_count = 0
                    for opp in opponent_pokemon_list:
                        if opp and not opp.fainted:
                            t_mult = opp.damage_multiplier(simulated_move.type)
                            if t_mult > 0.0:
                                est_dmg = eff_power * stab * t_mult * 0.75  # 0.75 spread damage penalty in Doubles
                                total_dmg_frac += min(1.5, est_dmg / max(opp.current_hp, 1.0))
                                alive_opp_count += 1
                    if alive_opp_count > 0:
                        avg_frac = total_dmg_frac / alive_opp_count
                        priority_score = 45.0 + 35.0 * min(1.0, avg_frac) + 10.0  # +10 spread bonus
                        if avg_frac >= 0.90:
                            priority_score += 20.0  # Threatens KO
                        if simulated_move.priority > 0:
                            priority_score += 10.0
                        slot_candidates.append(
                            (priority_score, SimulatedAction("move", slot=slot, move_id=move_id, target_idx=0, mega=False))
                        )
                        if active_pokemon.can_mega_evolve:
                            slot_candidates.append(
                                (priority_score + 10.0, SimulatedAction("move", slot=slot, move_id=move_id, target_idx=0, mega=True))
                            )
                else:
                    # Single-target attacks: evaluate against both Opponent 0 and Opponent 1
                    for target_index in (0, 1):
                        if target_index < len(opponent_pokemon_list):
                            opponent_pokemon = opponent_pokemon_list[target_index]
                            if opponent_pokemon and not opponent_pokemon.fainted:
                                type_mult = opponent_pokemon.damage_multiplier(simulated_move.type)
                                if type_mult == 0.0:
                                    continue  # Immune, skip this target
                                est_dmg = eff_power * stab * type_mult
                                dmg_frac = min(1.5, est_dmg / max(opponent_pokemon.current_hp, 1.0))
                                priority_score = 45.0 + 35.0 * min(1.0, dmg_frac)
                                if dmg_frac >= 0.95:
                                    priority_score += 25.0  # Decisive KO threat
                                if simulated_move.priority > 0:
                                    priority_score += 10.0
                                slot_candidates.append(
                                    (priority_score, SimulatedAction("move", slot=slot, move_id=move_id, target_idx=target_index, mega=False))
                                )
                                if active_pokemon.can_mega_evolve:
                                    slot_candidates.append(
                                        (priority_score + 10.0, SimulatedAction("move", slot=slot, move_id=move_id, target_idx=target_index, mega=True))
                                    )

                # Ally proc / support check with attacks (Pollen Puff healing, Weakness Policy proc)
                if is_doubles and partner and not partner.fainted:
                    if move_id == "pollenpuff":
                        heal_priority = 80.0 if partner.current_hp_fraction < 0.60 else 55.0
                        slot_candidates.append(
                            (heal_priority, SimulatedAction("move", slot=slot, move_id=move_id, target_idx=2, mega=False))
                        )
                    elif (partner.item == "weaknesspolicy" and partner.damage_multiplier(simulated_move.type) > 1.0):
                        slot_candidates.append(
                            (75.0, SimulatedAction("move", slot=slot, move_id=move_id, target_idx=2, mega=False))
                        )

            # 2. Status / Support / Disruption / Setup moves
            else:
                mid = simulated_move.id
                # Defensive Protection & Fake Out
                if mid in constants.PROTECT_MOVES:
                    if active_pokemon.protect_counter == 0:
                        priority_score = 75.0
                    elif active_pokemon.protect_counter == 1:
                        priority_score = 20.0
                    else:
                        priority_score = 5.0
                    slot_candidates.append(
                        (priority_score, SimulatedAction("move", slot=slot, move_id=mid, target_idx=0, mega=False))
                    )
                elif mid == "fakeout":
                    if active_pokemon.first_turn:
                        for target_index in (0, 1):
                            if target_index < len(opponent_pokemon_list):
                                opp = opponent_pokemon_list[target_index]
                                if opp and not opp.fainted and opp.damage_multiplier("normal") > 0.0:
                                    slot_candidates.append(
                                        (80.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=target_index, mega=False))
                                    )
                # High-Impact Speed & Field Control
                elif mid == "trickroom":
                    slot_candidates.append(
                        (85.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=0, mega=False))
                    )
                elif mid == "tailwind":
                    state_conditions = state.side1_conditions if (active_pokemon and active_pokemon.side == 1) \
                        else state.side2_conditions
                    if state_conditions.get("tailwind", 0) == 0:
                        slot_candidates.append(
                            (85.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=0, mega=False))
                        )
                # Sleep Infliction
                elif mid in ("spore", "sleeppowder", "hypnosis", "yawn"):
                    for target_index in (0, 1):
                        if target_index < len(opponent_pokemon_list):
                            opp = opponent_pokemon_list[target_index]
                            if opp and not opp.fainted and not opp.status:
                                # Grass types and Safety Goggles immune to Spore/Sleep Powder
                                if mid in ("spore", "sleeppowder") and ("grass" in opp.types or opp.item == "safetygoggles"):
                                    continue
                                score = 85.0 if mid == "spore" else 75.0
                                slot_candidates.append(
                                    (score, SimulatedAction("move", slot=slot, move_id=mid, target_idx=target_index, mega=False))
                                )
                # Disruption: Taunt, Encore, Disable, Torment
                elif mid == "encore":
                    for target_index in (0, 1):
                        if target_index < len(opponent_pokemon_list):
                            opp = opponent_pokemon_list[target_index]
                            if opp and not opp.fainted and opp.last_move_used and opp.encore_turns == 0:
                                slot_candidates.append(
                                    (85.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=target_index, mega=False))
                                )
                elif mid == "taunt":
                    for target_index in (0, 1):
                        if target_index < len(opponent_pokemon_list):
                            opp = opponent_pokemon_list[target_index]
                            if opp and not opp.fainted and opp.taunt_turns == 0:
                                slot_candidates.append(
                                    (80.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=target_index, mega=False))
                                )
                elif mid in ("disable", "torment"):
                    for target_index in (0, 1):
                        if target_index < len(opponent_pokemon_list):
                            opp = opponent_pokemon_list[target_index]
                            if opp and not opp.fainted and opp.last_move_used and opp.disable_turns == 0:
                                slot_candidates.append(
                                    (70.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=target_index, mega=False))
                                )
                # Redirection, Guards & Positioning
                elif mid in ("followme", "ragepowder"):
                    slot_candidates.append(
                        (80.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=0, mega=False))
                    )
                elif mid in ("wideguard", "quickguard", "craftyshield", "allyswitch"):
                    slot_candidates.append(
                        (75.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=0, mega=False))
                    )
                # Stat Setup
                elif mid in (
                    "swordsdance", "nastyplot", "calmmind", "dragondance", "bulkup", "quiverdance",
                    "irondefense", "shellsmash", "bellydrum", "shiftgear", "tidyup", "geomancy",
                    "victorydance", "coil", "growth", "workup", "acidarmor", "amnesia"
                ):
                    slot_candidates.append(
                        (75.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=0, mega=False))
                    )
                # Substitute
                elif mid == "substitute":
                    sub_score = 70.0 if active_pokemon.current_hp_fraction > 0.30 else 20.0
                    slot_candidates.append(
                        (sub_score, SimulatedAction("move", slot=slot, move_id=mid, target_idx=0, mega=False))
                    )
                # Status Infliction (Burn, Paralysis, Toxic)
                elif mid in ("willowisp", "thunderwave", "glare", "toxic", "stunspore"):
                    for target_index in (0, 1):
                        if target_index < len(opponent_pokemon_list):
                            opp = opponent_pokemon_list[target_index]
                            if opp and not opp.fainted and not opp.status:
                                if mid == "willowisp" and "fire" in opp.types:
                                    continue
                                if mid in ("thunderwave", "glare") and ("electric" in opp.types or "ground" in opp.types):
                                    continue
                                if mid == "toxic" and ("poison" in opp.types or "steel" in opp.types):
                                    continue
                                slot_candidates.append(
                                    (70.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=target_index, mega=False))
                                )
                # Ally Buffs
                elif is_ally_buff_move(simulated_move):
                    if is_doubles and partner and not partner.fainted:
                        buff_score = 80.0 if mid in ("coaching", "decorate") else 70.0
                        slot_candidates.append(
                            (buff_score, SimulatedAction("move", slot=slot, move_id=mid, target_idx=2, mega=False))
                        )
                # Recovery & Healing
                elif mid in (
                    "recover", "roost", "slackoff", "softboiled", "milkdrink",
                    "synthesis", "moonlight", "morningsun", "healorder", "wish", "lifedew"
                ):
                    heal_score = 75.0 if active_pokemon.current_hp_fraction < 0.40 else (50.0 if active_pokemon.current_hp_fraction < 0.70 else 20.0)
                    slot_candidates.append(
                        (heal_score, SimulatedAction("move", slot=slot, move_id=mid, target_idx=0, mega=False))
                    )
                elif mid == "strengthsap":
                    for target_index in (0, 1):
                        if target_index < len(opponent_pokemon_list):
                            opp = opponent_pokemon_list[target_index]
                            if opp and not opp.fainted:
                                sap_score = 75.0 if active_pokemon.current_hp_fraction < 0.50 else 50.0
                                slot_candidates.append(
                                    (sap_score, SimulatedAction("move", slot=slot, move_id=mid, target_idx=target_index, mega=False))
                                )
                # Pivots & Debuffs
                elif mid in ("partingshot", "chillyreception"):
                    alive_bench = sum(1 for b in bench_pokemon_list if b and not b.fainted)
                    if alive_bench > 0:
                        if mid == "partingshot":
                            for target_index in (0, 1):
                                if target_index < len(opponent_pokemon_list):
                                    opp = opponent_pokemon_list[target_index]
                                    if opp and not opp.fainted:
                                        slot_candidates.append(
                                            (70.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=target_index, mega=False))
                                        )
                        else:
                            slot_candidates.append(
                                (70.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=0, mega=False))
                            )
                elif is_self_or_field_move(simulated_move):
                    slot_candidates.append(
                        (60.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=0, mega=False))
                    )
                else:
                    # General status move targeting opponent
                    for target_index in (0, 1):
                        if target_index < len(opponent_pokemon_list):
                            opp = opponent_pokemon_list[target_index]
                            if opp and not opp.fainted:
                                slot_candidates.append(
                                    (55.0, SimulatedAction("move", slot=slot, move_id=mid, target_idx=target_index, mega=False))
                                )

        # 3. Switches to bench
        for bench_index, bench_pokemon in enumerate(bench_pokemon_list):
            if bench_pokemon and not bench_pokemon.fainted:
                switch_score = 65.0
                if bench_pokemon.ability == "intimidate":
                    switch_score += 15.0
                for opponent_pokemon in opponent_pokemon_list:
                    if opponent_pokemon and not opponent_pokemon.fainted:
                        for opponent_type in opponent_pokemon.types:
                            if bench_pokemon.damage_multiplier(opponent_type) < 1.0:
                                switch_score += 5.0
                slot_candidates.append(
                    (switch_score, SimulatedAction("switch", slot=slot, switch_idx=bench_index))
                )

        if not slot_candidates:
            return [(50.0, SimulatedAction("move", slot=slot, move_id="struggle", target_idx=0))]

        # Sort all slot candidates cleanly by priority score
        # Critically: Return ALL candidates without truncating, so NO move or target is ever skipped!
        slot_candidates.sort(key=lambda item: item[0], reverse=True)
        return slot_candidates

    def search(
        self,
        state: SimulatedBattleState,
        policy_prior: Optional[np.ndarray] = None,
    ) -> SearchResult:
        """
        Executes the multi-turn lookahead search to depth `self.depth`.
        Returns the optimal decision and game-theoretic value.
        """
        self.nodes_evaluated = 0

        # Fast path for force_switch phases: evaluate candidate replacement switches directly
        # Basically, if any Pokemon fainted
        if any(state.force_switch):
            candidates = self.generate_candidate_actions(state, side=1, policy_prior=policy_prior)
            if not candidates:
                best_action: tuple[SimulatedAction, SimulatedAction] = (SimulatedAction("pass", slot=0), SimulatedAction("pass", slot=1))
                return SearchResult(
                    best_action=best_action,
                    value=0.0,
                    action_distribution={best_action: 1.0},
                    predicted_opponent_actions={},
                    depth=1,
                    nodes_evaluated=1,
                    method="force_switch",
                )

            scored_candidates: list[tuple[float, tuple[SimulatedAction, SimulatedAction]]] = []
            dummy_opp: tuple[SimulatedAction, SimulatedAction] = (SimulatedAction("pass", slot=0), SimulatedAction("pass", slot=1))
            for action in candidates:
                next_battle_state = step_simulation(state, action, dummy_opp)
                state_value = self.evaluator.evaluate(next_battle_state)
                scored_candidates.append((state_value, action))

            scored_candidates.sort(key=lambda item: item[0], reverse=True)
            best_value, best_action = scored_candidates[0]
            distribution: dict[tuple[SimulatedAction, SimulatedAction], float] = {action: (1.0 if action == best_action else 0.0) for _, action in scored_candidates}
            return SearchResult(
                best_action=best_action,
                value=best_value,
                action_distribution=distribution,
                predicted_opponent_actions={},
                depth=1,
                nodes_evaluated=len(candidates),
                method="force_switch",
            )

        game_value, best_nash_action, distribution_player1, distribution_player2 = self._search_nash(
            state, depth=self.depth, policy_prior=policy_prior
        )

        resolved_action: tuple[SimulatedAction, SimulatedAction] = (
            best_nash_action if best_nash_action is not None
            else (SimulatedAction("pass", slot=0), SimulatedAction("pass", slot=1))
        )

        return SearchResult(
            best_action=resolved_action,
            value=game_value,
            action_distribution=distribution_player1,
            predicted_opponent_actions=distribution_player2,
            depth=self.depth,
            nodes_evaluated=self.nodes_evaluated,
            method="nash",
        )

    def _search_nash(
        self,
        state: SimulatedBattleState,
        depth: int,
        policy_prior: Optional[np.ndarray] = None,
    ) -> tuple[
        float,
        Optional[tuple[SimulatedAction, SimulatedAction]],
        dict[tuple[SimulatedAction, SimulatedAction], float],
        dict[tuple[SimulatedAction, SimulatedAction], float],
    ]:
        """Recursive Simultaneous Minimax with Nash Equilibrium matrix solving."""
        if depth <= 0 or state.is_finished():
            self.nodes_evaluated += 1
            leaf_value = self.evaluator.evaluate(state)
            return leaf_value, None, {}, {}

        # Use neural policy prior at root to guide action exploration; optimize branching factor at depth
        current_candidate_limit = self.max_candidates if depth == self.depth else min(6, self.max_candidates)
        candidates_player1 = self.generate_candidate_actions(
            state, side=1, max_candidates=current_candidate_limit, policy_prior=policy_prior if depth == self.depth else None
        )
        candidates_player2 = self.generate_candidate_actions(
            state, side=2, max_candidates=current_candidate_limit
        )

        if not candidates_player1:
            self.nodes_evaluated += 1
            fallback: tuple[SimulatedAction, SimulatedAction] = (SimulatedAction("pass", slot=0), SimulatedAction("pass", slot=1))
            return self.evaluator.evaluate(state), fallback, {fallback: 1.0}, {}

        if not candidates_player2:
            self.nodes_evaluated += len(candidates_player1)
            opp_pass = (SimulatedAction("pass", slot=0), SimulatedAction("pass", slot=1))
            
            best_value = -float("inf")
            best_action = candidates_player1[0]
            
            for action in candidates_player1:
                next_state = step_simulation(state, action, opp_pass)
                value = self.evaluator.evaluate(next_state)
                if value > best_value:
                    best_value = value
                    best_action = action
                    
            return best_value, best_action, {best_action: 1.0}, {opp_pass: 1.0}

        num_player1_actions = len(candidates_player1)
        num_player2_actions = len(candidates_player2)
        payoff_matrix = np.zeros((num_player1_actions, num_player2_actions), dtype=np.float64)

        for index_player1, action_player1 in enumerate(candidates_player1):
            for index_player2, action_player2 in enumerate(candidates_player2):
                next_battle_state = step_simulation(state, action_player1, action_player2)
                if depth - 1 == 0 or next_battle_state.is_finished():
                    self.nodes_evaluated += 1
                    payoff_matrix[index_player1, index_player2] = self.evaluator.evaluate(next_battle_state)
                else:
                    # Recursive lookahead for deeper plies (depth 2 or 3)
                    sub_value, _, _, _ = self._search_nash(next_battle_state, depth - 1)
                    payoff_matrix[index_player1, index_player2] = sub_value

        # Solve Nash equilibrium of payoff_matrix
        strategy_player1, strategy_player2, game_value = solve_zero_sum_matrix_game(payoff_matrix)

        best_player1_action_index = int(np.argmax(strategy_player1))
        best_action = candidates_player1[best_player1_action_index]

        distribution_player1: dict[tuple[SimulatedAction, SimulatedAction], float] = {
            candidates_player1[index]: float(strategy_player1[index])
            for index in range(num_player1_actions)
        }
        distribution_player2: dict[tuple[SimulatedAction, SimulatedAction], float] = {
            candidates_player2[index]: float(strategy_player2[index])
            for index in range(num_player2_actions)
        }

        return game_value, best_action, distribution_player1, distribution_player2

    def select_order(
        self,
        battle: Any,
        player: Optional[Player] = None,
        policy_prior: Optional[np.ndarray] = None,
    ) -> DoubleBattleOrder:
        """
        Takes a poke-env DoubleBattle and returns the corresponding
        DoubleBattleOrder ready to send to Showdown!
        """
        simulated_state = SimulatedBattleState.from_battle(battle, fill_unrevealed=True)
        search_result = self.search(simulated_state, policy_prior=policy_prior)
        best_action = search_result.best_action

        return self._to_doubles_order(best_action, battle, player)

    def _to_doubles_order(
        self,
        joint_simulated_action: tuple[SimulatedAction, SimulatedAction],
        battle: Any,
        player: Optional[Player] = None,
    ) -> DoubleBattleOrder:
        if not joint_simulated_action:
            first_pass: Any = PassBattleOrder()
            second_pass: Any = PassBattleOrder()
            return DoubleBattleOrder(first_pass, second_pass)

        slot_orders: list[Any] = []
        bench_pokemon_list: list[Any] = [
            pokemon for pokemon in battle.team.values()
            if pokemon and not pokemon.fainted and pokemon not in battle.active_pokemon and getattr(pokemon, "selected_in_teampreview", True)
        ]

        # 1. Handle force_switch requests from Showdown
        force_switch_flags = getattr(battle, "force_switch", [False, False])
        if any(force_switch_flags):
            available_switches_by_slot: list[list[Any]] = getattr(battle, "available_switches", [[], []])
            chosen_switch_targets: set[Any] = set()

            for slot_index, slot_action in enumerate(joint_simulated_action):
                if slot_index >= len(force_switch_flags) or not force_switch_flags[slot_index]:
                    # Not forced to switch, partner must pass
                    slot_orders.append(PassBattleOrder())
                    continue

                available_for_slot: list[Any] = [
                    p for p in (available_switches_by_slot[slot_index] if slot_index < len(available_switches_by_slot) else [])
                    if p not in chosen_switch_targets
                ]
                chosen_pokemon: Optional[Any] = None

                if slot_action and slot_action.type == "switch":
                    if slot_action.switch_idx < len(bench_pokemon_list):
                        candidate_mon = bench_pokemon_list[slot_action.switch_idx]
                        if candidate_mon in available_for_slot:
                            chosen_pokemon = candidate_mon

                if not chosen_pokemon and available_for_slot:
                    chosen_pokemon = available_for_slot[0]

                if chosen_pokemon:
                    chosen_switch_targets.add(chosen_pokemon)
                    slot_orders.append(player.create_order(chosen_pokemon) if player else chosen_pokemon)
                else:
                    slot_orders.append(PassBattleOrder())

            first_force_order: Any = slot_orders[0] if len(slot_orders) > 0 else PassBattleOrder()
            second_force_order: Any = slot_orders[1] if len(slot_orders) > 1 else PassBattleOrder()
            return DoubleBattleOrder(first_force_order, second_force_order)

        # 2. Handle normal turns
        for slot_index, slot_action in enumerate(joint_simulated_action):
            active_pokemon = battle.active_pokemon[slot_index] if slot_index < len(battle.active_pokemon) else None

            # Fainted slot must pass
            if not active_pokemon or active_pokemon.fainted:
                slot_orders.append(PassBattleOrder())
                continue

            available_switches: list[Any] = battle.available_switches[slot_index] if hasattr(battle, "available_switches") else []
            available_moves: list[Any] = battle.available_moves[slot_index] if hasattr(battle, "available_moves") else []

            # Switch action
            if slot_action and slot_action.type == "switch":
                chosen_pokemon = None
                if slot_action.switch_idx < len(bench_pokemon_list) and bench_pokemon_list[slot_action.switch_idx] in available_switches:
                    chosen_pokemon = bench_pokemon_list[slot_action.switch_idx]
                elif available_switches:
                    chosen_pokemon = available_switches[0]

                if chosen_pokemon:
                    slot_orders.append(player.create_order(chosen_pokemon) if player else chosen_pokemon)
                    continue

            # Move action (or switch fallback)
            chosen_move: Optional[Any] = None
            if slot_action and slot_action.type == "move":
                for pokemon_move in active_pokemon.moves.values():
                    if clean_id(pokemon_move.id) == slot_action.move_id and pokemon_move in available_moves:
                        chosen_move = pokemon_move
                        break
            if not chosen_move and available_moves:
                chosen_move = available_moves[0]

            if chosen_move:
                move_target_obj = getattr(chosen_move, "target", None)
                target_key = clean_id(getattr(move_target_obj, "name", str(move_target_obj)))

                # In Showdown Doubles:
                # Ally targets are negative: Slot 0's partner is slot 1 (-2), Slot 1's partner is slot 0 (-1)
                ally_target_position = -2 if slot_index == 0 else -1

                if slot_action and slot_action.target_idx == 2:
                    # Explicit ally targeting (e.g. Pollen Puff heal, Weakness Policy proc, Helping Hand)
                    target_position = ally_target_position
                elif target_key == "adjacentally":
                    # Moves that can ONLY target the ally (Coaching, Helping Hand, Floral Healing, Aromatic Mist)
                    target_position = ally_target_position
                elif target_key in {"normal", "any", "adjacentfoe"}:
                    # Opponent-targeting moves: 1 for opponent slot 0, 2 for opponent slot 1
                    target_position = slot_action.target_idx + 1 if (slot_action and slot_action.target_idx < 2) else 1
                    opp1: Optional[Any] = battle.opponent_active_pokemon[0] if len(battle.opponent_active_pokemon) > 0 else None
                    opp2: Optional[Any] = battle.opponent_active_pokemon[1] if len(battle.opponent_active_pokemon) > 1 else None
                    opp1_alive = opp1 is not None and not getattr(opp1, "fainted", False)
                    opp2_alive = opp2 is not None and not getattr(opp2, "fainted", False)
                    if target_position == 1 and not opp1_alive and opp2_alive:
                        target_position = 2
                    elif target_position == 2 and not opp2_alive and opp1_alive:
                        target_position = 1
                elif target_key == "adjacentallyorself":
                    # Move can target ally or self (e.g. Acupressure)
                    partner: Optional[Any] = battle.active_pokemon[1 - slot_index] if len(battle.active_pokemon) > 1 else None
                    if partner and not getattr(partner, "fainted", False):
                        target_position = ally_target_position
                    else:
                        target_position = -1 if slot_index == 0 else -2
                else:
                    # Spread moves, self-moves, whole field moves (Protect, Swords Dance, Heat Wave, Tailwind)
                    target_position = 0

                is_mega = slot_action.mega if slot_action else False
                if player:
                    slot_orders.append(player.create_order(chosen_move, move_target=target_position, mega=is_mega))
                else:
                    slot_orders.append(chosen_move)
            elif available_switches:
                slot_orders.append(player.create_order(available_switches[0]) if player else available_switches[0])
            else:
                slot_orders.append(PassBattleOrder())

        first_order: Any = slot_orders[0] if len(slot_orders) > 0 else PassBattleOrder()
        second_order: Any = slot_orders[1] if len(slot_orders) > 1 else PassBattleOrder()
        return DoubleBattleOrder(first_order, second_order)

    def to_env_action(
        self,
        joint_action: tuple[SimulatedAction, SimulatedAction],
        battle: Optional[Any] = None,
    ) -> int:
        """
        Maps a joint action (action_slot0, action_slot1) directly to a Gymnasium VGCEnv discrete action index (0-675).
        """
        if not joint_action or len(joint_action) < 2:
            return 0
        action_slot0_integer = min(25, max(0, self._slot_action_to_int(joint_action[0], slot_index=0, battle=battle)))
        action_slot1_integer = min(25, max(0, self._slot_action_to_int(joint_action[1], slot_index=1, battle=battle)))
        return int(min(675, max(0, action_slot0_integer * 26 + action_slot1_integer)))

    def _slot_action_to_int(
        self,
        slot_action: SimulatedAction,
        slot_index: int,
        battle: Optional[Any],
    ) -> int:
        if not slot_action or slot_action.type == "pass" or battle is None or not hasattr(battle, "team"):
            return 0

        bench: list[Any] = [
            p for p in battle.team.values()
            if p and not p.fainted and p not in battle.active_pokemon and getattr(p, "selected_in_teampreview", True)
        ]
        available_switches: list[Any] = battle.available_switches[slot_index] if hasattr(battle, "available_switches") else []

        # Handle force_switch scenario
        force_switch_flags = getattr(battle, "force_switch", [])
        if any(force_switch_flags):
            if slot_index < len(force_switch_flags) and not force_switch_flags[slot_index]:
                return 0  # Partner must pass
            if not bench or not available_switches:
                return 0  # Must pass if no bench Pokémon
            if slot_action.type == "switch":
                switch_index = min(1, min(len(bench) - 1, max(0, slot_action.switch_idx)))
                return min(25, max(24, 24 + switch_index))
            return 24  # Fallback to switch action

        active_pokemon: Optional[Any] = battle.active_pokemon[slot_index] if slot_index < len(battle.active_pokemon) else None
        if not active_pokemon or active_pokemon.fainted:
            return 0

        # Normal turn switch
        if slot_action.type == "switch":
            if not bench or not available_switches:
                return 0  # Cannot switch, fall back to move 0
            switch_index = min(1, min(len(bench) - 1, max(0, slot_action.switch_idx)))
            return min(25, max(24, 24 + switch_index))

        # Move action
        move_index = 0
        moves_list: list[Any] = list(active_pokemon.moves.values())
        for current_index, pokemon_move in enumerate(moves_list):
            if clean_id(pokemon_move.id) == slot_action.move_id:
                move_index = current_index
                break

        target_index = min(2, max(0, slot_action.target_idx))
        slot_action_code = move_index * 3 + target_index
        if slot_action.mega:
            slot_action_code += 12
        return min(25, max(0, slot_action_code))

class LookaheadPlayer(Player):
    """
    Autonomous poke-env battle agent for Doubles / VGC powered by 2-3 turn simultaneous
    lookahead search with Counterfactual Regret Matching (CFR) Nash equilibrium.
    """

    searcher: LookaheadSearch

    def __init__(
        self,
        depth: int = 2,
        max_candidates: int = 10,
        evaluator: Optional[BattleStateEvaluator] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("accept_open_team_sheet", True)
        if "account_configuration" not in kwargs:
            kwargs["account_configuration"] = AccountConfiguration(f"Lookahead_{uuid.uuid4().hex[:6]}", None)
        kwargs.pop("method", None)
        super().__init__(*args, **kwargs)
        self.searcher = LookaheadSearch(
            evaluator=evaluator or CompetitiveHeuristicEvaluator(),
            depth=depth,
            max_candidates=max_candidates,
        )

    def choose_move(self, battle: Any) -> BattleOrder:
        if self._trying_again.is_set():
            self._trying_again.clear()
            return self.choose_default_move()
        try:
            return self.searcher.select_order(battle, player=self)
        except Exception:
            return self.choose_default_move()
