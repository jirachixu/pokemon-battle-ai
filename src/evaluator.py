"""
Competitive battle state evaluator with optional Neural Network Value head integration.
Evaluates board states from Player 1's perspective.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Any
import torch

from src.simulation import SimulatedBattleState, SimulatedPokemon, SimulatedMove, clean_id
from src.state_encoder import StateEncoder

class BattleStateEvaluator(ABC):
    """Abstract base class for all battle state evaluators."""

    @abstractmethod
    def evaluate(self, state: SimulatedBattleState) -> float:
        """Computes evaluation score for state. Higher is better for Side 1."""
        raise NotImplementedError

class CompetitiveHeuristicEvaluator(BattleStateEvaluator):
    """
    Competitive evaluation function for Doubles / VGC Pokemon battles.
    Considers material balance, HP differentials, speed control, stat stages, status conditions,
    field control, and immediate offensive KO pressure.
    """

    def __init__(
        self,
        weight_win_loss: float = 100000.0,
        weight_alive_differential: float = 2500.0,
        weight_hp_differential: float = 1500.0,
        weight_speed_control: float = 350.0,
        weight_stat_boosts: float = 250.0,
        weight_status_penalty: float = 350.0,
        weight_threat_pressure: float = 500.0,
        weight_flinch: float = 1000.0,
    ):
        self.weight_win_loss = weight_win_loss
        self.weight_alive_differential = weight_alive_differential
        self.weight_hp_differential = weight_hp_differential
        self.weight_speed_control = weight_speed_control
        self.weight_stat_boosts = weight_stat_boosts
        self.weight_status_penalty = weight_status_penalty
        self.weight_threat_pressure = weight_threat_pressure
        self.weight_flinch = weight_flinch
    def evaluate(self, state: SimulatedBattleState) -> float:
        """Computes evaluation score for state. Higher is better for Side 1."""
        winner_side = state.winner()
        if winner_side == 1:
            return self.weight_win_loss - (state.turn * 10.0)  # Reward faster wins
        elif winner_side == 2:
            return -self.weight_win_loss + (state.turn * 10.0)  # Reward slower losses
        elif winner_side == 0:
            return 0.0

        evaluation_score = 0.0

        side1_alive_count = state.alive_side1_count()
        side2_alive_count = state.alive_side2_count()

        # 1. Material (Remaining Pokemon) Advantage
        evaluation_score += self.weight_alive_differential * (side1_alive_count - side2_alive_count)

        # 2. HP Differential
        side1_total_hp = sum(pokemon.current_hp_fraction for pokemon in state.all_side1_pokemon() if not pokemon.fainted)
        side2_total_hp = sum(pokemon.current_hp_fraction for pokemon in state.all_side2_pokemon() if not pokemon.fainted)
        evaluation_score += self.weight_hp_differential * (side1_total_hp - side2_total_hp)

        # Active HP priority: active Pokemon on field are more immediately impactful than bench
        side1_active_hp = sum(pokemon.current_hp_fraction for pokemon in state.side1_active if pokemon and not pokemon.fainted)
        side2_active_hp = sum(pokemon.current_hp_fraction for pokemon in state.side2_active if pokemon and not pokemon.fainted)
        evaluation_score += (self.weight_hp_differential * 0.5) * (side1_active_hp - side2_active_hp)

        # 3. Speed Control & Field Advantage
        is_trick_room_active = "trickroom" in state.fields
        side1_tailwind_turns = state.side1_conditions.get("tailwind", 0)
        side2_tailwind_turns = state.side2_conditions.get("tailwind", 0)
        evaluation_score += self.weight_speed_control * (side1_tailwind_turns - side2_tailwind_turns)

        # Active speed order comparison
        side1_living_active = [pokemon for pokemon in state.side1_active if pokemon and not pokemon.fainted]
        side2_living_active = [pokemon for pokemon in state.side2_active if pokemon and not pokemon.fainted]

        if side1_living_active and side2_living_active:
            side1_effective_speeds = [
                pokemon.calculate_effective_speed(state.weather, state.fields, state.side1_conditions)
                for pokemon in side1_living_active
            ]
            side2_effective_speeds = [
                pokemon.calculate_effective_speed(state.weather, state.fields, state.side2_conditions)
                for pokemon in side2_living_active
            ]

            max_side1_speed = max(side1_effective_speeds)
            max_side2_speed = max(side2_effective_speeds)

            if is_trick_room_active:
                # Slower speeds move first under Trick Room
                min_side1_speed = min(side1_effective_speeds)
                min_side2_speed = min(side2_effective_speeds)
                if min_side1_speed < min_side2_speed:
                    evaluation_score += self.weight_speed_control * 0.8
                else:
                    evaluation_score -= self.weight_speed_control * 0.8
            else:
                if max_side1_speed > max_side2_speed:
                    evaluation_score += self.weight_speed_control * 0.8
                else:
                    evaluation_score -= self.weight_speed_control * 0.8

        # 4. Stat Boosts Advantage
        for pokemon in state.all_side1_pokemon():
            if not pokemon.fainted:
                evaluation_score += self._evaluate_pokemon_boosts(pokemon)
        for pokemon in state.all_side2_pokemon():
            if not pokemon.fainted:
                evaluation_score -= self._evaluate_pokemon_boosts(pokemon)

        # 5. Status Conditions Penalty
        for pokemon in state.all_side1_pokemon():
            if not pokemon.fainted:
                evaluation_score -= self._evaluate_status_penalty(pokemon)
        for pokemon in state.all_side2_pokemon():
            if not pokemon.fainted:
                evaluation_score += self._evaluate_status_penalty(pokemon)
        # 5.1. Flinch Penalty
        for pokemon in state.all_side1_pokemon():
            if not pokemon.fainted:
                evaluation_score -= self._evaluate_flinch_penalty(pokemon)
        for pokemon in state.all_side2_pokemon():
            if not pokemon.fainted:
                evaluation_score += self._evaluate_flinch_penalty(pokemon)

        # 6. Immediate Lethal KO Threat Evaluation
        evaluation_score += self._evaluate_threat_matrix(side1_living_active, side2_living_active, state)

        return evaluation_score

    def _evaluate_pokemon_boosts(self, pokemon: SimulatedPokemon) -> float:
        boost_score = 0.0
        attack_boost = pokemon.boosts.get("atk", 0)
        special_attack_boost = pokemon.boosts.get("spa", 0)
        speed_boost = pokemon.boosts.get("spe", 0)
        defense_boost = pokemon.boosts.get("def", 0)
        special_defense_boost = pokemon.boosts.get("spd", 0)
        
        is_special_attacker = self._is_special_attacker(pokemon)
        is_physical_attacker = self._is_physical_attacker(pokemon)

        # Positive offensive stages
        if is_special_attacker:
            boost_score += max(0, special_attack_boost) * self.weight_stat_boosts
        if is_physical_attacker:
            boost_score += max(0, attack_boost) * self.weight_stat_boosts
        boost_score += max(0, speed_boost) * (self.weight_stat_boosts * 0.8)
        boost_score += max(0, defense_boost) * (self.weight_stat_boosts * 0.5)
        boost_score += max(0, special_defense_boost) * (self.weight_stat_boosts * 0.5)

        # Negative stages penalty
        if is_special_attacker:
            boost_score += min(0, special_attack_boost) * (self.weight_stat_boosts * 0.8)
        if is_physical_attacker:
            boost_score += min(0, attack_boost) * (self.weight_stat_boosts * 0.8)
        boost_score += min(0, speed_boost) * (self.weight_stat_boosts * 0.8)
        boost_score += min(0, defense_boost) * (self.weight_stat_boosts * 0.5)
        boost_score += min(0, special_defense_boost) * (self.weight_stat_boosts * 0.5)
        return boost_score

    def _evaluate_status_penalty(self, pokemon: SimulatedPokemon) -> float:
        if not pokemon.status:
            return 0.0
        penalty = 0.0
        if pokemon.status == "brn":
            penalty += self.weight_status_penalty if pokemon.ability != "guts" else (self.weight_status_penalty * 0.2)
        elif pokemon.status == "par":
            penalty += self.weight_status_penalty * 0.9
        elif pokemon.status == "slp":
            penalty += self.weight_status_penalty * 1.3
        elif pokemon.status in ("psn", "tox"):
            penalty += self.weight_status_penalty * 0.6
        return penalty

    def _evaluate_threat_matrix(
        self,
        side1_active: list[SimulatedPokemon],
        side2_active: list[SimulatedPokemon],
        state: SimulatedBattleState
    ) -> float:
        """Evaluates whether active Pokemon threaten immediate 1HKOs or 2HKOs."""
        threat_score = 0.0

        for user_pokemon in side1_active:
            for opponent_pokemon in side2_active:
                if self._can_knock_out(user_pokemon, opponent_pokemon, state):
                    threat_score += self.weight_threat_pressure

        for opponent_pokemon in side2_active:
            for user_pokemon in side1_active:
                if self._can_knock_out(opponent_pokemon, user_pokemon, state):
                    threat_score -= self.weight_threat_pressure

        return threat_score

    def _can_knock_out(
        self,
        attacker: SimulatedPokemon,
        target: SimulatedPokemon,
        state: SimulatedBattleState
    ) -> bool:
        """Fast check if attacker has any damaging move capable of reducing target to 0 HP."""
        if target.protect_counter > 0:
            return False
        for move in attacker.moves.values():
            if move.category == "status" or move.base_power <= 0:
                continue
            effective_type = move.type
            type_multiplier = target.damage_multiplier(effective_type)
            if type_multiplier == 0.0:
                continue
            same_type_attack_bonus = 1.5 if effective_type in attacker.types else 1.0
            is_special = (move.category == "special")
            attacker_stat = attacker.calculate_effective_attack(is_special, state.weather, state.fields)
            defender_stat = target.calculate_effective_defense(is_special, state.weather, state.fields)

            estimated_damage = (
                ((22.0 * move.base_power * (attacker_stat / max(1.0, defender_stat))) / 50.0) + 2.0
            ) * same_type_attack_bonus * type_multiplier
            if estimated_damage >= target.current_hp:
                return True
        return False
    
    def _is_special_attacker(self, pokemon: SimulatedPokemon) -> bool:
        """Heuristic to determine if a Pokemon is a special attacker. Specifically, if the Pokemon
        has any special moves."""
        special_moves = sum(1 for move in pokemon.moves.values() if move.category == "special" and move.base_power > 0)
        if special_moves == 0:
            return False
        return True
    
    def _is_physical_attacker(self, pokemon: SimulatedPokemon) -> bool:
        """Heuristic to determine if a Pokemon is primarily a physical attacker."""
        physical_moves = sum(1 for move in pokemon.moves.values() if move.category == "physical" and move.base_power > 0)
        if physical_moves == 0:
            return False
        return True

    def _evaluate_flinch_penalty(self, pokemon: SimulatedPokemon) -> float:
        """Evaluates the penalty for a Pokemon that is flinched and cannot act this turn."""
        if pokemon.just_flinched:
            return self.weight_flinch
        return 0.0
    
class NeuralValueEvaluator(BattleStateEvaluator):
    """
    Evaluator that queries the trained neural network value head (critic) from Stable-Baselines3
    or a PyTorch actor-critic network to evaluate leaf board states.
    """

    def __init__(self, model: Any, state_encoder: Optional[StateEncoder] = None, device: str = "cpu"):
        self.model = model
        self.device = device
        self.state_encoder = state_encoder if state_encoder is not None else StateEncoder()

    def evaluate(self, state: SimulatedBattleState) -> float:
        """
        Extracts observations and queries the critic value function: V_theta(s).
        Returns the neural network value scaled to match heuristic ranges.
        """
        if self.model is None or self.state_encoder is None:
            return 0.0

        # Check if model has a policy with predict_values (standard in Stable-Baselines3 MaskablePPO)
        if hasattr(self.model, "policy") and hasattr(self.model.policy, "predict_values"):
            # If we have an active battle attached, encode it
            raw_battle = getattr(state, "_raw_battle_reference", None)
            if raw_battle is not None:
                encoded_observation = self.state_encoder.encode(raw_battle)
                tensor_observation, _ = self.model.policy.obs_to_tensor(encoded_observation)
                with torch.no_grad():
                    value_tensor = self.model.policy.predict_values(tensor_observation)
                    return float(value_tensor.cpu().numpy().item()) * 5000.0
        return 0.0

class HybridEvaluator(BattleStateEvaluator):
    """
    Blends the handcrafted competitive heuristic with an optional trained neural network value head.
    """

    def __init__(
        self,
        heuristic_evaluator: Optional[CompetitiveHeuristicEvaluator] = None,
        neural_evaluator: Optional[NeuralValueEvaluator] = None,
        neural_weight: float = 0.35
    ):
        self.heuristic_evaluator = heuristic_evaluator or CompetitiveHeuristicEvaluator()
        self.neural_evaluator = neural_evaluator
        self.neural_weight = neural_weight if neural_evaluator else 0.0

    def evaluate(self, state: SimulatedBattleState) -> float:
        heuristic_score = self.heuristic_evaluator.evaluate(state)
        if self.neural_evaluator and self.neural_weight > 0.0:
            neural_score = self.neural_evaluator.evaluate(state)
            return (1.0 - self.neural_weight) * heuristic_score + (self.neural_weight * neural_score)
        return heuristic_score
