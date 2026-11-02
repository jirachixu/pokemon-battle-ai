"""
Competitive battle state evaluator with optional Neural Network Value head integration.
Evaluates board states from Player 1's perspective.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Any, overload
import torch
import numpy as np

from src.simulation import (
    SimulatedBattleState,
    SimulatedPokemon,
    SimulatedMove,
    clean_id,
    is_pokemon_grounded,
    is_charge_move_instant,
    get_effective_move_priority,
    _SOUND_MOVES,
    _TWO_TURN_CHARGE_MOVES,
)
from src.state_encoder import StateEncoder

class BattleStateEvaluator(ABC):
    """Abstract base class for all battle state evaluators."""
    @abstractmethod
    def evaluate(self, state: SimulatedBattleState | List[SimulatedBattleState]) -> torch.Tensor:
        """Computes evaluation score for state. Higher is better for Side 1."""
        ...

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
        weight_speed_control: float = 750.0,
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

    def evaluate(self, state: SimulatedBattleState | List[SimulatedBattleState]) -> torch.Tensor:
        if isinstance(state, list):
            return torch.tensor([self._evaluate_single(s) for s in state], dtype=torch.float32)
        return torch.tensor(self._evaluate_single(state), dtype=torch.float32)

    def _evaluate_single(self, state: SimulatedBattleState) -> float:
        """Computes evaluation score for state. Higher is better for Side 1."""
        winner_side = state.winner()
        if winner_side == 1:
            return self.weight_win_loss - (state.turn * 10.0)  # Reward faster wins
        elif winner_side == 2:
            return -self.weight_win_loss + (state.turn * 10.0)  # Reward slower losses
        elif winner_side == 0:
            return 0.0

        evaluation_score = 0.0

        team_alive_count = state.alive_team_count()
        opp_alive_count = state.alive_opponent_count()

        # 1. Material (Remaining Pokemon) Advantage
        evaluation_score += self.weight_alive_differential * (team_alive_count - opp_alive_count)

        # 2. HP Differential
        team_total_hp = sum(pokemon.current_hp_fraction for pokemon in state.all_team_pokemon() if not pokemon.fainted)
        opp_total_hp = (
            sum(pokemon.current_hp_fraction for pokemon in state.all_opponent_pokemon() if not pokemon.fainted)
            + float(getattr(state, "unrevealed_opponent_count", 0))
        )
        evaluation_score += self.weight_hp_differential * (team_total_hp - opp_total_hp)

        # Active HP priority: active Pokemon on field are more immediately impactful than bench
        team_active_hp = sum(pokemon.current_hp_fraction for pokemon in state.active_pokemon if pokemon and not pokemon.fainted)
        opp_active_hp = sum(pokemon.current_hp_fraction for pokemon in state.opponent_active_pokemon if pokemon and not pokemon.fainted)
        evaluation_score += (self.weight_hp_differential * 0.5) * (team_active_hp - opp_active_hp)

        # 3. Speed Control & Field Advantage
        is_trick_room_active = "trickroom" in state.fields
        team_tailwind_turns = state.side_conditions.get("tailwind", 0)
        opp_tailwind_turns = state.opponent_side_conditions.get("tailwind", 0)
        evaluation_score += self.weight_speed_control * (team_tailwind_turns - opp_tailwind_turns)

        # Active speed order comparison
        team_living_active = [pokemon for pokemon in state.active_pokemon if pokemon and not pokemon.fainted]
        opp_living_active = [pokemon for pokemon in state.opponent_active_pokemon if pokemon and not pokemon.fainted]

        if team_living_active and opp_living_active:
            team_effective_speeds = [
                pokemon.calculate_effective_speed(state.weather, state.fields, state.side_conditions)
                for pokemon in team_living_active
            ]
            opp_effective_speeds = [
                pokemon.calculate_effective_speed(state.weather, state.fields, state.opponent_side_conditions)
                for pokemon in opp_living_active
            ]

            max_team_speed = max(team_effective_speeds)
            max_opp_speed = max(opp_effective_speeds)

            if is_trick_room_active:
                # Slower speeds move first under Trick Room
                min_team_speed = min(team_effective_speeds)
                min_opp_speed = min(opp_effective_speeds)
                if min_team_speed < min_opp_speed:
                    evaluation_score += self.weight_speed_control
                else:
                    evaluation_score -= self.weight_speed_control
            else:
                if team_tailwind_turns == 0 and opp_tailwind_turns == 0:
                    if max_team_speed > max_opp_speed:
                        evaluation_score += self.weight_speed_control
                    else:
                        evaluation_score -= self.weight_speed_control

        # 4. Stat Boosts Advantage
        for pokemon in state.all_team_pokemon():
            if not pokemon.fainted:
                evaluation_score += self._evaluate_pokemon_boosts(pokemon)
        for pokemon in state.all_opponent_pokemon():
            if not pokemon.fainted:
                evaluation_score -= self._evaluate_pokemon_boosts(pokemon)

        # 5. Status Conditions Penalty
        for pokemon in state.all_team_pokemon():
            if not pokemon.fainted:
                evaluation_score -= self._evaluate_status_penalty(pokemon)
        for pokemon in state.all_opponent_pokemon():
            if not pokemon.fainted:
                evaluation_score += self._evaluate_status_penalty(pokemon)
        # 5.1. Flinch Penalty
        for pokemon in state.all_team_pokemon():
            if not pokemon.fainted:
                evaluation_score -= self._evaluate_flinch_penalty(pokemon)
        for pokemon in state.all_opponent_pokemon():
            if not pokemon.fainted:
                evaluation_score += self._evaluate_flinch_penalty(pokemon)

        # 5.2. Perish Song & Trapping Penalty (Active Pokemon)
        for pokemon in team_living_active:
            evaluation_score -= self._evaluate_perish_and_trapping_penalty(pokemon, state)
        for pokemon in opp_living_active:
            evaluation_score += self._evaluate_perish_and_trapping_penalty(pokemon, state)

        # 6. Immediate Lethal KO Threat Evaluation
        evaluation_score += self._evaluate_threat_matrix(team_living_active, opp_living_active, state)

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
        active_pokemon: list[SimulatedPokemon],
        opponent_active_pokemon: list[SimulatedPokemon],
        state: SimulatedBattleState
    ) -> float:
        """Evaluates whether active Pokemon threaten immediate 1HKOs or 2HKOs."""
        threat_score = 0.0

        for user_pokemon in active_pokemon:
            for opponent_pokemon in opponent_active_pokemon:
                if self._can_knock_out(user_pokemon, opponent_pokemon, state):
                    threat_score += self.weight_threat_pressure

        for opponent_pokemon in opponent_active_pokemon:
            for user_pokemon in active_pokemon:
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

        ignores_ability = attacker.ability in ("moldbreaker", "teravolt", "turboblaze")
        target_eff_ability = (
            target.ability
            if (not ignores_ability or target.ability in ("fullmetalbody", "shadowshield", "prismarmor", "comatose"))
            else ""
        )
        was_at_full_hp = (target.current_hp >= target.max_hp - 1e-6)
        if was_at_full_hp and target.max_hp > 1.0 and (target_eff_ability == "sturdy" or target.item == "focussash"):
            return False  # Cannot 1-hit KO through intact Sturdy or Focus Sash

        all_active = [p for p in state.active_pokemon + state.opponent_active_pokemon if p and not p.fainted]
        target_side_conditions = state.opponent_side_conditions if target.side == 2 else state.side_conditions

        for move in attacker.moves.values():
            if move.category == "status":
                continue
            if attacker.charging_move and move.id != attacker.charging_move:
                continue
            if move.id in ("fakeout", "firstimpression") and not attacker.first_turn:
                continue
            if attacker.throat_chop_turns > 0 and ("sound" in move.flags or move.id in _SOUND_MOVES):
                continue
            if (move.id in _TWO_TURN_CHARGE_MOVES or "charge" in move.flags) and not is_charge_move_instant(move.id, attacker, state.weather):
                continue  # Requires a charge turn first, cannot 1-turn KO right now
            if target_eff_ability == "soundproof" and ("sound" in move.flags or move.id in _SOUND_MOVES):
                continue
            if target_eff_ability == "bulletproof" and "bullet" in move.flags:
                continue

            eff_priority = get_effective_move_priority(move, attacker, state.fields)
            if eff_priority > 0:
                if "psychicterrain" in state.fields and is_pokemon_grounded(target, state.fields, ignore_ability=ignores_ability):
                    continue
                if not ignores_ability and target.ability in ("dazzling", "queenlymajesty", "armortail"):
                    continue

            base_power = move.base_power
            effective_type = move.type

            if move.id == "expandingforce" and "psychicterrain" in state.fields and is_pokemon_grounded(attacker, state.fields):
                base_power = 120.0
            elif move.id == "risingvoltage" and "electricterrain" in state.fields and is_pokemon_grounded(target, state.fields, ignore_ability=ignores_ability):
                base_power = 140.0
            elif move.id == "psyblade" and "electricterrain" in state.fields:
                base_power = 120.0
            elif move.id == "mistyexplosion" and "mistyterrain" in state.fields and is_pokemon_grounded(attacker, state.fields):
                base_power = 150.0
            elif move.id == "weatherball":
                if any(w in state.weather for w in ("sunnyday", "desolateland", "sun")) or attacker.ability == "megasol":
                    base_power, effective_type = 100.0, "fire"
                elif any(w in state.weather for w in ("raindance", "primordialsea", "rain")):
                    base_power, effective_type = 100.0, "water"
                elif any(w in state.weather for w in ("sandstorm", "sand")):
                    base_power, effective_type = 100.0, "rock"
                elif any(w in state.weather for w in ("snowscape", "hail", "snow")):
                    base_power, effective_type = 100.0, "ice"
            elif move.id == "terrainpulse" and is_pokemon_grounded(attacker, state.fields):
                if "electricterrain" in state.fields: base_power, effective_type = 100.0, "electric"
                elif "grassyterrain" in state.fields: base_power, effective_type = 100.0, "grass"
                elif "psychicterrain" in state.fields: base_power, effective_type = 100.0, "psychic"
                elif "mistyterrain" in state.fields: base_power, effective_type = 100.0, "fairy"

            if base_power <= 0:
                continue

            if effective_type == "normal":
                if attacker.ability == "pixilate": effective_type, base_power = "fairy", base_power * 1.2
                elif attacker.ability == "aerilate": effective_type, base_power = "flying", base_power * 1.2
                elif attacker.ability == "refrigerate": effective_type, base_power = "ice", base_power * 1.2
                elif attacker.ability == "galvanize": effective_type, base_power = "electric", base_power * 1.2
                elif attacker.ability == "dragonize": effective_type, base_power = "dragon", base_power * 1.2

            has_gravity = ("gravity" in state.fields)
            type_multiplier = target.damage_multiplier(effective_type, has_gravity=has_gravity)
            if type_multiplier == 0.0:
                if attacker.ability in ("scrappy", "mindseye") and effective_type in ("normal", "fighting"):
                    type_multiplier = 1.0
                else:
                    continue

            if effective_type == "ground" and target_eff_ability in ("levitate", "eelevate", "eartheater") and not is_pokemon_grounded(target, state.fields, ignore_ability=ignores_ability):
                continue
            if effective_type == "ground" and target_eff_ability == "eartheater":
                continue
            if effective_type == "fire" and target_eff_ability in ("flashfire", "wellbakedbody"):
                continue
            if effective_type == "water" and target_eff_ability in ("waterabsorb", "stormdrain", "dryskin"):
                continue
            if effective_type == "electric" and target_eff_ability in ("voltabsorb", "lightningrod", "motordrive"):
                continue
            if effective_type == "grass" and target_eff_ability == "sapsipper":
                continue

            if target_eff_ability == "terashell" and was_at_full_hp and type_multiplier > 0.0:
                type_multiplier = 0.5

            same_type_attack_bonus = (2.0 if attacker.ability == "adaptability" else 1.5) if effective_type in attacker.types else 1.0
            is_special = (move.category == "special")
            ignore_target_def_boosts = (attacker.ability == "unaware")
            ignore_attacker_atk_boosts = (target_eff_ability == "unaware")

            attacker_stat = attacker.calculate_effective_attack(
                is_special, state.weather, state.fields, ignore_boosts=ignore_attacker_atk_boosts
            )
            if move.id in ("electroshot", "meteorbeam") and attacker.charging_move != move.id and not ignore_attacker_atk_boosts:
                attacker_stat *= 1.5  # +1 SpA boost applied before 1-turn execution
            defender_stat = target.calculate_effective_defense(
                is_special, state.weather, state.fields, ignore_boosts=ignore_target_def_boosts, ignore_ability=ignores_ability
            )

            # Ruinous Quartet field reductions
            if not is_special:
                if attacker.ability != "tabletsofruin" and any(p.ability == "tabletsofruin" for p in all_active):
                    attacker_stat *= 0.75
                if target.ability != "swordofruin" and any(p.ability == "swordofruin" for p in all_active):
                    defender_stat *= 0.75
            else:
                if attacker.ability != "vesselofruin" and any(p.ability == "vesselofruin" for p in all_active):
                    attacker_stat *= 0.75
                if target.ability != "beadsofruin" and any(p.ability == "beadsofruin" for p in all_active):
                    defender_stat *= 0.75

            field_mult = 1.0
            has_aura_break = any(p.ability == "aurabreak" for p in all_active)
            if effective_type == "fairy" and any(p.ability == "fairyaura" for p in all_active):
                field_mult *= 0.75 if has_aura_break else (4.0 / 3.0)
            elif effective_type == "dark" and any(p.ability == "darkaura" for p in all_active):
                field_mult *= 0.75 if has_aura_break else (4.0 / 3.0)

            is_sunny = any(w in state.weather for w in ("sunnyday", "desolateland", "sun")) or attacker.ability == "megasol"
            is_rainy = any(w in state.weather for w in ("raindance", "primordialsea", "rain")) and attacker.ability != "megasol"
            is_bad_solar_weather = (
                any(w in state.weather for w in ("raindance", "primordialsea", "rain", "sandstorm", "sand", "snowscape", "hail", "snow"))
                and attacker.ability != "megasol"
            )
            if move.id in ("solarbeam", "solarblade") and is_bad_solar_weather:
                field_mult *= 0.5
            if effective_type == "fire":
                if is_sunny: field_mult *= 1.5
                elif is_rainy: field_mult *= 0.5
            elif effective_type == "water":
                if move.id == "hydrosteam" and is_sunny: field_mult *= 1.5
                elif is_rainy: field_mult *= 1.5
                elif is_sunny: field_mult *= 0.5

            if is_pokemon_grounded(attacker, state.fields):
                if "psychicterrain" in state.fields and effective_type == "psychic": field_mult *= 1.3
                elif "electricterrain" in state.fields and effective_type == "electric": field_mult *= 1.3
                elif "grassyterrain" in state.fields and effective_type == "grass": field_mult *= 1.3
            if is_pokemon_grounded(target, state.fields, ignore_ability=ignores_ability):
                if "mistyterrain" in state.fields and effective_type == "dragon":
                    field_mult *= 0.5
                if "grassyterrain" in state.fields and move.id in ("earthquake", "bulldoze", "magnitude"):
                    field_mult *= 0.5

            # Defensive ability & screen reductions
            if target_eff_ability in ("multiscale", "shadowshield") and was_at_full_hp:
                field_mult *= 0.5
            if target_eff_ability == "thickfat" and effective_type in ("fire", "ice"):
                field_mult *= 0.5
            elif target_eff_ability in ("heatproof", "waterbubble") and effective_type == "fire":
                field_mult *= 0.5
            elif target_eff_ability == "purifyingsalt" and effective_type == "ghost":
                field_mult *= 0.5
            if target_eff_ability == "fluffy":
                if "contact" in move.flags and attacker.item != "protectivepads":
                    field_mult *= 0.5
                if effective_type == "fire":
                    field_mult *= 2.0
            if target_eff_ability in ("filter", "solidrock", "prismarmor") and type_multiplier > 1.0:
                field_mult *= 0.75

            if attacker.ability != "infiltrator" and move.id not in ("brickbreak", "psychicfangs", "ragingbull"):
                if not is_special and ("reflect" in target_side_conditions or "auroraveil" in target_side_conditions):
                    field_mult *= 0.66
                elif is_special and ("lightscreen" in target_side_conditions or "auroraveil" in target_side_conditions):
                    field_mult *= 0.66

            estimated_damage = (
                ((22.0 * base_power * (attacker_stat / max(1.0, defender_stat))) / 50.0) + 2.0
            ) * same_type_attack_bonus * field_mult * type_multiplier
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

    def _evaluate_perish_and_trapping_penalty(
        self,
        pokemon: SimulatedPokemon,
        state: SimulatedBattleState,
    ) -> float:
        """Evaluates the danger of active Perish Song countdowns and trapping effects."""
        penalty = 0.0
        is_trapped = state.is_pokemon_trapped(pokemon)
        bench_list = state.bench if pokemon.side == 1 else state.opponent_bench
        has_alive_bench = any(b is not None and not b.fainted for b in bench_list)
        cannot_escape = is_trapped or not has_alive_bench

        if pokemon.perish_turns > 0:
            if cannot_escape:
                if pokemon.perish_turns == 1:
                    penalty += self.weight_alive_differential * 0.80
                elif pokemon.perish_turns == 2:
                    penalty += self.weight_alive_differential * 0.55
                else:
                    penalty += self.weight_alive_differential * 0.35
            else:
                if pokemon.perish_turns == 1:
                    penalty += self.weight_alive_differential * 0.45
                elif pokemon.perish_turns == 2:
                    penalty += self.weight_alive_differential * 0.20
                else:
                    penalty += self.weight_alive_differential * 0.10

        if is_trapped and has_alive_bench:
            opp_active = state.opponent_active_pokemon if pokemon.side == 1 else state.active_pokemon
            opp_has_perish = any(
                opp is not None and not opp.fainted and "perishsong" in opp.moves
                for opp in opp_active
            )
            penalty += 300.0 if opp_has_perish else 125.0

        return penalty
    
class NeuralValueEvaluator(BattleStateEvaluator):
    """
    Evaluator that queries the trained neural network value head (critic) from Stable-Baselines3
    or a PyTorch actor-critic network to evaluate leaf board states.
    """
    def __init__(self, model: Any, state_encoder: Optional[StateEncoder] = None):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_encoder = state_encoder if state_encoder is not None else StateEncoder()

    def evaluate_batch(self, states: List[SimulatedBattleState]) -> torch.Tensor:
        """
        Batches multiple simulated states into a single tensor dict and queries the critic.
        Returns a 1D tensor of values: shape (batch_size,).
        """
        if self.model is None or self.state_encoder is None or not states:
            return torch.zeros(len(states), device=self.device)
        
        if hasattr(self.model, "policy") and hasattr(self.model.policy, "predict_values"):
            encoded_list = [self.state_encoder.encode(s) for s in states]
            # Batch the numpy arrays along axis 0 for each key
            batched_np = {
                key: np.stack([obs[key] for obs in encoded_list], axis=0)
                for key in encoded_list[0].keys()
            }
            tensor_obs, _ = self.model.policy.obs_to_tensor(batched_np)
            with torch.no_grad():
                value_tensor = self.model.policy.predict_values(tensor_obs).view(-1)
                return torch.clamp(value_tensor, -1.0, 1.0)
        return torch.zeros(len(states), device=self.device)
    
    def evaluate(self, state: SimulatedBattleState | List[SimulatedBattleState]) -> torch.Tensor:
        """
        Evaluates a single state or a batch of states.
        """
        if isinstance(state, list):
            return self.evaluate_batch(state)
        return self.evaluate_batch([state])[0]

class HybridEvaluator(BattleStateEvaluator):
    """
    Blends the handcrafted competitive heuristic with an optional trained neural network value head.
    """
    def __init__(
        self,
        model: Any,
        heuristic_evaluator: Optional[CompetitiveHeuristicEvaluator] = None,
        neural_evaluator: Optional[NeuralValueEvaluator] = None,
        neural_weight: float = 0.35
    ):
        self.heuristic_evaluator = heuristic_evaluator or CompetitiveHeuristicEvaluator()
        self.neural_evaluator = neural_evaluator or (NeuralValueEvaluator(model=model) if model is not None else None)
        self.neural_weight = neural_weight if self.neural_evaluator else 0.0

    def evaluate_batch(self, states: List[SimulatedBattleState]) -> torch.Tensor:
        heuristic_scores = self.heuristic_evaluator.evaluate(states)
        heuristic_normalized = torch.tanh(heuristic_scores / 8000.0)
        if self.neural_evaluator and self.neural_weight > 0.0:
            neural_scores = self.neural_evaluator.evaluate(states).to(heuristic_normalized.device)
            return (1.0 - self.neural_weight) * heuristic_normalized + (self.neural_weight * neural_scores)
        return heuristic_normalized

    def evaluate(self, state: SimulatedBattleState | List[SimulatedBattleState]) -> torch.Tensor:
        if isinstance(state, list):
            return self.evaluate_batch(state)
        return self.evaluate_batch([state])[0]
