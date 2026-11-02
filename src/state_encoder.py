from enum import Enum
from typing import Optional, Union
import poke_env.battle as pkmn_b
import torch
import numpy as np
from src import constants
from src.item_priors import ITEM_PRIORS
from src.damage_calc import evaluate_move_power_and_multipliers, get_move_type_and_bp_boost, clean_str
from src.simulation import SimulatedBattleState, SimulatedPokemon, SimulatedMove

BattleType = Union[pkmn_b.DoubleBattle, SimulatedBattleState]
PokemonType_ = Union[pkmn_b.Pokemon, SimulatedPokemon]
MoveType = Union[pkmn_b.Move, SimulatedMove]

# Hook to track unburden activation status for each Pokemon instance
_UNBURDEN_ACTIVE_IDS: set[int] = set()

def _get_unburden_active(self) -> bool:
    # Checks if the Pokemon's memory address is in the set of active unburden IDs
    return id(self) in _UNBURDEN_ACTIVE_IDS

def _set_unburden_active(self, value: bool) -> None:
    # Adds or removes the Pokemon's memory address from the set based on the value
    if value:
        _UNBURDEN_ACTIVE_IDS.add(id(self))
    else:
        _UNBURDEN_ACTIVE_IDS.discard(id(self))

# Set the unburden_active property on the Pokemon class to use the above getter and setter
setattr(pkmn_b.Pokemon, "unburden_active", property(_get_unburden_active, _set_unburden_active))

# Hook to detect used items
_original_end_item = pkmn_b.Pokemon.end_item
def _patched_end_item(self, item: str):
    self.item = "none"
    if clean_str(getattr(self, "ability", "")) == "unburden":
        self.unburden_active = True
    _original_end_item(self, item)
    
_original_switch_out = pkmn_b.Pokemon.switch_out
def _patched_switch_out(self, fields):
    self.unburden_active = False
    _original_switch_out(self, fields)

pkmn_b.Pokemon.end_item = _patched_end_item
pkmn_b.Pokemon.switch_out = _patched_switch_out

class StateEncoder:
    def _calc_turns_remaining(
        self,
        start_turn: int,
        current_turn: int,
        max_turns: int = 5,
        is_simulated: bool = False,
    ) -> float:
        if is_simulated:
            remaining = max(1, min(max_turns, start_turn))
            return remaining / float(max_turns)
        elapsed = max(0, current_turn - start_turn)
        remaining = max(1, max_turns - elapsed)
        return remaining / float(max_turns)

    def is_mega_evolved(self, pokemon: PokemonType_) -> bool:
        """
        Determines if a Pokemon is in its Mega Evolved form.
        Args:
            pokemon (PokemonType_): The Pokemon object to check.
        Returns:
            bool: True if the Pokemon is Mega Evolved, False otherwise.
        """
        species = clean_str(pokemon.species)
        is_mega = (species.endswith(("mega", "megax", "megay", "megaz")) and species != "yanmega")
        return is_mega

    def encode_weather(self, battle: BattleType) -> torch.Tensor:
        """
        Encode the weather condition of the current turn of the battle.
        Args:
            battle (BattleType): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded weather condition in one-hot format.
        """
        # ["none", "sun", "rain", "sandstorm", "hail"]
        weather_tensor = torch.zeros(5, dtype=torch.float32)
        active_weather = {clean_str(w): turn for w, turn in getattr(battle, "weather", {}).items()}
        current_turn = getattr(battle, "turn", 0)
        is_sim = isinstance(battle, SimulatedBattleState)
            
        if any(w in active_weather for w in ("sunnyday", "sun", "desolateland")):
            if "desolateland" in active_weather:
                weather_tensor[1] = 1.0
            else:
                start_turn = active_weather.get("sunnyday", active_weather.get("sun", current_turn))
                weather_tensor[1] = self._calc_turns_remaining(start_turn, current_turn, 5, is_simulated=is_sim)
        elif any(w in active_weather for w in ("raindance", "rain", "primordialsea")):
            if "primordialsea" in active_weather:
                weather_tensor[2] = 1.0
            else:
                start_turn = active_weather.get("raindance", active_weather.get("rain", current_turn))
                weather_tensor[2] = self._calc_turns_remaining(start_turn, current_turn, 5, is_simulated=is_sim)
        elif any(w in active_weather for w in ("sandstorm", "sand")):
            start_turn = active_weather.get("sandstorm", active_weather.get("sand", current_turn))
            weather_tensor[3] = self._calc_turns_remaining(start_turn, current_turn, 5, is_simulated=is_sim)
        elif any(w in active_weather for w in ("hail", "snow", "snowscape")):
            start_turn = active_weather.get("hail", active_weather.get("snow", active_weather.get("snowscape", current_turn)))
            weather_tensor[4] = self._calc_turns_remaining(start_turn, current_turn, 5, is_simulated=is_sim)
        else:
            weather_tensor[0] = 1.0
        return weather_tensor

    def encode_terrain(self, battle: BattleType) -> torch.Tensor:
        """
        Encode the terrain condition of the current turn of the battle.
        Args:
            battle (BattleType): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded terrain condition in one-hot format.
        """
        # ["none", "electric", "grassy", "misty", "psychic"]
        terrain_tensor = torch.zeros(5, dtype=torch.float32)
        active_fields = {clean_str(f): turn for f, turn in getattr(battle, "fields", {}).items()}
        current_turn = getattr(battle, "turn", 0)
        is_sim = isinstance(battle, SimulatedBattleState)
        
        if "electricterrain" in active_fields:
            start_turn = active_fields["electricterrain"]
            terrain_tensor[1] = self._calc_turns_remaining(start_turn, current_turn, 5, is_simulated=is_sim)
        elif "grassyterrain" in active_fields:
            start_turn = active_fields["grassyterrain"]
            terrain_tensor[2] = self._calc_turns_remaining(start_turn, current_turn, 5, is_simulated=is_sim)
        elif "mistyterrain" in active_fields:
            start_turn = active_fields["mistyterrain"]
            terrain_tensor[3] = self._calc_turns_remaining(start_turn, current_turn, 5, is_simulated=is_sim)
        elif "psychicterrain" in active_fields:
            start_turn = active_fields["psychicterrain"]
            terrain_tensor[4] = self._calc_turns_remaining(start_turn, current_turn, 5, is_simulated=is_sim)
        else:
            terrain_tensor[0] = 1.0  # Default to "none" if unrecognized terrain or no terrain is active
        
        return terrain_tensor

    def encode_screens(self, battle: BattleType) -> torch.Tensor:
        """
        Encode the screen conditions (reflect, light screen, aurora veil) of both players in the current turn of the battle.
        Args:
            battle (BattleType): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded screen conditions for both players.
        """
        # [net_reflect, net_light_screen, net_aurora_veil]
        screens_tensor = torch.zeros(3, dtype=torch.float32)
        side_conds = {clean_str(c): turn for c, turn in getattr(battle, "side_conditions", {}).items()}
        opponent_side_conds = {clean_str(c): turn for c, turn in getattr(battle, "opponent_side_conditions", {}).items()}
        current_turn = getattr(battle, "turn", 0)
        is_sim = isinstance(battle, SimulatedBattleState)

        own_ref = self._calc_turns_remaining(side_conds["reflect"], current_turn, 5, is_simulated=is_sim) if "reflect" in side_conds else 0.0
        own_ls = self._calc_turns_remaining(side_conds["lightscreen"], current_turn, 5, is_simulated=is_sim) if "lightscreen" in side_conds else 0.0
        own_av = self._calc_turns_remaining(side_conds["auroraveil"], current_turn, 5, is_simulated=is_sim) if "auroraveil" in side_conds else 0.0

        opp_ref = self._calc_turns_remaining(opponent_side_conds["reflect"], current_turn, 5, is_simulated=is_sim) if "reflect" in opponent_side_conds else 0.0
        opp_ls = self._calc_turns_remaining(opponent_side_conds["lightscreen"], current_turn, 5, is_simulated=is_sim) if "lightscreen" in opponent_side_conds else 0.0
        opp_av = self._calc_turns_remaining(opponent_side_conds["auroraveil"], current_turn, 5, is_simulated=is_sim) if "auroraveil" in opponent_side_conds else 0.0

        screens_tensor[0] = own_ref - opp_ref
        screens_tensor[1] = own_ls - opp_ls
        screens_tensor[2] = own_av - opp_av

        return screens_tensor

    def encode_speed_modifiers(self, battle: BattleType) -> torch.Tensor:
        """
        Encode the speed modifiers of both players' active Pokémon in the current turn of the battle.
        Args:
            battle (BattleType): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded speed modifiers for both players' active Pokémon.
        """
        # [trick_room_active, net_tailwind]
        speed_modifiers_tensor = torch.zeros(2, dtype=torch.float32)
        active_fields = {clean_str(f): turn for f, turn in getattr(battle, "fields", {}).items()}
        current_turn = getattr(battle, "turn", 0)
        is_sim = isinstance(battle, SimulatedBattleState)
        if "trickroom" in active_fields:
            start_turn = active_fields["trickroom"]
            speed_modifiers_tensor[0] = self._calc_turns_remaining(start_turn, current_turn, 5, is_simulated=is_sim)

        side_conds = {clean_str(c): turn for c, turn in getattr(battle, "side_conditions", {}).items()}
        own_tw = 0.0
        if "tailwind" in side_conds:
            start_turn = side_conds["tailwind"]
            own_tw = self._calc_turns_remaining(start_turn, current_turn, 4, is_simulated=is_sim)

        opp_side_conds = {clean_str(c): turn for c, turn in getattr(battle, "opponent_side_conditions", {}).items()}
        opp_tw = 0.0
        if "tailwind" in opp_side_conds:
            start_turn = opp_side_conds["tailwind"]
            opp_tw = self._calc_turns_remaining(start_turn, current_turn, 4, is_simulated=is_sim)

        speed_modifiers_tensor[1] = own_tw - opp_tw

        return speed_modifiers_tensor

    def encode_moves(self, pokemon: PokemonType_, battle: BattleType) -> torch.Tensor:
        """
        Encode the moves of one own Pokemon in the current turn of the battle.
        Args:
            pokemon (PokemonType_): The Pokemon object containing the current state.
            battle (BattleType): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded moves of the Pokemon.
        """
        # [is_available (1), base_power (1), category (3), type (18), effective_damage (2)] = 25 features per move = 100 for 4
        moves_tensors = []
        if isinstance(pokemon, SimulatedPokemon):
            is_own = (pokemon.side == 1)
        elif isinstance(battle, pkmn_b.DoubleBattle):
            is_own = (pokemon in battle.active_pokemon or pokemon in battle.team.values())
        else:
            is_own = True

        targets = battle.opponent_active_pokemon if is_own else battle.active_pokemon
        target_1 = targets[0] if len(targets) > 0 and targets[0] and not targets[0].fainted else None
        target_2 = targets[1] if len(targets) > 1 and targets[1] and not targets[1].fainted else None

        
        for move in pokemon.moves.values():
            category = torch.zeros(3, dtype=torch.float32)
            c_cat = clean_str(getattr(move, "category", ""))
            cat_idx = constants.CATEGORY_TO_IDX.get(c_cat)
            if cat_idx is not None:
                category[cat_idx] = 1.0
            
            effective_type, _ = get_move_type_and_bp_boost(move, pokemon, battle)
            move_type = torch.zeros(18, dtype=torch.float32)
            eff_type_clean = clean_str(effective_type)
            if eff_type_clean in constants.TYPE_TO_IDX:
                move_type[constants.TYPE_TO_IDX[eff_type_clean]] = 1.0

            nominal_bp, effective_damage_1, effective_damage_2 = evaluate_move_power_and_multipliers(
                move, pokemon, target_1, target_2, battle
            )
            
            moves_tensor = torch.cat([
                torch.tensor([0.0]) if getattr(move, "current_pp", 10) == 0 else torch.tensor([1.0]),
                torch.tensor([nominal_bp / 150.0]),
                category,
                move_type,
                torch.tensor([effective_damage_1, effective_damage_2], dtype=torch.float32)
            ])
            
            moves_tensors.append(moves_tensor)

        while len(moves_tensors) < 4:
            moves_tensors.append(torch.zeros(25, dtype=torch.float32))

        return torch.cat(moves_tensors)

    def encode_single_pokemon(self, pokemon: Optional[PokemonType_], battle: BattleType, opponent: bool = False) -> torch.Tensor:
        """
        Generates the encoding vector for a Pokemon.
        Args:
            pokemon (Optional[PokemonType_]): The Pokemon object to be encoded.
            battle (BattleType): The battle object containing the current state.
        Returns:
            torch.Tensor: The encoded tensor for the Pokemon.
        """
        # [is_present, HP, types, status, boosts, is_first_turn, is_taunted, is_encored, moves, protect_success_rate, 
        # is_mega, base_stats, eff_speed] 
        # = 46 + 100 = 146 features    
        if not pokemon or pokemon.fainted:
            return torch.zeros(146, dtype=torch.float32) # If the slot is empty or fainted
        
        type_tensor = torch.zeros(18, dtype=torch.float32)
        for pkmn_type in getattr(pokemon, "types", ()):
            c_t = clean_str(pkmn_type)
            if c_t in constants.TYPE_TO_IDX:
                type_tensor[constants.TYPE_TO_IDX[c_t]] = 1.0 
        
        # [poison, toxic, burn, freeze, paralysis, sleep, none]
        status_tensor = torch.zeros(7, dtype=torch.float32)
        p_status = getattr(pokemon, "status", None)
        if p_status:
            c_status = clean_str(p_status)
            idx = constants.STATUS_TO_IDX.get(c_status)
            if idx is not None:
                status_tensor[idx] = 1.0
            else:
                status_tensor[6] = 1.0
        else:
            status_tensor[6] = 1.0  # No status condition
        
        boost_tensor = torch.zeros(7, dtype=torch.float32)
        for stat, boost in getattr(pokemon, "boosts", {}).items():
            stat_name = clean_str(stat)
            if stat_name in constants.BOOSTABLE_STAT_TO_IDX:
                boost_tensor[constants.BOOSTABLE_STAT_TO_IDX[stat_name]] = boost / 6.0  # Normalize to [-1, 1]
            
        has_protect = any(
            clean_str(getattr(move, "id", "")) in constants.PROTECT_MOVES 
            for move in getattr(pokemon, "moves", {}).values() 
            if move is not None
        )
        protect_counter = getattr(pokemon, "protect_counter", 0)
        protect_success_rate = 0.0 if not has_protect else 1.0 / (3.0 ** protect_counter)
            
        is_mega = self.is_mega_evolved(pokemon)
        if isinstance(pokemon, SimulatedPokemon):
            is_mega = is_mega or pokemon.is_mega
        
        if isinstance(battle, pkmn_b.DoubleBattle):
            active_pokemon = battle.active_pokemon if not opponent else battle.opponent_active_pokemon
            pokemon_index = active_pokemon.index(pokemon) if pokemon in active_pokemon else -1
            can_mega = battle.can_mega_evolve[pokemon_index] if 0 <= pokemon_index < len(battle.can_mega_evolve) and not opponent else False
        elif isinstance(battle, SimulatedBattleState):
            active_pokemon = battle.active_pokemon if not opponent else battle.opponent_active_pokemon
            pokemon_index = active_pokemon.index(pokemon) if pokemon in active_pokemon else -1
            can_mega = not battle.mega_used if not opponent else not battle.opponent_mega_used
            can_mega = can_mega and pokemon.can_mega_evolve if isinstance(pokemon, SimulatedPokemon) else False
        
        stat_keys = ("atk", "def", "hp", "spa", "spd", "spe")
        base_stats_tensor = torch.tensor([float(pokemon.base_stats.get(k, 80)) for k in stat_keys], dtype=torch.float32) / 150.0
        
        hp_frac = pokemon.current_hp_fraction if pokemon.current_hp_fraction is not None else 0.0
        if pokemon.fainted:
            hp_frac = 0.0
        
        if is_mega:
            mega_tri_state = 1.0
        elif pokemon_index >= 0 and can_mega:
            mega_tri_state = 0.5
        else:
            mega_tri_state = 0.0
            
        is_first_turn = torch.tensor([1.0]) if getattr(pokemon, "first_turn", False) else torch.tensor([0.0])
        
        # Taunt check (normalized to [0.0, 1.0])
        if isinstance(pokemon, SimulatedPokemon):
            is_taunted = min(1.0, pokemon.taunt_turns / 3.0)
        elif hasattr(pokemon, "effects") and pkmn_b.Effect.TAUNT in pokemon.effects:
            elapsed = pokemon.effects[pkmn_b.Effect.TAUNT]
            is_taunted = max(0.33, (3 - elapsed) / 3.0)
        else:
            is_taunted = 0.0
        is_taunted_tensor = torch.tensor([is_taunted], dtype=torch.float32)

        # Encore check
        if isinstance(pokemon, SimulatedPokemon):
            is_encored = min(1.0, pokemon.encore_turns / 3.0)
        elif hasattr(pokemon, "effects") and pkmn_b.Effect.ENCORE in pokemon.effects:
            elapsed = pokemon.effects[pkmn_b.Effect.ENCORE]
            is_encored = max(0.33, (3 - elapsed) / 3.0)
        else:
            is_encored = 0.0
        is_encored_tensor = torch.tensor([is_encored], dtype=torch.float32)
        
        
        conds = getattr(battle, "opponent_side_conditions", {}) if opponent else getattr(battle, "side_conditions", {})
        side_conds = {clean_str(c): turn for c, turn in conds.items()}
        
        active_weather = {clean_str(w): turn for w, turn in getattr(battle, "weather", {}).items()}
        active_fields = {clean_str(f): turn for f, turn in getattr(battle, "fields", {}).items()}
        base_speed = pokemon.base_stats.get("spe", 80)
        speed_boost = pokemon.boosts.get("spe", 0)
        speed_mult = (2 + speed_boost) / 2 if speed_boost > 0 else 2 / (2 - speed_boost) if speed_boost < 0 else 1.0
        speed_mult = speed_mult * 2.0 if "tailwind" in side_conds else speed_mult
        speed_mult = speed_mult * 2.0 if getattr(pokemon, "unburden_active", False) else speed_mult

        ability_id = clean_str(getattr(pokemon, "ability", ""))
        item_id = clean_str(getattr(pokemon, "item", ""))
        status_id = clean_str(getattr(pokemon, "status", ""))
        is_sun = any(w in active_weather for w in ("sunnyday", "sun", "desolateland")) or ability_id == "megasol"
        is_rain = any(w in active_weather for w in ("raindance", "rain", "primordialsea")) and ability_id != "megasol"
        is_sand = any(w in active_weather for w in ("sandstorm", "sand"))
        is_snow = any(w in active_weather for w in ("hail", "snow", "snowscape"))

        if (
            (ability_id == "chlorophyll" and is_sun)
            or (ability_id == "swiftswim" and is_rain)
            or (ability_id == "sandrush" and is_sand)
            or (ability_id == "slushrush" and is_snow)
            or (ability_id == "surgesurfer" and "electricterrain" in active_fields)
        ):
            speed_mult *= 2.0
        elif ability_id in ("protosynthesis", "quarkdrive") and getattr(pokemon, "paradox_boost_stat", None) == "spe":
            speed_mult *= 1.5
        elif ability_id == "quickfeet" and bool(status_id):
            speed_mult *= 1.5

        if (pkmn_b.Status.PAR == getattr(pokemon, "status", "") or status_id == "par") and ability_id != "quickfeet":
            speed_mult *= 0.5
        if item_id == "choicescarf":
            speed_mult *= 1.5
        elif item_id == "ironball":
            speed_mult *= 0.5
        speed_mult = -speed_mult if "trickroom" in active_fields else speed_mult
        eff_speed = base_speed * speed_mult
        
        pkmn_tensor = torch.cat([
            torch.tensor([1.0, hp_frac]),
            type_tensor,
            status_tensor,
            boost_tensor,
            is_first_turn,
            is_taunted_tensor,
            is_encored_tensor,
            self.encode_moves(pokemon, battle),
            torch.tensor([protect_success_rate, mega_tri_state], dtype=torch.float32),
            base_stats_tensor,
            torch.tensor([max(-1.0, min(1.0, eff_speed / 500.0))], dtype=torch.float32)
        ])
        
        return pkmn_tensor

    def encode_bench(self, battle: BattleType) -> torch.Tensor:
        """
        Generates the encoding vector for the player's bench (non-active Pokémon).
        Args:
            battle (BattleType): The battle object containing the current state.
        Returns:
            torch.Tensor: The encoded tensor for the player's bench.
        """
        if isinstance(battle, pkmn_b.DoubleBattle):
            bench = [
                pokemon for pokemon in battle.team.values() 
                if pokemon is not None 
                and not pokemon.fainted 
                and pokemon not in battle.active_pokemon 
                and getattr(pokemon, "selected_in_teampreview", True)
            ]
        else:
            bench = [p for p in battle.bench if p is not None and not p.fainted]

        bench_tensors = []
        for i in range(2):
            mon = bench[i] if i < len(bench) else None
            bench_tensors.append(self.encode_single_pokemon(mon, battle=battle))

        return torch.cat(bench_tensors)
    
    def encode_opponent_bench(self, battle: BattleType) -> torch.Tensor:
        """
        Generates the encoding vector for the opponent's bench (non-active Pokémon).
        Args:
            battle (BattleType): The battle object containing the current state.
        Returns:
            torch.Tensor: The encoded tensor for the opponent's bench.
        """
        if isinstance(battle, pkmn_b.DoubleBattle):
            bench = [
                pokemon for pokemon in battle.opponent_team.values() 
                if pokemon is not None 
                and not pokemon.fainted 
                and pokemon not in battle.opponent_active_pokemon 
                and getattr(pokemon, "revealed", False)
            ]
            opp_fainted = sum(1 for p in battle.opponent_team.values() if p is not None and p.fainted)
            opp_active_alive = sum(1 for p in battle.opponent_active_pokemon if p is not None and not p.fainted)
            opp_brought_cap = min(4, max(len(battle.opponent_team), opp_active_alive + len(bench) + opp_fainted))
            unrevealed_alive = max(0, opp_brought_cap - opp_fainted - opp_active_alive - len(bench))
        else:
            bench = [p for p in battle.opponent_bench if p is not None and not p.fainted]
            unrevealed_alive = getattr(battle, "unrevealed_opponent_count", 0)

        bench_tensors = []
        unrevealed_used = 0
        for i in range(2):
            if i < len(bench):
                bench_tensors.append(self.encode_single_pokemon(bench[i], battle=battle, opponent=True))
            elif unrevealed_used < unrevealed_alive:
                # Unrevealed reserve Pokemon in back: alive (is_present=1, hp_frac=1, status=none), species unknown
                unrevealed_tensor = torch.zeros(146, dtype=torch.float32)
                unrevealed_tensor[0] = 1.0
                unrevealed_tensor[1] = 1.0
                unrevealed_tensor[26] = 1.0
                bench_tensors.append(unrevealed_tensor)
                unrevealed_used += 1
            else:
                bench_tensors.append(self.encode_single_pokemon(None, battle=battle, opponent=True))

        return torch.cat(bench_tensors)
    
    def encode_single_ability(self, pokemon: Optional[PokemonType_]) -> torch.Tensor:
        """
        Encodes a single Pokemon's ability into a 215-dim vector.
        Args:
            pokemon (Optional[PokemonType_]): The Pokemon object to be encoded.
        Returns:
            torch.Tensor: A tensor representing the encoded ability of the Pokemon.
        """
        vec = torch.zeros(215, dtype=torch.float32)
        if not pokemon or pokemon.fainted:
            return vec

        clean_name = clean_str(pokemon.ability)
        if clean_name and clean_name != "none":
            ability_idx = constants.ABILITY_TO_IDX.get(clean_name)
            if ability_idx is not None:
                vec[ability_idx] = 1.0
                return vec

        if isinstance(pokemon, pkmn_b.Pokemon) and pokemon.possible_abilities:
            valid_indices = [
                constants.ABILITY_TO_IDX[clean_str(a)]
                for a in pokemon.possible_abilities
                if clean_str(a) in constants.ABILITY_TO_IDX
            ]
            if valid_indices:
                prob = 1.0 / len(valid_indices)
                for a_idx in valid_indices:
                    vec[a_idx] = prob

        return vec

    def encode_abilities(self, battle: BattleType) -> torch.Tensor:
        """
        Encodes abilities of all active and bench Pokemon for both players into an 8x215 tensor in consistent order.
        Args:
            battle (BattleType): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded abilities of all relevant Pokemon.
        """
        abilities_tensor = torch.zeros((8, 215), dtype=torch.float32)

        for i in range(2):
            p1_mon = battle.active_pokemon[i] if i < len(battle.active_pokemon) else None
            abilities_tensor[i] = self.encode_single_ability(p1_mon)

            p2_mon = battle.opponent_active_pokemon[i] if i < len(battle.opponent_active_pokemon) else None
            abilities_tensor[2 + i] = self.encode_single_ability(p2_mon)

        if isinstance(battle, pkmn_b.DoubleBattle):
            own_bench = [
                p for p in battle.team.values()
                if p is not None
                and not p.fainted
                and p not in battle.active_pokemon
                and getattr(p, "selected_in_teampreview", True)
            ]
        else:
            own_bench = [p for p in battle.bench if p is not None and not p.fainted]

        for i in range(2):
            mon = own_bench[i] if i < len(own_bench) else None
            abilities_tensor[4 + i] = self.encode_single_ability(mon)

        if isinstance(battle, pkmn_b.DoubleBattle):
            opp_bench = [
                p for p in battle.opponent_team.values()
                if p is not None
                and not p.fainted
                and p not in battle.opponent_active_pokemon
                and getattr(p, "revealed", False)
            ]
        else:
            opp_bench = [p for p in battle.opponent_bench if p is not None and not p.fainted]

        for i in range(2):
            mon = opp_bench[i] if i < len(opp_bench) else None
            abilities_tensor[6 + i] = self.encode_single_ability(mon)

        return abilities_tensor
    
    def encode_single_item(self, pokemon: Optional[PokemonType_]) -> torch.Tensor:
        """
        Encodes a single Pokemon's item.
        Args:
            pokemon (Optional[PokemonType_]): The Pokemon object to be encoded.
        Returns:
            torch.Tensor: A tensor representing the encoded item of the Pokemon.
        """
        vec = torch.zeros(167, dtype=torch.float32)
        if not pokemon or pokemon.fainted:
            return vec
        
        raw_item = pokemon.item
        clean_name = clean_str(raw_item)
        if clean_name in ("none", ""):
            none_idx = constants.ITEM_TO_IDX.get("none")
            if none_idx is not None:
                vec[none_idx] = 1.0
            return vec

        if clean_name:
            item_idx = constants.ITEM_TO_IDX.get(clean_name)
            if item_idx is not None:
                vec[item_idx] = 1.0
                return vec

        is_mega = self.is_mega_evolved(pokemon)
        species = clean_str(pokemon.species)
        if isinstance(pokemon, SimulatedPokemon):
            is_mega = is_mega or pokemon.is_mega
        if is_mega:
            mega_item = constants.MEGA_TO_MEGA_STONE.get(species)
            if not mega_item:
                return vec
            item_idx = constants.ITEM_TO_IDX.get(mega_item)
            if item_idx is not None:
                vec[item_idx] = 1.0
                return vec
        
        item_prior = ITEM_PRIORS.get(species, None)
        if item_prior:
            for item, prob in item_prior.items():
                item_idx = constants.ITEM_TO_IDX.get(item)
                if item_idx is not None:
                    vec[item_idx] = prob

        return vec
    
    def encode_items(self, battle: BattleType) -> torch.Tensor:
        """
        Encodes items of all active and bench Pokemon for both players into an 8x167 tensor in consistent order.
        Args:
            battle (BattleType): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded items of all relevant Pokemon.
        """
        items_tensor = torch.zeros((8, 167), dtype=torch.float32)

        for i in range(2):
            p1_mon = battle.active_pokemon[i] if i < len(battle.active_pokemon) else None
            items_tensor[i] = self.encode_single_item(p1_mon)

            p2_mon = battle.opponent_active_pokemon[i] if i < len(battle.opponent_active_pokemon) else None
            items_tensor[2 + i] = self.encode_single_item(p2_mon)

        if isinstance(battle, pkmn_b.DoubleBattle):
            own_bench = [
                p for p in battle.team.values()
                if p is not None
                and not p.fainted
                and p not in battle.active_pokemon
                and getattr(p, "selected_in_teampreview", True)
            ]
        else:
            own_bench = [p for p in battle.bench if p is not None and not p.fainted]

        for i in range(2):
            mon = own_bench[i] if i < len(own_bench) else None
            items_tensor[4 + i] = self.encode_single_item(mon)

        if isinstance(battle, pkmn_b.DoubleBattle):
            opp_bench = [
                p for p in battle.opponent_team.values()
                if p is not None
                and not p.fainted
                and p not in battle.opponent_active_pokemon
                and getattr(p, "revealed", False)
            ]
        else:
            opp_bench = [p for p in battle.opponent_bench if p is not None and not p.fainted]

        for i in range(2):
            mon = opp_bench[i] if i < len(opp_bench) else None
            items_tensor[6 + i] = self.encode_single_item(mon)

        return items_tensor
    
    # def mega_consumed(self, battle: BattleType) -> torch.Tensor:
    #     """
    #     Determines if a Mega Evolution has been consumed for both players.
    #     Args:
    #         battle (BattleType): The battle object containing the current state.
    #     Returns:
    #         torch.Tensor: A tensor indicating if a Mega Evolution has been consumed for both players.
    #     """
    #     mega_tensor = torch.zeros(2, dtype=torch.float32)
    #     if isinstance(battle, SimulatedBattleState):
    #         mega_tensor[0] = 1.0 if battle.mega_used else 0.0
    #         mega_tensor[1] = 1.0 if battle.opponent_mega_used else 0.0
    #         return mega_tensor
    #     if isinstance(battle, pkmn_b.DoubleBattle):
    #         opp_bench = [
    #             p for p in battle.opponent_team.values()
    #             if p is not None
    #             and not p.fainted
    #             and p not in battle.opponent_active_pokemon
    #             and getattr(p, "revealed", False)
    #         ]
    #         bench = [
    #             p for p in battle.team.values()
    #             if p is not None
    #             and not p.fainted
    #             and p not in battle.active_pokemon
    #             and getattr(p, "selected_in_teampreview", True)
    #         ]
    #         mega_tensor[0] = 1.0 if any(self.is_mega_evolved(p) for p in bench) or any(self.is_mega_evolved(p) for p in battle.active_pokemon) else 0.0
    #     return mega_tensor
    
    def encode(self, battle: BattleType) -> dict[str, np.ndarray]:
        """
        Generates the complete encoding vector for the current state of the battle.
        Args:
            battle (BattleType): The battle object containing the current state.
        Returns:
            dict[str, np.ndarray]: The complete encoded arrays for the battle state.
        """
        field_effects = torch.cat([
            self.encode_weather(battle),
            self.encode_terrain(battle),
            self.encode_screens(battle),
            self.encode_speed_modifiers(battle)
        ])
        active_tensors = []
        for i in range(2):
            mon = battle.active_pokemon[i] if i < len(battle.active_pokemon) else None
            active_tensors.append(self.encode_single_pokemon(mon, battle=battle))
        active_pokemon = torch.cat(active_tensors)

        opp_active_tensors = []
        for i in range(2):
            mon = battle.opponent_active_pokemon[i] if i < len(battle.opponent_active_pokemon) else None
            opp_active_tensors.append(self.encode_single_pokemon(mon, battle=battle, opponent=True))
        opp_active_pokemon = torch.cat(opp_active_tensors)

        bench = self.encode_bench(battle)
        opp_bench = self.encode_opponent_bench(battle)
        
        return {
            "numeric": torch.cat([
                field_effects, 
                active_pokemon, 
                opp_active_pokemon, 
                bench, 
                opp_bench,
            ]).numpy(),
            "abilities": self.encode_abilities(battle).numpy(),
            "items": self.encode_items(battle).numpy()
        }
    
class TeamPreviewEncoder:
    def __init__(self, state_encoder: Optional[StateEncoder] = None):
        self.state_encoder = state_encoder if state_encoder is not None else StateEncoder()

    def encode(self, battle: BattleType) -> dict[str, torch.Tensor]:
        """
        Generates the encoding tensors for the team preview state of the battle.
        Args:
            battle (BattleType): The battle object containing the current state.
        Returns:
            dict[str, torch.Tensor]: The complete encoded arrays for the team preview state.
        """
        if isinstance(battle, pkmn_b.DoubleBattle):
            p1_mons = (list(battle.team.values()) + [None] * 6)[:6]
            p2_mons = (list(battle.opponent_team.values()) + [None] * 6)[:6]
        else:
            p1_mons = (battle.all_team_pokemon() + [None] * 6)[:6]
            p2_mons = (battle.all_opponent_pokemon() + [None] * 6)[:6]
        all_mons = p1_mons + p2_mons
        
        numeric_list = []
        abilities_list = []
        items_list = []
        
        for mon in all_mons:
            numeric_list.append(self.state_encoder.encode_single_pokemon(mon, battle)) # (142,)
            abilities_list.append(self.state_encoder.encode_single_ability(mon))        # (215,)
            items_list.append(self.state_encoder.encode_single_item(mon))               # (167,)

        return {
            "numeric": torch.cat(numeric_list),       # Shape: (12 * 142,) = (1704,)
            "abilities": torch.stack(abilities_list), # Shape: (12, 215)
            "items": torch.stack(items_list),         # Shape: (12, 167)
        }