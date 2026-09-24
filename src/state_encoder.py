import poke_env.battle as pkmn_b
import torch
from src import constants
import numpy as np
from src.item_priors import ITEM_PRIORS
from src.damage_calc import evaluate_move_power_and_multipliers, get_move_type_and_bp_boost

# Hook to detect used items
_original_end_item = pkmn_b.Pokemon.end_item
def _patched_end_item(self, item: str):
    self.item = "none"
    _original_end_item(self, item)

pkmn_b.Pokemon.end_item = _patched_end_item

class StateEncoder:
    def encode_weather(self, battle: pkmn_b.DoubleBattle) -> torch.Tensor:
        """
        Encode the weather condition of the current turn of the battle.
        Args:
            battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded weather condition in one-hot format.
        """
        # ["none", "sun", "rain", "sandstorm", "hail"]
        weather_tensor = torch.zeros(5, dtype=torch.float32)
            
        if pkmn_b.Weather.SUNNYDAY in battle.weather or pkmn_b.Weather.DESOLATELAND in battle.weather:
            weather_tensor[1] = 1.0
        elif pkmn_b.Weather.RAINDANCE in battle.weather or pkmn_b.Weather.PRIMORDIALSEA in battle.weather:
            weather_tensor[2] = 1.0
        elif pkmn_b.Weather.SANDSTORM in battle.weather:
            weather_tensor[3] = 1.0
        elif pkmn_b.Weather.HAIL in battle.weather:
            weather_tensor[4] = 1.0
        else:
            weather_tensor[0] = 1.0  # Default to "none" if unrecognized weather or no weather is active
        
        return weather_tensor

    def encode_terrain(self, battle: pkmn_b.DoubleBattle) -> torch.Tensor:
        """
        Encode the terrain condition of the current turn of the battle.
        Args:
            battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded terrain condition in one-hot format.
        """
        # ["none", "electric", "grassy", "misty", "psychic"]
        terrain_tensor = torch.zeros(5, dtype=torch.float32)
        
        if pkmn_b.Field.ELECTRIC_TERRAIN in battle.fields:
            terrain_tensor[1] = 1.0
        elif pkmn_b.Field.GRASSY_TERRAIN in battle.fields:
            terrain_tensor[2] = 1.0
        elif pkmn_b.Field.MISTY_TERRAIN in battle.fields:
            terrain_tensor[3] = 1.0
        elif pkmn_b.Field.PSYCHIC_TERRAIN in battle.fields:
            terrain_tensor[4] = 1.0
        else:
            terrain_tensor[0] = 1.0  # Default to "none" if unrecognized terrain or no terrain is active
        
        return terrain_tensor

    def encode_speed_modifiers(self, battle: pkmn_b.DoubleBattle) -> torch.Tensor:
        """
        Encode the speed modifiers of both players' active Pokémon in the current turn of the battle.
        Args:
            battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded speed modifiers for both players' active Pokémon.
        """
        # [trick_room_active, tailwind_active, opponent_tailwind_active]
        speed_modifiers_tensor = torch.zeros(3, dtype=torch.float32)
        
        if pkmn_b.Field.TRICK_ROOM in battle.fields:
            speed_modifiers_tensor[0] = 1.0
        if pkmn_b.SideCondition.TAILWIND in battle.side_conditions:
            speed_modifiers_tensor[1] = 1.0
        if pkmn_b.SideCondition.TAILWIND in battle.opponent_side_conditions:
            speed_modifiers_tensor[2] = 1.0

        return speed_modifiers_tensor

    def encode_moves(self, pokemon: pkmn_b.Pokemon, battle: pkmn_b.DoubleBattle) -> torch.Tensor:
        """
        Encode the moves of one own Pokemon in the current turn of the battle.
        Args:
            pokemon (pkmn_b.Pokemon): The Pokemon object containing the current state.
            battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded moves of the Pokemon.
        """
        # [is_available (1), base_power (1), category (3), type (18), effective_damage (2)] = 25 features per move = 100 for 4
        moves_tensors = []
        # If pokemon is ours, targets are opponents; if opponent, targets are ours
        if pokemon in battle.active_pokemon or pokemon in battle.team.values():
            targets = battle.opponent_active_pokemon
        else:
            targets = battle.active_pokemon

        target_1 = targets[0] if len(targets) > 0 and targets[0] and not targets[0].fainted else None
        target_2 = targets[1] if len(targets) > 1 and targets[1] and not targets[1].fainted else None
        
        for move in pokemon.moves.values():
            category = torch.zeros(3, dtype=torch.float32)
            category[constants.CATEGORY_TO_IDX[move.category]] = 1.0
            
            effective_type, _ = get_move_type_and_bp_boost(move, pokemon)
            move_type = torch.zeros(18, dtype=torch.float32)
            move_type[constants.TYPE_TO_IDX[effective_type]] = 1.0

            nominal_bp, effective_damage_1, effective_damage_2 = evaluate_move_power_and_multipliers(
                move, pokemon, target_1, target_2, battle
            )
            
            moves_tensor = torch.cat([
                torch.tensor([0.0]) if move.current_pp == 0 else torch.tensor([1.0]),
                torch.tensor([nominal_bp / 150.0]),
                category,
                move_type,
                torch.tensor([effective_damage_1, effective_damage_2], dtype=torch.float32)
            ])
            
            moves_tensors.append(moves_tensor)

        while len(moves_tensors) < 4:
            moves_tensors.append(torch.zeros(25, dtype=torch.float32))

        return torch.cat(moves_tensors)

    def encode_single_pokemon(self, pokemon: pkmn_b.Pokemon | None, battle: pkmn_b.DoubleBattle) -> torch.Tensor:
        """
        Generates the encoding vector for a Pokemon.
        Args:
            pokemon (pkmn_b.Pokemon): The Pokemon object to be encoded.
            battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        Returns:
            torch.Tensor: The encoded tensor for the Pokemon.
        """
        # [is_present, HP, types, status, boosts, moves, protect_success_rate, is_mega, base_stats] = 42 + 100 = 142 features    
        if not pokemon:
            return torch.zeros(142, dtype=torch.float32) # If the slot is empty
        
        type_tensor = torch.zeros(18, dtype=torch.float32)
        
        for pkmn_type in pokemon.types:
            type_tensor[constants.TYPE_TO_IDX[pkmn_type]] = 1.0 
        
        # [poison, toxic, burn, freeze, paralysis, sleep, none]
        status_tensor = torch.zeros(7, dtype=torch.float32)
        if pokemon.status:
            status_tensor[constants.STATUS_TO_IDX[pokemon.status]] = 1.0
        else:
            status_tensor[6] = 1.0  # No status condition
        
        boost_tensor = torch.zeros(7, dtype=torch.float32)
        for stat, boost in pokemon.boosts.items():
            boost_tensor[constants.BOOSTABLE_STAT_TO_IDX[stat]] = boost / 6.0  # Normalize to [-1, 1]
            
        has_protect = any(
            move.id in constants.PROTECT_MOVES 
            for move in pokemon.moves.values() 
            if move is not None
        )
        protect_success_rate = 0.0 if not has_protect else 1.0 / (3.0 ** pokemon.protect_counter)
            
        species = pokemon.species.lower().replace("-", "").replace(" ", "")
        is_mega = (species.endswith(("mega", "megax", "megay", "megaz")) and species != "yanmega") or species.endswith("primal")
        
        base_stats_tensor = torch.tensor(list(pokemon.base_stats.values()), dtype=torch.float32) / 150.0
        
        pkmn_tensor = torch.cat([
            torch.tensor([1.0, pokemon.current_hp_fraction]),
            type_tensor,
            status_tensor,
            boost_tensor,
            self.encode_moves(pokemon, battle),
            torch.tensor([protect_success_rate, 1.0 if is_mega else 0.0]),
            base_stats_tensor
        ])
        
        return pkmn_tensor

    def encode_bench(self, battle: pkmn_b.DoubleBattle) -> torch.Tensor:
        """
        Generates the encoding vector for the player's bench (non-active Pokémon).
        Args:
            battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        Returns:
            torch.Tensor: The encoded tensor for the player's bench.
        """
        bench_tensors = []
        bench = [
            pokemon for pokemon in battle.team.values() 
            if pokemon is not None 
            and not pokemon.fainted 
            and pokemon not in battle.active_pokemon 
            and pokemon.selected_in_teampreview
        ]
        
        for pokemon in bench:
            # type_tensor = torch.zeros(18, dtype=torch.float32)
            # for pkmn_type in pokemon.types:
            #     type_tensor[constants.TYPE_TO_IDX[pkmn_type]] = 1.0
            # # 20 features
            # bench_tensor = torch.cat([
            #     torch.tensor([1.0]) if not pokemon.fainted else torch.tensor([0.0]),
            #     torch.tensor([pokemon.current_hp_fraction]),
            #     type_tensor
            # ])
            bench_tensor = self.encode_single_pokemon(pokemon, battle=battle)
            bench_tensors.append(bench_tensor)
            
        while len(bench_tensors) < 2:
            bench_tensors.append(torch.zeros(142, dtype=torch.float32))

        return torch.cat(bench_tensors)
    
    def encode_opponent_bench(self, battle: pkmn_b.DoubleBattle) -> torch.Tensor:
        """
        Generates the encoding vector for the opponent's bench (non-active Pokémon).
        Args:
            battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        Returns:
            torch.Tensor: The encoded tensor for the opponent's bench.
        """
        bench_tensors = []
        bench = [
            pokemon for pokemon in battle.opponent_team.values() 
            if pokemon is not None 
            and not pokemon.fainted 
            and pokemon not in battle.opponent_active_pokemon 
            and pokemon.revealed
        ]
        
        for pokemon in bench:
            bench_tensor = self.encode_single_pokemon(pokemon, battle=battle)
            bench_tensors.append(bench_tensor)
            
        while len(bench_tensors) < 2:
            bench_tensors.append(torch.zeros(142, dtype=torch.float32))

        return torch.cat(bench_tensors)
    
    def encode_single_ability(self, pokemon: pkmn_b.Pokemon | None) -> torch.Tensor:
        """
        Encodes a single Pokemon's ability into a 215-dim vector.
        Args:
            pokemon (pkmn_b.Pokemon): The Pokemon object to be encoded.
        Returns:
            torch.Tensor: A tensor representing the encoded ability of the Pokemon.
        """
        vec = torch.zeros(215, dtype=torch.float32)
        if not pokemon or pokemon.fainted:
            return vec

        if pokemon.ability:
            clean_name = pokemon.ability.lower().replace("-", "").replace(" ", "")
            ability_idx = constants.ABILITY_TO_IDX.get(clean_name)
            if ability_idx is not None:
                vec[ability_idx] = 1.0
            return vec

        if pokemon.possible_abilities:
            valid_indices = [
                constants.ABILITY_TO_IDX[a.lower().replace("-", "").replace(" ", "")]
                for a in pokemon.possible_abilities
                if a.lower().replace("-", "").replace(" ", "") in constants.ABILITY_TO_IDX
            ]
            if valid_indices:
                prob = 1.0 / len(valid_indices)
                for a_idx in valid_indices:
                    vec[a_idx] = prob

        return vec

    def encode_abilities(self, battle: pkmn_b.DoubleBattle) -> torch.Tensor:
        """
        Encodes abilities of all active and bench Pokemon for both players into an 8x215 tensor in consistent order.
        Args:
            battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded abilities of all relevant Pokemon.
        """
        abilities_tensor = torch.zeros((8, 215), dtype=torch.float32)

        for i in range(2):
            p1_mon = battle.active_pokemon[i] if i < len(battle.active_pokemon) else None
            abilities_tensor[i] = self.encode_single_ability(p1_mon)

            p2_mon = battle.opponent_active_pokemon[i] if i < len(battle.opponent_active_pokemon) else None
            abilities_tensor[2 + i] = self.encode_single_ability(p2_mon)

        own_bench = [
            p for p in battle.team.values()
            if p is not None
            and not p.fainted
            and p not in battle.active_pokemon
            and p.selected_in_teampreview
        ]
        for i in range(2):
            mon = own_bench[i] if i < len(own_bench) else None
            abilities_tensor[4 + i] = self.encode_single_ability(mon)

        opp_bench = [
            p for p in battle.opponent_team.values()
            if p is not None
            and not p.fainted
            and p not in battle.opponent_active_pokemon
            and p.revealed
        ]
        for i in range(2):
            mon = opp_bench[i] if i < len(opp_bench) else None
            abilities_tensor[6 + i] = self.encode_single_ability(mon)

        return abilities_tensor
    
    def encode_single_item(self, pokemon: pkmn_b.Pokemon | None) -> torch.Tensor:
        """
        Encodes a single Pokemon's item.
        Args:
            pokemon (pkmn_b.Pokemon): The Pokemon object to be encoded.
        Returns:
            torch.Tensor: A tensor representing the encoded item of the Pokemon.
        """
        vec = torch.zeros(167, dtype=torch.float32)
        if not pokemon or pokemon.fainted:
            return vec
        
        if pokemon.item in ("none", ""):
            vec[constants.ITEM_TO_IDX.get("none")] = 1.0
            return vec

        if pokemon.item:
            clean_name = pokemon.item.lower().replace("-", "").replace(" ", "")
            item_idx = constants.ITEM_TO_IDX.get(clean_name)
            if item_idx is not None:
                vec[item_idx] = 1.0
            return vec

        species = pokemon.species.lower().replace("-", "").replace(" ", "")
        is_mega = (species.endswith(("mega", "megax", "megay", "megaz")) and species != "yanmega") or species.endswith("primal")
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
    
    def encode_items(self, battle: pkmn_b.DoubleBattle) -> torch.Tensor:
        """
        Encodes items of all active and bench Pokemon for both players into an 8x167 tensor in consistent order.
        Args:
            battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded items of all relevant Pokemon.
        """
        items_tensor = torch.zeros((8, 167), dtype=torch.float32)

        for i in range(2):
            p1_mon = battle.active_pokemon[i] if i < len(battle.active_pokemon) else None
            items_tensor[i] = self.encode_single_item(p1_mon)

            p2_mon = battle.opponent_active_pokemon[i] if i < len(battle.opponent_active_pokemon) else None
            items_tensor[2 + i] = self.encode_single_item(p2_mon)

        own_bench = [
            p for p in battle.team.values()
            if p is not None
            and not p.fainted
            and p not in battle.active_pokemon
            and p.selected_in_teampreview
        ]
        for i in range(2):
            mon = own_bench[i] if i < len(own_bench) else None
            items_tensor[4 + i] = self.encode_single_item(mon)

        opp_bench = [
            p for p in battle.opponent_team.values()
            if p is not None
            and not p.fainted
            and p not in battle.opponent_active_pokemon
            and p.revealed
        ]
        for i in range(2):
            mon = opp_bench[i] if i < len(opp_bench) else None
            items_tensor[6 + i] = self.encode_single_item(mon)

        return items_tensor
    
    def encode(self, battle: pkmn_b.DoubleBattle) -> dict[str, np.ndarray]:
        """
        Generates the complete encoding vector for the current state of the battle.
        Args:
            battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        Returns:
            dict[str, np.ndarray]: The complete encoded arrays for the battle state.
        """
        field_effects = torch.cat([
            self.encode_weather(battle),
            self.encode_terrain(battle),
            self.encode_speed_modifiers(battle)
        ])
        active_pokemon = torch.cat([self.encode_single_pokemon(pokemon, battle) for pokemon in battle.active_pokemon])
        opp_active_pokemon = torch.cat([self.encode_single_pokemon(pokemon, battle) for pokemon in battle.opponent_active_pokemon])
        bench = self.encode_bench(battle)
        opp_bench = self.encode_opponent_bench(battle)        
        return {
            "numeric": torch.cat([field_effects, active_pokemon, opp_active_pokemon, bench, opp_bench]).numpy(),
            "abilities": self.encode_abilities(battle).numpy(),
            "items": self.encode_items(battle).numpy()
        }
    