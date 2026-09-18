import poke_env.battle as pkmn_b
import torch
from src import constants

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

    def encode_moves(self, pokemon: pkmn_b.Pokemon) -> torch.Tensor:
        """
        Encode the moves of a Pokémon in the current turn of the battle.
        Args:
            pokemon (pkmn_b.Pokemon): The Pokémon object containing the current state.
        Returns:
            torch.Tensor: A tensor representing the encoded moves of the Pokémon.
        """
        # [is_available (1), base_power (1), category (3), type (18)] = 23 features per move = 92 for 4
        moves_tensors = []
        for move in pokemon.moves.values():
            # [physical, special, status]
            category = torch.zeros(3, dtype=torch.float32)
            category[constants.CATEGORY_TO_IDX[move.category]] = 1.0
            
            move_type = torch.zeros(18, dtype=torch.float32)
            move_type[constants.TYPE_TO_IDX[move.type]] = 1.0
            
            moves_tensor = torch.cat([
                torch.tensor([0.0]) if move.current_pp == 0 else torch.tensor([1.0]),
                torch.tensor([move.base_power / 150.0]),
                category,
                move_type
            ])
            
            moves_tensors.append(moves_tensor)
        
        while len(moves_tensors) < 4:
            moves_tensors.append(torch.zeros(23, dtype=torch.float32))
        
        return torch.cat(moves_tensors)

    def encode_single_pokemon(self, pokemon: pkmn_b.Pokemon | None) -> torch.Tensor:
        """
        Generates the encoding vector for a Pokemon.
        Args:
            pokemon (pkmn_b.Pokemon): The Pokemon object to be encoded.
        Returns:
            torch.Tensor: The encoded tensor for the Pokemon.
        """
        # [is_present, HP, types, status, boosts, moves, protected_last, is_mega] = 36 + 92 = 128 features    
        if not pokemon:
            return torch.zeros(128, dtype=torch.float32) # If the slot is empty
        
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
        
        did_protect = 1.0 if pokemon.last_move and pokemon.last_move.id in constants.PROTECT_MOVES else 0.0
            
        species = pokemon.species.lower().replace("-", "").replace(" ", "")
        is_mega = (species.endswith(("mega", "megax", "megay", "megaz")) and species != "yanmega") or species.endswith("primal")
        
        pkmn_tensor = torch.cat([
            torch.tensor([1.0, pokemon.current_hp_fraction]),
            type_tensor,
            status_tensor,
            boost_tensor,
            self.encode_moves(pokemon),
            torch.tensor([did_protect, 1.0 if is_mega else 0.0])
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
            type_tensor = torch.zeros(18, dtype=torch.float32)
            for pkmn_type in pokemon.types:
                type_tensor[constants.TYPE_TO_IDX[pkmn_type]] = 1.0
            # 20 features
            bench_tensor = torch.cat([
                torch.tensor([1.0]) if not pokemon.fainted else torch.tensor([0.0]),
                torch.tensor([pokemon.current_hp_fraction]),
                type_tensor
            ])
            bench_tensors.append(bench_tensor)
            
        while len(bench_tensors) < 2:
            bench_tensors.append(torch.zeros(20, dtype=torch.float32))

        return torch.cat(bench_tensors)
    
    def encode(self, battle: pkmn_b.DoubleBattle) -> torch.Tensor:
        """
        Generates the complete encoding vector for the current state of the battle.
        Args:
            battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        Returns:
            torch.Tensor: The complete encoded tensor for the battle state.
        """
        field_effects = torch.cat([
            self.encode_weather(battle),
            self.encode_terrain(battle),
            self.encode_speed_modifiers(battle)
        ])
        active_pokemon = torch.cat([self.encode_single_pokemon(pokemon) for pokemon in battle.active_pokemon])
        opp_active_pokemon = torch.cat([self.encode_single_pokemon(pokemon) for pokemon in battle.opponent_active_pokemon])
        bench = self.encode_bench(battle)
        return torch.cat([field_effects, active_pokemon, opp_active_pokemon, bench])
    