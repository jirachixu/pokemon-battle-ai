import poke_env.battle as pkmn_b
import torch

def encode_weather(battle: pkmn_b.Battle) -> torch.Tensor:
    """
    Encode the weather condition of the current turn of the battle.
    Args:
        battle (pkmn_b.Battle): The battle object containing the current state.
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

def encode_terrain(battle: pkmn_b.Battle) -> torch.Tensor:
    """
    Encode the terrain condition of the current turn of the battle.
    Args:
        battle (pkmn_b.Battle): The battle object containing the current state.
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

def encode_speed_modifiers(battle: pkmn_b.Battle) -> torch.Tensor:
    """
    Encode the speed modifiers of both players' active Pokémon in the current turn of the battle.
    Args:
        battle (pkmn_b.Battle): The battle object containing the current state.
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