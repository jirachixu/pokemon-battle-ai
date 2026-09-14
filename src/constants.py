import poke_env.battle as pkmn_b

CATEGORY_TO_IDX = {
    pkmn_b.MoveCategory.PHYSICAL: 0,
    pkmn_b.MoveCategory.SPECIAL: 1,
    pkmn_b.MoveCategory.STATUS: 2,
}

TYPE_TO_IDX = {
    pkmn_b.PokemonType.NORMAL: 0,
    pkmn_b.PokemonType.FIRE: 1,
    pkmn_b.PokemonType.WATER: 2,
    pkmn_b.PokemonType.ELECTRIC: 3,
    pkmn_b.PokemonType.GRASS: 4,
    pkmn_b.PokemonType.ICE: 5,
    pkmn_b.PokemonType.FIGHTING: 6,
    pkmn_b.PokemonType.POISON: 7,
    pkmn_b.PokemonType.GROUND: 8,
    pkmn_b.PokemonType.FLYING: 9,
    pkmn_b.PokemonType.PSYCHIC: 10,
    pkmn_b.PokemonType.BUG: 11,
    pkmn_b.PokemonType.ROCK: 12,
    pkmn_b.PokemonType.GHOST: 13,
    pkmn_b.PokemonType.DRAGON: 14,
    pkmn_b.PokemonType.DARK: 15,
    pkmn_b.PokemonType.STEEL: 16,
    pkmn_b.PokemonType.FAIRY: 17
}

STATUS_TO_IDX = {
    pkmn_b.Status.PSN: 0,
    pkmn_b.Status.TOX: 1,
    pkmn_b.Status.BRN: 2,
    pkmn_b.Status.FRZ: 3,
    pkmn_b.Status.PAR: 4,
    pkmn_b.Status.SLP: 5
}

BOOSTABLE_STAT_TO_IDX = {
    "atk": 0,
    "def": 1,
    "spa": 2,
    "spd": 3,
    "spe": 4,
    "accuracy": 5,
    "evasion": 6
}

PROTECT_MOVES = {
    "protect",
    "detect",
    "spikyshield",
    "banefulbunker",
    "kingsshield",
    "silktrap",
    "burningbulwark",
}