import math
from enum import Enum
from typing import Optional, Tuple, Union
import poke_env.battle as pkmn_b
import src.constants as constants
from src.simulation import SimulatedBattleState, SimulatedPokemon, SimulatedMove

BattleType = Union[pkmn_b.DoubleBattle, SimulatedBattleState]
PokemonType_ = Union[pkmn_b.Pokemon, SimulatedPokemon]
MoveType = Union[pkmn_b.Move, SimulatedMove]
PokemonTypeEnumOrStr = Union[pkmn_b.PokemonType, str]

_hook_installed = False

def init_battle_damage_tracking() -> None:
    """
    Installs a lightweight hook into AbstractBattle.parse_message to track
    direct attack damage hits for Rage Fist without mutating Pokemon.__slots__.
    """
    global _hook_installed
    if _hook_installed:
        return
    _orig_parse = pkmn_b.AbstractBattle.parse_message

    def _patched_parse(self, split_message: list[str]):
        if len(split_message) > 3 and split_message[1] == "-damage":
            # Filter out passive damage (weather, status, recoil, etc.)
            is_passive = any(chunk.startswith("[from]") for chunk in split_message[4:])
            if not is_passive:
                ident = split_message[2]
                if not hasattr(self, "_hit_counts"):
                    self._hit_counts = {}
                self._hit_counts[ident] = self._hit_counts.get(ident, 0) + 1
        _orig_parse(self, split_message)

    pkmn_b.AbstractBattle.parse_message = _patched_parse
    _hook_installed = True

init_battle_damage_tracking()

def clean_str(val: Optional[Union[str, Enum]]) -> str:
    """Normalize strings or enums for robust comparison."""
    if not val:
        return ""
    if isinstance(val, Enum):
        val = val.name
    return str(val).lower().replace(" ", "").replace("-", "").replace("_", "")


def _has_field(battle: BattleType, field_name: str) -> bool:
    fields = getattr(battle, "fields", {})
    if not fields:
        return False
    for f in fields:
        if clean_str(f) == field_name:
            return True
    return False

def _has_weather(battle: BattleType, weather_names: tuple[str, ...]) -> bool:
    weather = getattr(battle, "weather", {})
    if not weather:
        return False
    for w in weather:
        if clean_str(w) in weather_names:
            return True
    return False

def is_grounded(pokemon: Optional[PokemonType_], battle: BattleType) -> bool:
    """Checks if a Pokémon is grounded."""
    if pokemon is None:
        return True
    if _has_field(battle, "gravity"):
        return True
    if clean_str(getattr(pokemon, "item", None)) == "ironball":
        return True
    types = getattr(pokemon, "types", ())
    for t in types:
        if clean_str(t) == "flying":
            return False
    ability = clean_str(getattr(pokemon, "ability", None))
    if ability in ("levitate", "eelevate"):
        return False
    if clean_str(getattr(pokemon, "item", None)) == "airballoon":
        return False
    return True

def get_move_type_and_bp_boost(
    move: MoveType,
    attacker: PokemonType_,
    battle: Optional[BattleType] = None
) -> Tuple[PokemonTypeEnumOrStr, float]:
    """Resolves moves whose type is modified by abilities, weather, or terrain."""
    ability = clean_str(getattr(attacker, "ability", None))
    move_type = getattr(move, "type", "normal")
    move_id = clean_str(getattr(move, "id", ""))
    c_move_type = clean_str(move_type)
    bp_boost = 1.0

    is_enum = isinstance(move_type, pkmn_b.PokemonType)

    if battle is not None:
        if move_id == "weatherball":
            if _has_weather(battle, ("sunnyday", "sun", "desolateland")) or ability == "megasol":
                return (pkmn_b.PokemonType.FIRE if is_enum else "fire"), 1.0
            elif _has_weather(battle, ("raindance", "rain", "primordialsea")):
                return (pkmn_b.PokemonType.WATER if is_enum else "water"), 1.0
            elif _has_weather(battle, ("sandstorm", "sand")):
                return (pkmn_b.PokemonType.ROCK if is_enum else "rock"), 1.0
            elif _has_weather(battle, ("snowscape", "snow", "hail")):
                return (pkmn_b.PokemonType.ICE if is_enum else "ice"), 1.0
        elif move_id == "terrainpulse" and is_grounded(attacker, battle):
            if _has_field(battle, "electricterrain"):
                return (pkmn_b.PokemonType.ELECTRIC if is_enum else "electric"), 1.0
            elif _has_field(battle, "grassyterrain"):
                return (pkmn_b.PokemonType.GRASS if is_enum else "grass"), 1.0
            elif _has_field(battle, "psychicterrain"):
                return (pkmn_b.PokemonType.PSYCHIC if is_enum else "psychic"), 1.0
            elif _has_field(battle, "mistyterrain"):
                return (pkmn_b.PokemonType.FAIRY if is_enum else "fairy"), 1.0

    if c_move_type == "normal":
        if ability == "dragonize":
            return (pkmn_b.PokemonType.DRAGON if is_enum else "dragon"), 1.2
        elif ability == "pixilate":
            return (pkmn_b.PokemonType.FAIRY if is_enum else "fairy"), 1.2
        elif ability == "aerilate":
            return (pkmn_b.PokemonType.FLYING if is_enum else "flying"), 1.2
        elif ability == "refrigerate":
            return (pkmn_b.PokemonType.ICE if is_enum else "ice"), 1.2
        elif ability == "galvanize":
            return (pkmn_b.PokemonType.ELECTRIC if is_enum else "electric"), 1.2

    if ability == "liquidvoice" and "sound" in getattr(move, "flags", set()):
        return (pkmn_b.PokemonType.WATER if is_enum else "water"), 1.0

    return move_type, bp_boost

def get_dynamic_base_power(
    move: MoveType,
    attacker: PokemonType_,
    target: Optional[PokemonType_],
    battle: BattleType
) -> float:
    """
    Calculates exact dynamic base power for variable-power moves in Gen 9 VGC & Champions.
    Works for both user Pokémon and opponent revealed moves.
    """
    move_id = clean_str(getattr(move, "id", ""))
    raw_bp = float(getattr(move, "base_power", 0.0))

    # Attacker Current HP Scaling
    if move_id in ("waterspout", "eruption", "dragonenergy"):
        hp_frac = attacker.current_hp_fraction if getattr(attacker, "current_hp_fraction", None) is not None else 1.0
        return max(1.0, math.floor(150.0 * hp_frac))

    if move_id in ("flail", "reversal"):
        hp_frac = attacker.current_hp_fraction if getattr(attacker, "current_hp_fraction", None) is not None else 1.0
        if hp_frac <= 0.0417: return 200.0
        if hp_frac <= 0.1042: return 150.0
        if hp_frac <= 0.2083: return 100.0
        if hp_frac <= 0.3542: return 80.0
        if hp_frac <= 0.6875: return 40.0
        return 20.0

    # Target Current HP Scaling
    if move_id == "hardpress":
        if target is None or getattr(target, "current_hp_fraction", None) is None: return 100.0
        return max(1.0, math.floor(100.0 * target.current_hp_fraction))

    if move_id == "crushgrip":
        if target is None or getattr(target, "current_hp_fraction", None) is None: return 120.0
        return max(1.0, math.floor(120.0 * target.current_hp_fraction))

    if move_id == "brine":
        if target and getattr(target, "current_hp_fraction", None) is not None and target.current_hp_fraction <= 0.5:
            return 130.0
        return 65.0

    if move_id == "lastrespects":
        if isinstance(battle, pkmn_b.DoubleBattle):
            is_self = attacker in battle.team.values()
            team_mons = list(battle.team.values() if is_self else battle.opponent_team.values())
            fainted_allies = sum(1 for p in team_mons if p and p.fainted)
        else:
            is_self = attacker.side == 1 if isinstance(attacker, SimulatedPokemon) else True
            sim_team = battle.all_team_pokemon() if is_self else battle.all_opponent_pokemon()
            fainted_allies = sum(1 for p in sim_team if p and p.fainted)
        return 50.0 + (50.0 * fainted_allies)

    if move_id == "ragefist":
        hits = 0
        hit_counts = getattr(battle, "_hit_counts", None) or getattr(battle, "hit_counts", {})
        if hit_counts:
            attacker_species = clean_str(getattr(attacker, "species", ""))
            attacker_name = clean_str(getattr(attacker, "name", ""))
            for ident, count in hit_counts.items():
                c_ident = clean_str(ident)
                if attacker_species and attacker_species in c_ident:
                    hits = count
                    break
                if attacker_name and attacker_name in c_ident:
                    hits = count
                    break
        return min(350.0, 50.0 + (50.0 * hits))

    # Target Weight Tier
    if move_id in ("lowkick", "grassknot"):
        if target is None or getattr(target, "weight", None) is None:
            return 60.0
        w = target.weight
        if w < 10.0:  return 20.0
        if w < 25.0:  return 40.0
        if w < 50.0:  return 60.0
        if w < 100.0: return 80.0
        if w < 200.0: return 100.0
        return 120.0

    # Weight Ratio
    if move_id in ("heavyslam", "heatcrash"):
        att_w = getattr(attacker, "weight", None)
        tgt_w = getattr(target, "weight", None) if target else None
        if not att_w or not tgt_w:
            return 60.0
        ratio = att_w / max(tgt_w, 0.1)
        if ratio >= 5.0: return 120.0
        if ratio >= 4.0: return 100.0
        if ratio >= 3.0: return 80.0
        if ratio >= 2.0: return 60.0
        return 40.0

    # Status Condition Scaling
    if move_id == "facade":
        c_status = clean_str(getattr(attacker, "status", None))
        if c_status in ("brn", "par", "psn", "tox", "burn", "paralysis", "poison", "toxic"):
            return 140.0
        return 70.0

    if move_id in ("hex", "infernalparade"):
        base = 60.0 if move_id == "infernalparade" else 65.0
        return (base * 2.0) if (target and getattr(target, "status", None) is not None) else base

    if move_id in ("venoshock", "barbbarrage"):
        base = 60.0 if move_id == "barbbarrage" else 65.0
        tgt_status = clean_str(getattr(target, "status", None)) if target else ""
        return (base * 2.0) if (tgt_status in ("psn", "tox", "poison", "toxic")) else base

    # Item Interactions
    if move_id == "knockoff":
        if target is not None and clean_str(getattr(target, "item", None)) not in ("", "none", "unknownitem"):
            if not str(target.item).endswith("ite"):  # Mega stones cannot be knocked off
                return 97.5
        return 65.0

    if move_id == "acrobatics":
        if clean_str(getattr(attacker, "item", None)) in ("", "none"):
            return 110.0
        return 55.0

    # Terrain & Weather Field Attacks
    if move_id == "expandingforce":
        if _has_field(battle, "psychicterrain") and is_grounded(attacker, battle):
            return 120.0
        return 80.0

    if move_id == "risingvoltage":
        if _has_field(battle, "electricterrain") and target and is_grounded(target, battle):
            return 140.0
        return 70.0

    if move_id == "psyblade":
        if _has_field(battle, "electricterrain"):
            return 120.0
        return 80.0

    if move_id == "mistyexplosion":
        if _has_field(battle, "mistyterrain") and is_grounded(attacker, battle):
            return 150.0
        return 100.0

    if move_id == "weatherball":
        active_weather = (
            _has_weather(
                battle,
                ("sunnyday", "sun", "desolateland", "raindance", "rain", "primordialsea", "sandstorm", "sand", "snowscape", "snow", "hail")
            )
            or clean_str(getattr(attacker, "ability", None)) == "megasol"
        )
        return 100.0 if active_weather else 50.0

    if move_id == "terrainpulse":
        active_terrain = (
            _has_field(battle, "electricterrain") or
            _has_field(battle, "grassyterrain") or
            _has_field(battle, "mistyterrain") or
            _has_field(battle, "psychicterrain")
        )
        return 100.0 if (active_terrain and is_grounded(attacker, battle)) else 50.0

    # Stat Boost Accumulation
    if move_id in ("storedpower", "powertrip"):
        pos_boosts = sum(max(0, val) for val in getattr(attacker, "boosts", {}).values())
        return 20.0 + (20.0 * pos_boosts)

    return raw_bp

def get_spread_multiplier(
    move: MoveType,
    attacker: PokemonType_,
    target_1: Optional[PokemonType_],
    target_2: Optional[PokemonType_],
    battle: BattleType
) -> float:
    """Calculates the damage multiplier for spread moves."""
    c_category = clean_str(getattr(move, "category", ""))
    if c_category == "status":
        return 1.0

    is_spread = False
    target_val = clean_str(getattr(move, "target", ""))
    if target_val in ("alladjacentfoes", "alladjacent"):
        is_spread = True
    elif getattr(move, "is_spread", False):
        is_spread = True
    elif clean_str(getattr(move, "id", "")) == "expandingforce" and _has_field(battle, "psychicterrain") and is_grounded(attacker, battle):
        is_spread = True

    if not is_spread:
        return 1.0

    alive_targets = 0
    if target_1 is not None and not getattr(target_1, "fainted", False):
        alive_targets += 1
    if target_2 is not None and not getattr(target_2, "fainted", False):
        alive_targets += 1

    return 0.75 if alive_targets >= 2 else 1.0

def get_attacker_ability_multiplier(
    move: MoveType,
    raw_base_power: float,
    effective_move_type: PokemonTypeEnumOrStr,
    attacker: PokemonType_,
    battle: BattleType
) -> float:
    """Calculates offensive ability modifiers."""
    ability = clean_str(getattr(attacker, "ability", None))
    if not ability:
        return 1.0

    flags = getattr(move, "flags", set())
    mult = 1.0

    if ability == "toughclaws" and "contact" in flags:
        mult *= 1.3
    elif ability == "sharpness" and "slicing" in flags:
        mult *= 1.5
    elif ability == "strongjaw" and "bite" in flags:
        mult *= 1.5
    elif ability == "megalauncher" and "pulse" in flags:
        mult *= 1.5
    elif ability == "ironfist" and "punch" in flags:
        mult *= 1.2
    elif ability == "punkrock" and "sound" in flags:
        mult *= 1.3
    elif ability == "technician" and raw_base_power <= 60.0:
        mult *= 1.5
    elif ability == "sheerforce" and bool(getattr(move, "secondary", None)):
        mult *= 1.3
    elif ability == "reckless" and (getattr(move, "recoil", 0.0) > 0.0 or getattr(move, "has_crash_damage", False)):
        mult *= 1.2
    elif ability == "parentalbond":
        mult *= 1.25

    c_move_type = clean_str(effective_move_type)
    if ability == "firemane" and c_move_type == "fire":
        mult *= 1.5
    elif ability == "transistor" and c_move_type == "electric":
        mult *= 1.3
    elif ability == "dragonsmaw" and c_move_type == "dragon":
        mult *= 1.5
    elif ability in ("steelworker", "steelyspirit") and c_move_type == "steel":
        mult *= 1.5
    elif ability == "rockypayload" and c_move_type == "rock":
        mult *= 1.5
    elif ability == "waterbubble" and c_move_type == "water":
        mult *= 2.0
    elif ability == "sandforce" and _has_weather(battle, ("sandstorm", "sand")) and c_move_type in ("rock", "ground", "steel"):
        mult *= 1.3

    c_category = clean_str(getattr(move, "category", ""))
    is_physical = (c_category == "physical")
    is_special = (c_category == "special")

    c_status = clean_str(getattr(attacker, "status", None))
    if ability in ("hugepower", "purepower") and is_physical:
        mult *= 2.0
    elif ability == "gorillatactics" and is_physical:
        mult *= 1.5
    elif ability == "guts" and is_physical and bool(c_status):
        mult *= 1.5
    elif ability == "toxicboost" and is_physical and c_status in ("psn", "tox", "poison", "toxic"):
        mult *= 1.5
    elif ability == "flareboost" and is_special and c_status in ("brn", "burn"):
        mult *= 1.5
    elif ability == "supremeoverlord":
        if isinstance(battle, pkmn_b.DoubleBattle):
            is_self = attacker in battle.team.values()
            team_mons = list(battle.team.values() if is_self else battle.opponent_team.values())
            fainted_count = sum(1 for p in team_mons if p and p.fainted)
        else:
            is_self = attacker.side == 1 if isinstance(attacker, SimulatedPokemon) else True
            sim_team = battle.all_team_pokemon() if is_self else battle.all_opponent_pokemon()
            fainted_count = sum(1 for p in sim_team if p and p.fainted)
        mult *= (1.0 + (0.10 * fainted_count))

    is_sun = (
        _has_weather(battle, ("sunnyday", "sun", "desolateland")) or
        ability == "megasol"
    )
    if ability == "hadronengine" and is_special and _has_field(battle, "electricterrain"):
        mult *= 1.33
    elif ability == "orichalcumpulse" and is_physical and is_sun:
        mult *= 1.33
    elif ability == "solarpower" and is_special and is_sun:
        mult *= 1.5

    hp_frac = attacker.current_hp_fraction if getattr(attacker, "current_hp_fraction", None) is not None else 1.0
    if hp_frac <= 0.33:
        if ability == "blaze" and c_move_type == "fire":
            mult *= 1.5
        elif ability == "torrent" and c_move_type == "water":
            mult *= 1.5
        elif ability == "overgrow" and c_move_type == "grass":
            mult *= 1.5
        elif ability == "swarm" and c_move_type == "bug":
            mult *= 1.5

    return mult

def get_target_defense_multiplier(
    move: MoveType,
    effective_move_type: PokemonTypeEnumOrStr,
    attacker: PokemonType_,
    target: Optional[PokemonType_],
    battle: BattleType,
    raw_type_multiplier: float
) -> float:
    """Handles target immunities and defensive reductions."""
    if target is None:
        return 1.0

    raw_ability = clean_str(getattr(target, "ability", None))
    attacker_ability = clean_str(getattr(attacker, "ability", None))
    ignores_ability = attacker_ability in ("moldbreaker", "teravolt", "turboblaze")
    ability = (
        raw_ability
        if (not ignores_ability or raw_ability in ("fullmetalbody", "shadowshield", "prismarmor", "comatose"))
        else ""
    )
    flags = getattr(move, "flags", set())
    c_move_type = clean_str(effective_move_type)
    c_category = clean_str(getattr(move, "category", ""))

    if c_move_type == "ground":
        if ability == "eartheater":
            return 0.0
        if ability in ("levitate", "eelevate") and not is_grounded(target, battle):
            return 0.0
    elif c_move_type == "fire":
        if ability in ("flashfire", "wellbakedbody"):
            return 0.0
    elif c_move_type == "water":
        if ability in ("waterabsorb", "stormdrain", "dryskin"):
            return 0.0
    elif c_move_type == "electric":
        if ability in ("voltabsorb", "lightningrod", "motordrive"):
            return 0.0
    elif c_move_type == "grass":
        if ability == "sapsipper":
            return 0.0

    if ability == "bulletproof" and "bullet" in flags:
        return 0.0
    if ability == "soundproof" and "sound" in flags:
        return 0.0
    if ability == "goodasgold" and c_category == "status":
        return 0.0

    mult = 1.0
    hp_frac = getattr(target, "current_hp_fraction", 0.0) or 0.0

    if ability == "auraguard" and "contact" in flags:
        mult *= 0.5
    elif ability == "thickfat" and c_move_type in ("fire", "ice"):
        mult *= 0.5
    elif ability in ("heatproof", "waterbubble") and c_move_type == "fire":
        mult *= 0.5
    elif ability == "purifyingsalt" and c_move_type == "ghost":
        mult *= 0.5
    elif ability == "punkrock" and "sound" in flags:
        mult *= 0.5
    elif ability == "fluffy":
        if "contact" in flags and clean_str(getattr(attacker, "item", None)) != "protectivepads":
            mult *= 0.5
        if c_move_type == "fire":
            mult *= 2.0
    elif ability == "furcoat" and c_category == "physical":
        mult *= 0.5
    elif ability == "icescales" and c_category == "special":
        mult *= 0.5

    if ability in ("multiscale", "shadowshield") and hp_frac >= 0.999:
        mult *= 0.5

    if ability in ("solidrock", "filter", "prismarmor") and raw_type_multiplier > 1.0:
        mult *= 0.75

    return mult

def get_held_item_multiplier(
    move: MoveType,
    effective_move_type: PokemonTypeEnumOrStr,
    attacker: PokemonType_,
    target: Optional[PokemonType_],
    raw_type_multiplier: float
) -> float:
    """Calculates damage modifiers from held items for both attacker and target."""
    c_category = clean_str(getattr(move, "category", ""))
    if c_category == "status":
        return 1.0
    mult = 1.0
    attacker_item = clean_str(str(getattr(attacker, "item", "") or ""))
    flags = getattr(move, "flags", set())
    is_physical = (c_category == "physical")
    is_special = (c_category == "special")
    c_move_type = clean_str(effective_move_type)

    if attacker_item == "choiceband" and is_physical:
        mult *= 1.5
    elif attacker_item == "choicespecs" and is_special:
        mult *= 1.5
    elif attacker_item == "lifeorb":
        mult *= 1.3
    elif attacker_item == "expertbelt" and raw_type_multiplier > 1.0:
        mult *= 1.2
    elif clean_str(constants.TYPE_BOOSTING_ITEMS.get(attacker_item)) == c_move_type:
        mult *= 1.2
    elif attacker_item in ("adamantcrystal", "adamantorb") and c_move_type in ("steel", "dragon"):
        mult *= 1.2
    elif attacker_item in ("lustrousglobe", "lustrousorb") and c_move_type in ("water", "dragon"):
        mult *= 1.2
    elif attacker_item in ("griseouscore", "griseousorb") and c_move_type in ("ghost", "dragon"):
        mult *= 1.2
    elif attacker_item == "souldew" and c_move_type in ("psychic", "dragon"):
        mult *= 1.2
    elif attacker_item == "punchingglove" and "punch" in flags:
        mult *= 1.1
    elif attacker_item == "muscleband" and is_physical:
        mult *= 1.1
    elif attacker_item == "wiseglasses" and is_special:
        mult *= 1.1

    if target is not None:
        target_item = clean_str(str(getattr(target, "item", "") or ""))
        resist_type = constants.TYPE_RESIST_BERRIES.get(target_item)
        if resist_type and clean_str(resist_type) == c_move_type:
            if raw_type_multiplier > 1.0 or target_item == "chilanberry":
                mult *= 0.5
        elif target_item == "assaultvest" and is_special:
            mult *= (2.0 / 3.0)
        elif target_item == "eviolite":
            mult *= (2.0 / 3.0)

    return mult

def get_effective_damage_multiplier(
    move: MoveType,
    raw_base_power: float,
    attacker: PokemonType_,
    target: Optional[PokemonType_],
    battle: BattleType
) -> float:
    """
    Computes complete damage multiplier across:
    (BP-Boost * Attacker-Ability * STAB * Weather * Terrain * Type-Matchup * Tinted-Lens * Target-Defense)
    """
    c_category = clean_str(getattr(move, "category", ""))
    if c_category == "status" or raw_base_power == 0.0 or target is None or getattr(target, "fainted", False):
        return 0.0

    move_type, bp_boost = get_move_type_and_bp_boost(move, attacker, battle)
    mult = bp_boost

    mult *= get_attacker_ability_multiplier(move, raw_base_power, move_type, attacker, battle)

    c_move_type = clean_str(move_type)
    att_types = [clean_str(t) for t in getattr(attacker, "types", ())]
    if c_move_type in att_types:
        ability = clean_str(getattr(attacker, "ability", None))
        mult *= 2.0 if ability == "adaptability" else 1.5

    attacker_ability = clean_str(getattr(attacker, "ability", None))
    move_id = clean_str(getattr(move, "id", ""))
    is_sun = (
        _has_weather(battle, ("sunnyday", "sun", "desolateland")) or
        attacker_ability == "megasol"
    )
    is_rain = _has_weather(battle, ("raindance", "rain", "primordialsea")) and attacker_ability != "megasol"
    is_bad_solar_weather = (
        _has_weather(battle, ("raindance", "rain", "primordialsea", "sandstorm", "sand", "snowscape", "snow", "hail"))
        and attacker_ability != "megasol"
    )
    if move_id in ("solarbeam", "solarblade") and is_bad_solar_weather:
        mult *= 0.5

    if c_move_type == "fire":
        if is_sun:
            mult *= 1.5
        elif is_rain:
            mult *= 0.5
    elif c_move_type == "water":
        if move_id == "hydrosteam" and is_sun:
            mult *= 1.5
        elif is_rain:
            mult *= 1.5
        elif is_sun:
            mult *= 0.5

    if is_grounded(attacker, battle):
        if c_move_type == "grass" and _has_field(battle, "grassyterrain"):
            mult *= 1.3
        elif c_move_type == "electric" and _has_field(battle, "electricterrain"):
            mult *= 1.3
        elif c_move_type == "psychic" and _has_field(battle, "psychicterrain"):
            mult *= 1.3

    if is_grounded(target, battle):
        if c_move_type == "dragon" and _has_field(battle, "mistyterrain"):
            mult *= 0.5
        elif clean_str(getattr(move, "id", "")) in ("earthquake", "bulldoze", "magnitude") and _has_field(battle, "grassyterrain"):
            mult *= 0.5

    if isinstance(target, pkmn_b.Pokemon):
        if isinstance(move_type, pkmn_b.PokemonType):
            type_mult = target.damage_multiplier(move_type)
        elif isinstance(move, pkmn_b.Move):
            type_mult = target.damage_multiplier(move)
        else:
            type_mult = target.damage_multiplier(pkmn_b.PokemonType.from_name(clean_str(move_type)))
    else:
        type_mult = target.damage_multiplier(move_type)

    if type_mult == 0.0 and attacker_ability in ("scrappy", "mindseye") and c_move_type in ("normal", "fighting"):
        type_mult = 1.0

    ignores_ability = attacker_ability in ("moldbreaker", "teravolt", "turboblaze")
    target_ability = clean_str(getattr(target, "ability", None))
    if not ignores_ability and target_ability == "terashell" and (getattr(target, "current_hp_fraction", 0.0) or 0.0) >= 0.999 and type_mult > 0.0:
        type_mult = 0.5

    if attacker_ability == "tintedlens" and 0.0 < type_mult < 1.0:
        type_mult *= 2.0

    # Field-wide Aura & Ruin Abilities (Fairy Aura, Dark Aura, Aura Break, Tablets/Sword/Vessel/Beads of Ruin)
    active_mons = [
        p for p in (getattr(battle, "active_pokemon", []) + getattr(battle, "opponent_active_pokemon", []))
        if p is not None and not getattr(p, "fainted", False)
    ]
    active_abilities = [clean_str(getattr(p, "ability", None)) for p in active_mons]
    has_aura_break = "aurabreak" in active_abilities
    if c_move_type == "fairy" and "fairyaura" in active_abilities:
        mult *= 0.75 if has_aura_break else (4.0 / 3.0)
    elif c_move_type == "dark" and "darkaura" in active_abilities:
        mult *= 0.75 if has_aura_break else (4.0 / 3.0)

    is_special = (c_category == "special")
    target_raw_ability = clean_str(getattr(target, "ability", None))
    if not is_special:
        if attacker_ability != "tabletsofruin" and "tabletsofruin" in active_abilities:
            mult *= 0.75
        if target_raw_ability != "swordofruin" and "swordofruin" in active_abilities:
            mult *= (4.0 / 3.0)
    else:
        if attacker_ability != "vesselofruin" and "vesselofruin" in active_abilities:
            mult *= 0.75
        if target_raw_ability != "beadsofruin" and "beadsofruin" in active_abilities:
            mult *= (4.0 / 3.0)

    mult *= type_mult

    mult *= get_target_defense_multiplier(move, move_type, attacker, target, battle, type_mult)
    mult *= get_held_item_multiplier(move, move_type, attacker, target, type_mult)
    
    target_is_opponent = target not in battle.active_pokemon
    side_conditions = battle.opponent_side_conditions if target_is_opponent else battle.side_conditions
    if not attacker_ability == "infiltrator":
        if isinstance(battle, pkmn_b.DoubleBattle):
            if c_category == "physical" and pkmn_b.SideCondition.REFLECT in side_conditions or pkmn_b.SideCondition.AURORA_VEIL in side_conditions:
                mult *= (2 / 3)
            elif c_category == "special" and pkmn_b.SideCondition.LIGHT_SCREEN in side_conditions or pkmn_b.SideCondition.AURORA_VEIL in side_conditions:
                mult *= (2 / 3)
        elif isinstance(battle, SimulatedBattleState):
            if c_category == "physical" and "reflect" in side_conditions or "auroraveil" in side_conditions:
                mult *= (2 / 3)
            elif c_category == "special" and "lightscreen" in side_conditions or "auroraveil" in side_conditions:
                mult *= (2 / 3)

    return mult

def evaluate_move_power_and_multipliers(
    move: MoveType,
    attacker: PokemonType_,
    target_1: Optional[PokemonType_],
    target_2: Optional[PokemonType_],
    battle: BattleType
) -> Tuple[float, float, float]:
    """
    Evaluates move against Target 1 and Target 2 with exact target-dependent base powers
    and spread move 0.75x penalty.
    Returns:
        nominal_base_power (float): The base power of the move without target-specific adjustments.
        effective_damage_1 (float): The effective damage multiplier against Target 1.
        effective_damage_2 (float): The effective damage multiplier against Target 2.
    """

    bp_1 = get_dynamic_base_power(move, attacker, target_1, battle)
    bp_2 = get_dynamic_base_power(move, attacker, target_2, battle)
    nominal_bp = float(getattr(move, "base_power", 0.0))

    spread_penalty = get_spread_multiplier(move, attacker, target_1, target_2, battle)

    mult_1 = get_effective_damage_multiplier(move, bp_1, attacker, target_1, battle) * spread_penalty
    mult_2 = get_effective_damage_multiplier(move, bp_2, attacker, target_2, battle) * spread_penalty

    eff_damage_1 = (bp_1 * mult_1) / 300.0
    eff_damage_2 = (bp_2 * mult_2) / 300.0

    return nominal_bp, eff_damage_1, eff_damage_2
