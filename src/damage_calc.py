import math
from typing import Optional, Tuple
import poke_env.battle as pkmn_b
import src.constants as constants

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

def clean_str(val: Optional[str]) -> str:
    """Normalize strings for robust comparison."""
    if not val:
        return ""
    return str(val).lower().replace(" ", "").replace("-", "").replace("_", "")

def is_grounded(pokemon: Optional[pkmn_b.Pokemon], battle: pkmn_b.DoubleBattle) -> bool:
    """Checks if a Pokémon is grounded."""
    if pokemon is None:
        return True
    if pkmn_b.Field.GRAVITY in battle.fields:
        return True
    if clean_str(getattr(pokemon, "item", None)) == "ironball":
        return True
    if pkmn_b.PokemonType.FLYING in getattr(pokemon, "types", ()):
        return False
    ability = clean_str(getattr(pokemon, "ability", None))
    if ability in ("levitate", "eelevate"):
        return False
    if clean_str(getattr(pokemon, "item", None)) == "airballoon":
        return False
    return True

def get_move_type_and_bp_boost(
    move: pkmn_b.Move,
    attacker: pkmn_b.Pokemon
) -> Tuple[pkmn_b.PokemonType, float]:
    """Resolves moves whose type is modified by abilities."""
    ability = clean_str(getattr(attacker, "ability", None))
    move_type = move.type
    bp_boost = 1.0

    if move_type == pkmn_b.PokemonType.NORMAL:
        if ability == "dragonize":
            return pkmn_b.PokemonType.DRAGON, 1.2
        elif ability == "pixilate":
            return pkmn_b.PokemonType.FAIRY, 1.2
        elif ability == "aerilate":
            return pkmn_b.PokemonType.FLYING, 1.2
        elif ability == "refrigerate":
            return pkmn_b.PokemonType.ICE, 1.2
        elif ability == "galvanize":
            return pkmn_b.PokemonType.ELECTRIC, 1.2

    if ability == "liquidvoice" and "sound" in getattr(move, "flags", set()):
        return pkmn_b.PokemonType.WATER, 1.0

    return move_type, bp_boost

def get_dynamic_base_power(
    move: pkmn_b.Move,
    attacker: pkmn_b.Pokemon,
    target: Optional[pkmn_b.Pokemon],
    battle: pkmn_b.DoubleBattle
) -> float:
    """
    Calculates exact dynamic base power for variable-power moves in Gen 9 VGC & Champions.
    Works for both user Pokémon and opponent revealed moves.
    """
    move_id = move.id
    raw_bp = float(move.base_power)

    # Attacker Current HP Scaling
    if move_id in ("waterspout", "eruption", "dragonenergy"):
        hp_frac = attacker.current_hp_fraction if attacker.current_hp_fraction is not None else 1.0
        return max(1.0, math.floor(150.0 * hp_frac))

    if move_id in ("flail", "reversal"):
        hp_frac = attacker.current_hp_fraction if attacker.current_hp_fraction is not None else 1.0
        if hp_frac <= 0.0417: return 200.0
        if hp_frac <= 0.1042: return 150.0
        if hp_frac <= 0.2083: return 100.0
        if hp_frac <= 0.3542: return 80.0
        if hp_frac <= 0.6875: return 40.0
        return 20.0

    # Target Current HP Scaling
    if move_id == "hardpress":
        if target is None or target.current_hp_fraction is None: return 100.0
        return max(1.0, math.floor(100.0 * target.current_hp_fraction))

    if move_id == "crushgrip":
        if target is None or target.current_hp_fraction is None: return 120.0
        return max(1.0, math.floor(120.0 * target.current_hp_fraction))

    if move_id == "brine":
        if target and target.current_hp_fraction is not None and target.current_hp_fraction <= 0.5:
            return 130.0
        return 65.0

    if move_id == "lastrespects":
        is_self = attacker in battle.team.values()
        team = battle.team.values() if is_self else battle.opponent_team.values()
        fainted_allies = sum(1 for p in team if p and p.fainted)
        return 50.0 + (50.0 * fainted_allies)

    if move_id == "ragefist":
        hits = 0
        if hasattr(battle, "_hit_counts"):
            attacker_species = clean_str(attacker.species)
            attacker_name = clean_str(attacker.name)
            hit_counts = getattr(battle, "_hit_counts", {})
            for ident, count in hit_counts.items():
                c_ident = clean_str(ident)
                if attacker_species in c_ident or attacker_name in c_ident:
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
        if attacker.status in (pkmn_b.Status.BRN, pkmn_b.Status.PAR, pkmn_b.Status.PSN, pkmn_b.Status.TOX):
            return 140.0
        return 70.0

    if move_id in ("hex", "infernalparade"):
        base = 60.0 if move_id == "infernalparade" else 65.0
        return (base * 2.0) if (target and target.status is not None) else base

    if move_id in ("venoshock", "barbbarrage"):
        base = 60.0 if move_id == "barbbarrage" else 65.0
        return (base * 2.0) if (target and target.status in (pkmn_b.Status.PSN, pkmn_b.Status.TOX)) else base

    # Item Interactions
    if move_id == "knockoff":
        if target is not None and getattr(target, "item", None) not in (None, "none", "unknown_item"):
            if not str(target.item).endswith("ite"):  # Mega stones cannot be knocked off
                return 97.5
        return 65.0

    if move_id == "acrobatics":
        if getattr(attacker, "item", None) in (None, "none"):
            return 110.0
        return 55.0

    # Terrain & Weather Field Attacks
    if move_id == "expandingforce":
        if pkmn_b.Field.PSYCHIC_TERRAIN in battle.fields and is_grounded(attacker, battle):
            return 120.0
        return 80.0

    if move_id == "risingvoltage":
        if pkmn_b.Field.ELECTRIC_TERRAIN in battle.fields and target and is_grounded(target, battle):
            return 140.0
        return 70.0

    if move_id == "psyblade":
        if pkmn_b.Field.ELECTRIC_TERRAIN in battle.fields and is_grounded(attacker, battle):
            return 120.0
        return 80.0

    if move_id == "mistyexplosion":
        if pkmn_b.Field.MISTY_TERRAIN in battle.fields and is_grounded(attacker, battle):
            return 150.0
        return 100.0

    if move_id == "weatherball":
        active_weather = (
            pkmn_b.Weather.SUNNYDAY in battle.weather or pkmn_b.Weather.DESOLATELAND in battle.weather or
            pkmn_b.Weather.RAINDANCE in battle.weather or pkmn_b.Weather.PRIMORDIALSEA in battle.weather or
            pkmn_b.Weather.SANDSTORM in battle.weather or pkmn_b.Weather.SNOWSCAPE in battle.weather or
            pkmn_b.Weather.HAIL in battle.weather
        )
        return 100.0 if active_weather else 50.0

    if move_id == "terrainpulse":
        active_terrain = (
            pkmn_b.Field.ELECTRIC_TERRAIN in battle.fields or
            pkmn_b.Field.GRASSY_TERRAIN in battle.fields or
            pkmn_b.Field.MISTY_TERRAIN in battle.fields or
            pkmn_b.Field.PSYCHIC_TERRAIN in battle.fields
        )
        return 100.0 if (active_terrain and is_grounded(attacker, battle)) else 50.0

    # Stat Boost Accumulation
    if move_id in ("storedpower", "powertrip"):
        pos_boosts = sum(max(0, val) for val in attacker.boosts.values())
        return 20.0 + (20.0 * pos_boosts)

    return raw_bp

def get_spread_multiplier(
    move: pkmn_b.Move,
    attacker: pkmn_b.Pokemon,
    target_1: Optional[pkmn_b.Pokemon],
    target_2: Optional[pkmn_b.Pokemon],
    battle: pkmn_b.DoubleBattle
) -> float:
    """Calculates the damage multiplier for spread moves."""
    if move.category == pkmn_b.MoveCategory.STATUS:
        return 1.0

    is_spread = False
    if move.target in (pkmn_b.Target.ALL_ADJACENT_FOES, pkmn_b.Target.ALL_ADJACENT):
        is_spread = True
    elif move.id == "expandingforce" and pkmn_b.Field.PSYCHIC_TERRAIN in battle.fields and is_grounded(attacker, battle):
        is_spread = True

    if not is_spread:
        return 1.0

    alive_targets = 0
    if target_1 is not None and not target_1.fainted:
        alive_targets += 1
    if target_2 is not None and not target_2.fainted:
        alive_targets += 1

    return 0.75 if alive_targets >= 2 else 1.0

def get_attacker_ability_multiplier(
    move: pkmn_b.Move,
    raw_base_power: float,
    effective_move_type: pkmn_b.PokemonType,
    attacker: pkmn_b.Pokemon,
    battle: pkmn_b.DoubleBattle
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
    elif ability == "reckless" and getattr(move, "recoil", 0.0) > 0.0:
        mult *= 1.2
    elif ability == "parentalbond":
        mult *= 1.25

    if ability == "firemane" and effective_move_type == pkmn_b.PokemonType.FIRE:
        mult *= 1.5
    elif ability == "transistor" and effective_move_type == pkmn_b.PokemonType.ELECTRIC:
        mult *= 1.3
    elif ability == "dragonsmaw" and effective_move_type == pkmn_b.PokemonType.DRAGON:
        mult *= 1.5
    elif ability in ("steelworker", "steelyspirit") and effective_move_type == pkmn_b.PokemonType.STEEL:
        mult *= 1.5
    elif ability == "rockypayload" and effective_move_type == pkmn_b.PokemonType.ROCK:
        mult *= 1.5
    elif ability == "waterbubble" and effective_move_type == pkmn_b.PokemonType.WATER:
        mult *= 2.0

    is_physical = (move.category == pkmn_b.MoveCategory.PHYSICAL)
    is_special = (move.category == pkmn_b.MoveCategory.SPECIAL)

    if ability in ("hugepower", "purepower") and is_physical:
        mult *= 2.0
    elif ability == "gorillatactics" and is_physical:
        mult *= 1.5
    elif ability == "guts" and is_physical and attacker.status is not None:
        mult *= 1.5
    elif ability == "toxicboost" and is_physical and attacker.status in (pkmn_b.Status.PSN, pkmn_b.Status.TOX):
        mult *= 1.5
    elif ability == "flareboost" and is_special and attacker.status == pkmn_b.Status.BRN:
        mult *= 1.5
    elif ability == "supremeoverlord":
        is_self = attacker in battle.team.values()
        team = battle.team.values() if is_self else battle.opponent_team.values()
        fainted_count = sum(1 for p in team if p and p.fainted)
        mult *= (1.0 + (0.10 * fainted_count))

    is_sun = (
        pkmn_b.Weather.SUNNYDAY in battle.weather or
        pkmn_b.Weather.DESOLATELAND in battle.weather or
        ability == "megasol"
    )
    if ability == "hadronengine" and is_special and pkmn_b.Field.ELECTRIC_TERRAIN in battle.fields:
        mult *= 1.33
    elif ability == "orichalcumpulse" and is_physical and is_sun:
        mult *= 1.33
    elif ability == "solarpower" and is_special and is_sun:
        mult *= 1.5

    hp_frac = attacker.current_hp_fraction if attacker.current_hp_fraction is not None else 1.0
    if hp_frac <= 0.33:
        if ability == "blaze" and effective_move_type == pkmn_b.PokemonType.FIRE:
            mult *= 1.5
        elif ability == "torrent" and effective_move_type == pkmn_b.PokemonType.WATER:
            mult *= 1.5
        elif ability == "overgrow" and effective_move_type == pkmn_b.PokemonType.GRASS:
            mult *= 1.5
        elif ability == "swarm" and effective_move_type == pkmn_b.PokemonType.BUG:
            mult *= 1.5

    return mult

def get_target_defense_multiplier(
    move: pkmn_b.Move,
    effective_move_type: pkmn_b.PokemonType,
    attacker: pkmn_b.Pokemon,
    target: Optional[pkmn_b.Pokemon],
    battle: pkmn_b.DoubleBattle,
    raw_type_multiplier: float
) -> float:
    """Handles target immunities and defensive reductions."""
    if target is None:
        return 1.0

    ability = clean_str(getattr(target, "ability", None))
    attacker_ability = clean_str(getattr(attacker, "ability", None))
    flags = getattr(move, "flags", set())

    if effective_move_type == pkmn_b.PokemonType.GROUND:
        if ability in ("levitate", "eelevate", "eartheater") and not is_grounded(target, battle):
            return 0.0
    elif effective_move_type == pkmn_b.PokemonType.FIRE:
        if ability in ("flashfire", "wellbakedbody"):
            return 0.0
    elif effective_move_type == pkmn_b.PokemonType.WATER:
        if ability in ("waterabsorb", "stormdrain", "dryskin"):
            return 0.0
    elif effective_move_type == pkmn_b.PokemonType.ELECTRIC:
        if ability in ("voltabsorb", "lightningrod", "motordrive"):
            return 0.0
    elif effective_move_type == pkmn_b.PokemonType.GRASS:
        if ability == "sapsipper":
            return 0.0

    if ability == "bulletproof" and "bullet" in flags:
        return 0.0
    if ability == "soundproof" and "sound" in flags:
        return 0.0
    if ability == "goodasgold" and move.category == pkmn_b.MoveCategory.STATUS:
        return 0.0

    mult = 1.0

    # Scrappy
    if attacker_ability == "scrappy" and raw_type_multiplier == 0.0:
        if effective_move_type in (pkmn_b.PokemonType.NORMAL, pkmn_b.PokemonType.FIGHTING):
            mult *= 1.0

    if ability == "auraguard" and "contact" in flags:
        mult *= 0.5
    elif ability == "thickfat" and effective_move_type in (pkmn_b.PokemonType.FIRE, pkmn_b.PokemonType.ICE):
        mult *= 0.5
    elif ability == "heatproof" and effective_move_type == pkmn_b.PokemonType.FIRE:
        mult *= 0.5
    elif ability == "purifyingsalt" and effective_move_type == pkmn_b.PokemonType.GHOST:
        mult *= 0.5
    elif ability == "fluffy":
        if "contact" in flags:
            mult *= 0.5
        if effective_move_type == pkmn_b.PokemonType.FIRE:
            mult *= 2.0
    elif ability == "furcoat" and move.category == pkmn_b.MoveCategory.PHYSICAL:
        mult *= 0.5
    elif ability == "icescales" and move.category == pkmn_b.MoveCategory.SPECIAL:
        mult *= 0.5
    elif ability in ("multiscale", "shadowshield") and target.current_hp_fraction == 1.0:
        mult *= 0.5

    if ability in ("solidrock", "filter", "prismarmor") and raw_type_multiplier > 1.0:
        mult *= 0.75

    return mult

def get_held_item_multiplier(
    move: pkmn_b.Move,
    effective_move_type: pkmn_b.PokemonType,
    attacker: pkmn_b.Pokemon,
    target: pkmn_b.Pokemon | None,
    raw_type_multiplier: float
) -> float:
    """Calculates damage modifiers from held items for both attacker and target."""
    if move.category == pkmn_b.MoveCategory.STATUS:
        return 1.0
    mult = 1.0
    attacker_item = clean_str(str(getattr(attacker, "item", "") or ""))
    flags = getattr(move, "flags", set())
    is_physical = (move.category == pkmn_b.MoveCategory.PHYSICAL)
    is_special = (move.category == pkmn_b.MoveCategory.SPECIAL)

    if attacker_item == "choiceband" and is_physical:
        mult *= 1.5
    elif attacker_item == "choicespecs" and is_special:
        mult *= 1.5
    elif attacker_item == "lifeorb":
        mult *= 1.3
    elif attacker_item == "expertbelt" and raw_type_multiplier > 1.0:
        mult *= 1.2
    elif constants.TYPE_BOOSTING_ITEMS.get(attacker_item) == effective_move_type:
        mult *= 1.2
    elif attacker_item in ("adamantcrystal", "adamantorb") and effective_move_type in (pkmn_b.PokemonType.STEEL, pkmn_b.PokemonType.DRAGON):
        mult *= 1.2
    elif attacker_item in ("lustrousglobe", "lustrousorb") and effective_move_type in (pkmn_b.PokemonType.WATER, pkmn_b.PokemonType.DRAGON):
        mult *= 1.2
    elif attacker_item in ("griseouscore", "griseousorb") and effective_move_type in (pkmn_b.PokemonType.GHOST, pkmn_b.PokemonType.DRAGON):
        mult *= 1.2
    elif attacker_item == "souldew" and effective_move_type in (pkmn_b.PokemonType.PSYCHIC, pkmn_b.PokemonType.DRAGON):
        mult *= 1.2
    elif attacker_item == "punchingglove" and "punch" in flags:
        mult *= 1.1
    elif attacker_item == "muscleband" and is_physical:
        mult *= 1.1
    elif attacker_item == "wiseglasses" and is_special:
        mult *= 1.1

    if target is not None:
        target_item = clean_str(str(getattr(target, "item", "") or ""))
        if constants.TYPE_RESIST_BERRIES.get(target_item) == effective_move_type:
            if raw_type_multiplier > 1.0 or target_item == "chilanberry":
                mult *= 0.5
        elif target_item == "assaultvest" and is_special:
            mult *= (2.0 / 3.0)
        elif target_item == "eviolite":
            mult *= (2.0 / 3.0)

    return mult

def get_effective_damage_multiplier(
    move: pkmn_b.Move,
    raw_base_power: float,
    attacker: pkmn_b.Pokemon,
    target: Optional[pkmn_b.Pokemon],
    battle: pkmn_b.DoubleBattle
) -> float:
    """
    Computes complete damage multiplier across:
    (BP-Boost * Attacker-Ability * STAB * Weather * Terrain * Type-Matchup * Tinted-Lens * Target-Defense)
    """
    if move.category == pkmn_b.MoveCategory.STATUS or raw_base_power == 0.0 or target is None or target.fainted:
        return 0.0

    move_type, bp_boost = get_move_type_and_bp_boost(move, attacker)
    mult = bp_boost

    mult *= get_attacker_ability_multiplier(move, raw_base_power, move_type, attacker, battle)

    if move_type in getattr(attacker, "types", ()):
        ability = clean_str(getattr(attacker, "ability", None))
        mult *= 2.0 if ability == "adaptability" else 1.5

    attacker_ability = clean_str(getattr(attacker, "ability", None))
    is_sun = (
        pkmn_b.Weather.SUNNYDAY in battle.weather or
        pkmn_b.Weather.DESOLATELAND in battle.weather or
        attacker_ability == "megasol"
    )
    is_rain = (
        pkmn_b.Weather.RAINDANCE in battle.weather or
        pkmn_b.Weather.PRIMORDIALSEA in battle.weather
    )

    if move_type == pkmn_b.PokemonType.FIRE:
        if is_sun:
            mult *= 1.5
        elif is_rain:
            mult *= 0.5
    elif move_type == pkmn_b.PokemonType.WATER:
        if is_rain:
            mult *= 1.5
        elif is_sun:
            mult *= 0.5

    if is_grounded(attacker, battle):
        if move_type == pkmn_b.PokemonType.GRASS and pkmn_b.Field.GRASSY_TERRAIN in battle.fields:
            mult *= 1.3
        elif move_type == pkmn_b.PokemonType.ELECTRIC and pkmn_b.Field.ELECTRIC_TERRAIN in battle.fields:
            mult *= 1.3
        elif move_type == pkmn_b.PokemonType.PSYCHIC and pkmn_b.Field.PSYCHIC_TERRAIN in battle.fields:
            mult *= 1.3

    type_mult = target.damage_multiplier(move_type)

    if attacker_ability == "tintedlens" and 0.0 < type_mult < 1.0:
        type_mult *= 2.0

    mult *= type_mult

    mult *= get_target_defense_multiplier(move, move_type, attacker, target, battle, type_mult)
    mult *= get_held_item_multiplier(move, move_type, attacker, target, type_mult)

    return mult

def evaluate_move_power_and_multipliers(
    move: pkmn_b.Move,
    attacker: pkmn_b.Pokemon,
    target_1: Optional[pkmn_b.Pokemon],
    target_2: Optional[pkmn_b.Pokemon],
    battle: pkmn_b.DoubleBattle
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
    nominal_bp = float(move.base_power)

    spread_penalty = get_spread_multiplier(move, attacker, target_1, target_2, battle)

    mult_1 = get_effective_damage_multiplier(move, bp_1, attacker, target_1, battle) * spread_penalty
    mult_2 = get_effective_damage_multiplier(move, bp_2, attacker, target_2, battle) * spread_penalty

    eff_damage_1 = (bp_1 * mult_1) / 300.0
    eff_damage_2 = (bp_2 * mult_2) / 300.0

    return nominal_bp, eff_damage_1, eff_damage_2
