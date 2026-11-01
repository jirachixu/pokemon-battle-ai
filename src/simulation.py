"""
High-performance forward simulation engine for competitive Pokemon Doubles (VGC / Champions).
Supports simultaneous-turn resolution, damage calculation, and multi-turn lookahead.
"""
import math
import unicodedata
import os
import re
import json
from typing import Optional, Union, Any

import poke_env.battle as pkmn_b
from poke_env.battle import Effect
import src.constants as constants
from src.constants import _SOUND_MOVES, _TWO_TURN_CHARGE_MOVES, _BULLET_MOVES, _DEFAULT_RECOIL_RATIOS
import src.meta_priors as meta_priors

# In Champions, IVs are max
MAXIMUM_INDIVIDUAL_VALUE: int = 31

# ----------------- Static Data Loader -----------------

_MOVES_CACHE: dict[str, dict] = {}
_POKEDEX_CACHE: dict[str, dict] = {}
_TYPECHART_CACHE: dict[str, dict] = {}

def _load_static_data() -> None:
    global _MOVES_CACHE, _POKEDEX_CACHE, _TYPECHART_CACHE
    if _MOVES_CACHE:
        return

    # Check potential locations for poke_env static data directory
    candidate_directories = [
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".venv", "lib", "python3.14", "site-packages", "poke_env", "data", "static")
        ),
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".venv", "Lib", "site-packages", "poke_env", "data", "static")
        ),
    ]

    discovered_directory = None
    for candidate_directory in candidate_directories:
        if os.path.exists(candidate_directory):
            discovered_directory = candidate_directory
            break

    if discovered_directory:
        moves_file_path = os.path.join(discovered_directory, "moves", "gen9moves.json")
        pokedex_file_path = os.path.join(discovered_directory, "pokedex", "gen9pokedex.json")
        typechart_file_path = os.path.join(discovered_directory, "typechart", "gen9typechart.json")

        if os.path.exists(moves_file_path):
            with open(moves_file_path, "r", encoding="utf-8") as moves_file:
                _MOVES_CACHE = json.load(moves_file)
        if os.path.exists(pokedex_file_path):
            with open(pokedex_file_path, "r", encoding="utf-8") as pokedex_file:
                _POKEDEX_CACHE = json.load(pokedex_file)
        if os.path.exists(typechart_file_path):
            with open(typechart_file_path, "r", encoding="utf-8") as typechart_file:
                _TYPECHART_CACHE = json.load(typechart_file)

_load_static_data()

def clean_id(value: Any) -> str:
    """Normalizes identifiers to lowercase alphanumeric, stripping accents and mojibake."""
    if not value:
        return ""
    if hasattr(value, "name"):
        value = value.name
    s = str(value)
    # Strip common UTF-8 mojibake artifacts
    s = s.replace("â€™", "").replace("â", "").replace("’", "").replace("'", "")
    # Decompose unicode accents (e.g. Flabébé -> Flabebe)
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()

def get_type_effectiveness(attacking_type: str, defending_types: tuple[str, ...], has_gravity: bool = False) -> float:
    """Calculates type effectiveness multiplier against defending types."""
    if not _TYPECHART_CACHE:
        _load_static_data()

    total_multiplier = 1.0
    cleaned_attacking_type = clean_id(attacking_type)
    attacking_type_capitalized = cleaned_attacking_type.capitalize()
    for defending_type in defending_types:
        cleaned_defending_type = clean_id(defending_type)
        if has_gravity and cleaned_attacking_type == "ground" and cleaned_defending_type == "flying":
            # Gravity grounds Flying types, making Ground hit neutrally
            continue
        chart_entry = _TYPECHART_CACHE.get(cleaned_defending_type)
        if not chart_entry:
            continue
        damage_taken_dictionary = chart_entry.get("damageTaken", {})
        effectiveness_code = damage_taken_dictionary.get(attacking_type_capitalized, 0)
        if effectiveness_code == 1:
            total_multiplier *= 2.0
        elif effectiveness_code == 2:
            total_multiplier *= 0.5
        elif effectiveness_code == 3:
            total_multiplier *= 0.0
    return total_multiplier

# ----------------- Competitive VGC & Champions Reg M-C Lookup Tables -----------------

_WIND_MOVES: set[str] = {
    "aircutter", "bleakwindstorm", "blizzard", "fairywind", "gust",
    "heatwave", "hurricane", "icywind", "petalblizzard", "sandsearstorm",
    "springtidestorm", "tailwind", "twister", "whirlwind", "wildboltstorm",
}

_GUARANTEED_CRIT_MOVES: set[str] = {
    "surgingstrikes", "wickedblow", "flowertrick", "stormthrow",
    "frostbreath", "zippyzap", "laserfocus",
}

_HIGH_CRIT_MOVES: set[str] = {
    "aeroblast", "aircutter", "aquacutter", "attackorder", "blazekick",
    "crabhammer", "crosschop", "crosspoison", "direclaw", "drillrun",
    "esperwing", "ivycudgel", "karatechop", "leafblade", "nightslash",
    "poisonTail", "poisontail", "psychocut", "razorleaf", "razorwind",
    "shadowclaw", "skyattack", "slash", "snipeshot", "spacialrend",
    "stoneedge", "triplearrows",
}

_PIVOT_MOVES: set[str] = {
    "uturn", "voltswitch", "flipturn", "partingshot", "batonpass",
    "chillyreception", "shedtail", "teleport",
}

_PROTECT_BREAKING_MOVES: set[str] = {
    "feint", "hyperspacefury", "hyperspacehole", "phantomforce",
    "shadowforce", "mightycleave",
}

_RECHARGE_MOVES: set[str] = {
    "hyperbeam", "gigaimpact", "blastburn", "frenzyplant", "hydrocannon",
    "roaroftime", "rockwrecker", "prismaticlaser", "meteorassault", "eternabeam",
}

_RAMPAGE_MOVES: set[str] = {
    "outrage", "thrash", "petaldance", "ragingfury",
}

_UNSUPPRESSIBLE_ABILITIES: set[str] = {
    "neutralizinggas", "comatose", "disguise", "iceface", "multitype",
    "rkssystem", "schooling", "shieldsdown", "stancechange", "terashift",
    "terashell", "teraformzero", "zerotohero", "asone", "asoneglastrier",
    "asonespectrier", "commander", "gulpmissile", "hungerswitch",
}

_PINCH_HEAL_BERRIES: set[str] = {
    "figyberry", "wikiberry", "magoberry", "aguavberry", "iapapaberry",
}

_PINCH_STAT_BERRIES: dict[str, tuple[str, int]] = {
    "liechiberry": ("atk", 1),
    "ganlonberry": ("def", 1),
    "salacberry": ("spe", 1),
    "petayaberry": ("spa", 1),
    "apicotberry": ("spd", 1),
    "starfberry": ("atk", 2),
}

_STATUS_CURE_BERRIES: dict[str, tuple[str, ...]] = {
    "lumberry": ("slp", "par", "brn", "psn", "tox", "frz", "confusion"),
    "chestoberry": ("slp",),
    "cheriberry": ("par",),
    "pechaberry": ("psn", "tox"),
    "rawstberry": ("brn",),
    "aspearberry": ("frz",),
    "persimberry": ("confusion",),
}

# ----------------- Simulated Core Objects -----------------

class SimulatedMove:
    __slots__ = (
        "id", "name", "type", "category", "base_power", "accuracy",
        "priority", "target", "flags", "secondary", "current_pp",
        "max_pp", "is_spread", "recoil", "self_destruct_recoil", "has_crash_damage",
        "crit_ratio", "will_crit", "multihit", "drain", "heal_fraction",
        "self_boosts", "target_boosts", "self_switch", "breaks_protect",
        "tracks_target", "override_offensive_stat", "override_defensive_stat"
    )

    def __init__(
        self,
        move_id: str,
        name: str = "",
        move_type: str = "normal",
        category: str = "physical",
        base_power: float = 0.0,
        accuracy: float = 1.0,
        priority: int = 0,
        target: str = "normal",
        flags: set[str] | None = None,
        secondary: Any = None,
        current_pp: int = 10,
        max_pp: int = 10,
        is_spread: Optional[bool] = None,
        recoil: Optional[float] = None,
        self_destruct_recoil: Optional[float] = None,
        has_crash_damage: Optional[bool] = None,
        crit_ratio: int = 1,
        will_crit: Optional[bool] = None,
        multihit: Optional[Union[int, tuple[int, int]]] = None,
        drain: float = 0.0,
        heal_fraction: float = 0.0,
        self_boosts: Optional[dict[str, int]] = None,
        target_boosts: Optional[dict[str, int]] = None,
        self_switch: Optional[bool] = None,
        breaks_protect: Optional[bool] = None,
        tracks_target: Optional[bool] = None,
        override_offensive_stat: Optional[str] = None,
        override_defensive_stat: Optional[str] = None,
    ):
        self.id = clean_id(move_id)
        self.name = name or self.id
        self.type = clean_id(move_type)
        self.category = clean_id(category)  # "physical", "special", "status"
        self.base_power = float(base_power)
        self.accuracy = float(accuracy)
        self.priority = int(priority)
        self.target = str(target)
        self.flags = flags if flags is not None else set()
        if self.id in _SOUND_MOVES:
            self.flags.add("sound")
        if self.id in _TWO_TURN_CHARGE_MOVES:
            self.flags.add("charge")
        if self.id in _BULLET_MOVES:
            self.flags.add("bullet")
        if self.id in _WIND_MOVES:
            self.flags.add("wind")
        self.secondary = secondary
        self.current_pp = int(current_pp)
        self.max_pp = int(max_pp)
        self.is_spread = is_spread if is_spread is not None else (self.target in ("allAdjacentFoes", "allAdjacent"))
        self.recoil: float = float(recoil) if recoil is not None else _DEFAULT_RECOIL_RATIOS.get(self.id, 0.0)
        self.self_destruct_recoil: float = (
            float(self_destruct_recoil)
            if self_destruct_recoil is not None
            else (0.50 if self.id in ("mindblown", "steelbeam", "chloroblast") else 0.0)
        )
        self.has_crash_damage: bool = (
            bool(has_crash_damage)
            if has_crash_damage is not None
            else (self.id in ("highjumpkick", "jumpkick", "axekick", "supercellslam"))
        )
        self.will_crit: bool = (
            bool(will_crit)
            if will_crit is not None
            else (self.id in _GUARANTEED_CRIT_MOVES)
        )
        self.crit_ratio: int = (
            4 if self.will_crit
            else (max(2, int(crit_ratio)) if self.id in _HIGH_CRIT_MOVES else int(crit_ratio))
        )
        self.multihit: Optional[Union[int, tuple[int, int]]] = multihit
        self.drain: float = float(drain)
        self.heal_fraction: float = float(heal_fraction)
        self.self_boosts: Optional[dict[str, int]] = self_boosts.copy() if self_boosts else None
        self.target_boosts: Optional[dict[str, int]] = target_boosts.copy() if target_boosts else None
        self.self_switch: bool = (
            bool(self_switch)
            if self_switch is not None
            else (self.id in _PIVOT_MOVES)
        )
        self.breaks_protect: bool = (
            bool(breaks_protect)
            if breaks_protect is not None
            else (self.id in _PROTECT_BREAKING_MOVES)
        )
        self.tracks_target: bool = (
            bool(tracks_target)
            if tracks_target is not None
            else (self.id == "snipeshot")
        )
        self.override_offensive_stat: Optional[str] = override_offensive_stat
        self.override_defensive_stat: Optional[str] = override_defensive_stat

    @classmethod
    def from_id(cls, move_id: str) -> "SimulatedMove":
        cleaned_move_id = clean_id(move_id)
        raw_move_data = _MOVES_CACHE.get(cleaned_move_id, {})
        flags = set(raw_move_data.get("flags", {}).keys())
        raw_accuracy = raw_move_data.get("accuracy", 100)
        accuracy_float = 1.0 if (raw_accuracy is True or raw_accuracy == 100) else (
            raw_accuracy / 100.0 if isinstance(raw_accuracy, (int, float)) else 1.0
        )
        priority = raw_move_data.get("priority", 0)

        # Explicit competitive priority brackets
        if cleaned_move_id in constants.PROTECT_MOVES or cleaned_move_id == "endure":
            priority = 4
        elif cleaned_move_id in ("wideguard", "quickguard", "craftyshield", "upperhand"):
            priority = 3
        elif cleaned_move_id == "fakeout":
            priority = 3
        elif cleaned_move_id in ("followme", "ragepowder", "allyswitch", "feint", "firstimpression"):
            priority = 2
        elif cleaned_move_id == "extremespeed":
            priority = 2
        elif cleaned_move_id in (
            "aquajet", "machpunch", "bulletpunch", "iceshard",
            "shadowsneak", "suckerpunch", "vacuumwave", "thunderclap",
            "accelerock", "quickattack", "watershuriken", "jetpunch", "babydolleyes"
        ):
            priority = 1
        elif cleaned_move_id == "trickroom":
            priority = -7

        raw_recoil = raw_move_data.get("recoil")
        if isinstance(raw_recoil, list) and len(raw_recoil) == 2 and raw_recoil[1] > 0:
            recoil_val = float(raw_recoil[0]) / float(raw_recoil[1])
        else:
            recoil_val = _DEFAULT_RECOIL_RATIOS.get(cleaned_move_id, 0.0)

        self_destruct_val = 0.50 if (
            raw_move_data.get("mindBlownRecoil")
            or raw_move_data.get("chloroblastRecoil")
            or cleaned_move_id in ("mindblown", "steelbeam", "chloroblast")
        ) else 0.0

        crash_val = bool(
            raw_move_data.get("hasCrashDamage")
            or cleaned_move_id in ("highjumpkick", "jumpkick", "axekick", "supercellslam")
        )

        will_crit_val = bool(raw_move_data.get("willCrit")) or cleaned_move_id in _GUARANTEED_CRIT_MOVES
        crit_ratio_val = int(raw_move_data.get("critRatio", 1))
        if cleaned_move_id in _HIGH_CRIT_MOVES:
            crit_ratio_val = max(2, crit_ratio_val)

        raw_multihit = raw_move_data.get("multihit")
        multihit_val: Optional[Union[int, tuple[int, int]]] = None
        if isinstance(raw_multihit, int):
            multihit_val = raw_multihit
        elif isinstance(raw_multihit, list) and len(raw_multihit) == 2:
            multihit_val = (int(raw_multihit[0]), int(raw_multihit[1]))

        raw_drain = raw_move_data.get("drain")
        if isinstance(raw_drain, list) and len(raw_drain) == 2 and raw_drain[1] > 0:
            drain_val = float(raw_drain[0]) / float(raw_drain[1])
        else:
            drain_val = 0.75 if cleaned_move_id in ("drainingkiss", "oblivionwing") else (
                0.50 if cleaned_move_id in (
                    "drainpunch", "gigadrain", "hornleech", "bitterblade",
                    "paraboliccharge", "matchagotcha", "leechlife", "absorb", "megadrain"
                ) else 0.0
            )

        raw_heal = raw_move_data.get("heal")
        if isinstance(raw_heal, list) and len(raw_heal) == 2 and raw_heal[1] > 0:
            heal_val = float(raw_heal[0]) / float(raw_heal[1])
        else:
            heal_val = 0.0

        raw_self = raw_move_data.get("self") or raw_move_data.get("selfBoost")
        self_boosts_val = raw_self.get("boosts") if isinstance(raw_self, dict) else None
        target_boosts_val = raw_move_data.get("boosts") if isinstance(raw_move_data.get("boosts"), dict) else None

        self_switch_val = bool(raw_move_data.get("selfSwitch")) or cleaned_move_id in _PIVOT_MOVES
        breaks_protect_val = bool(raw_move_data.get("breaksProtect")) or cleaned_move_id in _PROTECT_BREAKING_MOVES
        tracks_target_val = bool(raw_move_data.get("tracksTarget")) or cleaned_move_id == "snipeshot"

        override_off = raw_move_data.get("overrideOffensiveStat")
        override_def = raw_move_data.get("overrideDefensiveStat")

        return cls(
            move_id=cleaned_move_id,
            name=raw_move_data.get("name", cleaned_move_id),
            move_type=clean_id(raw_move_data.get("type", "normal")),
            category=clean_id(raw_move_data.get("category", "physical")),
            base_power=float(raw_move_data.get("basePower", 0)),
            accuracy=accuracy_float,
            priority=priority,
            target=raw_move_data.get("target", "normal"),
            flags=flags,
            secondary=raw_move_data.get("secondary"),
            current_pp=raw_move_data.get("pp", 10),
            max_pp=raw_move_data.get("pp", 10),
            recoil=recoil_val,
            self_destruct_recoil=self_destruct_val,
            has_crash_damage=crash_val,
            crit_ratio=crit_ratio_val,
            will_crit=will_crit_val,
            multihit=multihit_val,
            drain=drain_val,
            heal_fraction=heal_val,
            self_boosts=self_boosts_val,
            target_boosts=target_boosts_val,
            self_switch=self_switch_val,
            breaks_protect=breaks_protect_val,
            tracks_target=tracks_target_val,
            override_offensive_stat=override_off,
            override_defensive_stat=override_def,
        )

    def clone(self) -> "SimulatedMove":
        cloned_move = SimulatedMove.__new__(SimulatedMove)
        cloned_move.id = self.id
        cloned_move.name = self.name
        cloned_move.type = self.type
        cloned_move.category = self.category
        cloned_move.base_power = self.base_power
        cloned_move.accuracy = self.accuracy
        cloned_move.priority = self.priority
        cloned_move.target = self.target
        cloned_move.flags = self.flags.copy()
        cloned_move.secondary = self.secondary
        cloned_move.current_pp = self.current_pp
        cloned_move.max_pp = self.max_pp
        cloned_move.is_spread = self.is_spread
        cloned_move.recoil = self.recoil
        cloned_move.self_destruct_recoil = self.self_destruct_recoil
        cloned_move.has_crash_damage = self.has_crash_damage
        cloned_move.crit_ratio = self.crit_ratio
        cloned_move.will_crit = self.will_crit
        cloned_move.multihit = self.multihit
        cloned_move.drain = self.drain
        cloned_move.heal_fraction = self.heal_fraction
        cloned_move.self_boosts = self.self_boosts.copy() if self.self_boosts else None
        cloned_move.target_boosts = self.target_boosts.copy() if self.target_boosts else None
        cloned_move.self_switch = self.self_switch
        cloned_move.breaks_protect = self.breaks_protect
        cloned_move.tracks_target = self.tracks_target
        cloned_move.override_offensive_stat = self.override_offensive_stat
        cloned_move.override_defensive_stat = self.override_defensive_stat
        return cloned_move

class SimulatedPokemon:
    __slots__ = (
        "species", "name", "types", "base_types", "roost_active", "protean_activated",
        "unburden_active",
        "level", "current_hp", "max_hp",
        "current_hp_fraction", "base_stats", "stats", "boosts",
        "status", "toxic_turns", "item", "ability", "moves",
        "protect_counter", "is_protected", "first_turn", "is_mega", "can_mega_evolve",
        "fainted", "side", "slot", "flinch", "weight", "helping_hand",
        "taunt_turns", "encore_turns", "encore_move", "disabled_move",
        "disable_turns", "torment", "destiny_bond", "cursed", "last_move_used", "just_flinched",
        "perish_turns", "is_trapped", "imprison", "throat_chop_turns",
        "charging_move", "charging_target", "effects",
        "evs", "nature", "base_ability", "protect_variant", "endure_active",
        "substitute_hp", "leech_seed", "leech_seed_slot", "salt_cure",
        "syrup_bomb_turns", "yawn_turns", "confusion_turns", "sleep_turns",
        "flash_fire_active", "charge_active", "glaive_rush_vulnerable",
        "disguise_busted", "ice_face_busted", "paradox_boost_stat", "booster_energy_active",
        "once_per_battle_ability_used", "hero_form_active", "took_damage_this_turn",
        "last_move_failed", "moved_this_turn", "switched_in_this_turn",
        "cud_chew_berry", "last_consumed_berry", "crit_stage", "minimize_active",
        "must_recharge", "locked_move", "locked_move_turns", "truant_resting",
        "slow_start_turns", "consecutive_move_uses"
    )

    def __init__(
        self,
        species: str,
        name: str = "",
        types: tuple[str, ...] = ("normal",),
        level: int = 50,
        hp: float = 100.0,
        max_hp: float = 100.0,
        base_stats: dict[str, int] | None = None,
        stats: dict[str, int] | None = None,
        boosts: dict[str, int] | None = None,
        status: Optional[str] = None,
        item: str = "none",
        ability: str = "none",
        moves: dict[str, SimulatedMove] | None = None,
        protect_counter: int = 0,
        first_turn: bool = True,
        is_mega: bool = False,
        can_mega_evolve: bool = False,
        side: int = 1,
        slot: int = 0,
        weight: float = 50.0,
        just_flinched: bool = False,
        perish_turns: int = 0,
        is_trapped: bool = False,
        imprison: bool = False,
        throat_chop_turns: int = 0,
        charging_move: Optional[str] = None,
        charging_target: int = 0,
        evs: Optional[dict[str, int]] = None,
        nature: str = "hardy",
    ):
        self.species = clean_id(species)
        self.name = name or self.species
        self.types = tuple(clean_id(type_name) for type_name in types)
        self.base_types = self.types
        self.roost_active: bool = False
        self.protean_activated: bool = False
        self.unburden_active: bool = False
        self.level = level
        self.max_hp = max(1.0, float(max_hp))
        self.current_hp = max(0.0, min(self.max_hp, float(hp)))
        self.current_hp_fraction = self.current_hp / self.max_hp

        self.evs: dict[str, int] = evs.copy() if evs is not None else {}
        self.nature: str = clean_id(nature) or "hardy"
        self.base_stats = base_stats or {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80}
        self.stats = stats or self._calculate_stats_from_base(effort_values_spread=self.evs, nature=self.nature)
        self.boosts = boosts if boosts is not None else {
            "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0
        }
        self.status = clean_id(status) if status else None
        self.toxic_turns = 0
        self.item = clean_id(item)
        self.ability = clean_id(ability)
        self.base_ability: str = self.ability
        self.moves = moves if moves is not None else {}
        self.protect_counter = protect_counter
        self.is_protected: bool = False
        self.protect_variant: Optional[str] = None
        self.endure_active: bool = False
        self.first_turn = first_turn
        self.is_mega = is_mega
        self.can_mega_evolve = can_mega_evolve
        self.fainted = self.current_hp <= 0.0
        self.side = side
        self.slot = slot
        self.flinch = False
        self.weight = float(weight)
        self.helping_hand = False

        self.taunt_turns: int = 0
        self.encore_turns: int = 0
        self.encore_move: Optional[str] = None
        self.disabled_move: Optional[str] = None
        self.disable_turns: int = 0
        self.torment: bool = False
        self.destiny_bond: bool = False
        self.cursed: bool = False
        self.last_move_used: Optional[str] = None
        self.just_flinched: bool = just_flinched
        self.perish_turns: int = perish_turns
        self.is_trapped: bool = is_trapped
        self.imprison: bool = imprison
        self.throat_chop_turns: int = throat_chop_turns
        self.charging_move: Optional[str] = charging_move
        self.charging_target: int = charging_target
        self.effects = {}

        # Extended competitive VGC / Champions state slots
        self.substitute_hp: float = 0.0
        self.leech_seed: bool = False
        self.leech_seed_slot: int = 0
        self.salt_cure: bool = False
        self.syrup_bomb_turns: int = 0
        self.yawn_turns: int = 0
        self.confusion_turns: int = 0
        self.sleep_turns: int = 0
        self.flash_fire_active: bool = False
        self.charge_active: bool = False
        self.glaive_rush_vulnerable: bool = False
        self.disguise_busted: bool = False
        self.ice_face_busted: bool = False
        self.paradox_boost_stat: Optional[str] = None
        self.booster_energy_active: bool = False
        self.once_per_battle_ability_used: bool = False
        self.hero_form_active: bool = (self.species == "palafinhero")
        self.took_damage_this_turn: bool = False
        self.last_move_failed: bool = False
        self.moved_this_turn: bool = False
        self.switched_in_this_turn: bool = False
        self.cud_chew_berry: Optional[str] = None
        self.last_consumed_berry: Optional[str] = None
        self.crit_stage: int = 0
        self.minimize_active: bool = False
        self.must_recharge: bool = False
        self.locked_move: Optional[str] = None
        self.locked_move_turns: int = 0
        self.truant_resting: bool = False
        self.slow_start_turns: int = 5 if self.ability == "slowstart" else 0
        self.consecutive_move_uses: int = 0

    def _calculate_stats_from_base(
        self,
        effort_values_spread: dict[str, int] | None = None,
        nature: str = "hardy"
    ) -> dict[str, int]:
        """Calculates level 50 competitive stats with permanently maxed IVs (31) for Pokemon Champions."""
        effort_values = effort_values_spread if effort_values_spread is not None else self.evs
        nature = nature if nature != "hardy" or not getattr(self, "nature", "") else self.nature
        calculated_stats = {}
        for stat_name in ("hp", "atk", "def", "spa", "spd", "spe"):
            base_stat = self.base_stats.get(stat_name, 80)
            effort_value = effort_values.get(stat_name, 0)
            if stat_name == "hp":
                # Special check for Shedinja (HP is always 1)
                if base_stat == 1:
                    calculated_stats["hp"] = 1
                else:
                    calculated_stats["hp"] = (
                        math.floor((2 * base_stat + MAXIMUM_INDIVIDUAL_VALUE + math.floor(effort_value / 4)) * self.level / 100)
                        + 10 + self.level
                    )
            else:
                raw_stat = (
                    math.floor((2 * base_stat + MAXIMUM_INDIVIDUAL_VALUE + math.floor(effort_value / 4)) * self.level / 100) + 5
                )
                nature_multiplier = 1.0
                if nature in ("adamant", "lonely", "brave", "naughty") and stat_name == "atk":
                    nature_multiplier = 1.1
                elif nature in ("modest", "mild", "quiet", "rash") and stat_name == "spa":
                    nature_multiplier = 1.1
                elif nature in ("jolly", "timid", "hasty", "naive") and stat_name == "spe":
                    nature_multiplier = 1.1
                elif nature in ("bold", "impish", "relaxed", "lax") and stat_name == "def":
                    nature_multiplier = 1.1
                elif nature in ("calm", "careful", "sassy", "gentle") and stat_name == "spd":
                    nature_multiplier = 1.1

                if nature in ("modest", "timid", "calm", "bold") and stat_name == "atk":
                    nature_multiplier = 0.9
                elif nature in ("adamant", "jolly", "careful", "impish") and stat_name == "spa":
                    nature_multiplier = 0.9
                elif nature in ("brave", "quiet", "relaxed", "sassy") and stat_name == "spe":
                    nature_multiplier = 0.9
                elif nature in ("lonely", "mild", "gentle", "hasty") and stat_name == "def":
                    nature_multiplier = 0.9
                elif nature in ("naughty", "lax", "rash", "naive") and stat_name == "spd":
                    nature_multiplier = 0.9

                calculated_stats[stat_name] = math.floor(raw_stat * nature_multiplier)
        return calculated_stats

    @classmethod
    def from_species(
        cls,
        species: str,
        side: int = 1,
        slot: int = 0,
        revealed_moves: list[str] | None = None,
        revealed_item: Optional[str] = None,
        revealed_ability: Optional[str] = None,
        fill_champions_priors: bool = True
    ) -> "SimulatedPokemon":
        cleaned_species = clean_id(species)

        # Check custom Champions Mega stats first
        if cleaned_species in constants.MEGA_SPECIES_STATS:
            mega_data = constants.MEGA_SPECIES_STATS[cleaned_species]
            types = mega_data["types"]
            ability = revealed_ability or mega_data["ability"]
            base_stats = mega_data["base_stats"]
            is_mega = True
            weight = float(_POKEDEX_CACHE.get(cleaned_species, {}).get("weightkg", 80.0))
        else:
            pokedex_data = _POKEDEX_CACHE.get(cleaned_species, {})
            types = tuple(clean_id(type_name) for type_name in pokedex_data.get("types", ("normal",)))
            base_stats = pokedex_data.get(
                "baseStats", {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80}
            )
            is_mega = cleaned_species.endswith(("mega", "megax", "megay", "megaz")) and cleaned_species != "yanmega"
            weight = float(pokedex_data.get("weightkg", 50.0))
            ability = clean_id(revealed_ability or "")
            if not ability and "abilities" in pokedex_data:
                ability = clean_id(pokedex_data["abilities"].get("0", "none"))

        item = clean_id(revealed_item or "none")
        spread = meta_priors.get_champions_common_spread(cleaned_species) if fill_champions_priors else {"nature": "hardy", "evs": {}}

        # Empirical Champions Reg M-C prior inference for missing attributes
        if fill_champions_priors:
            if not ability or ability == "none":
                ability_priors = meta_priors.get_champions_ability_priors(cleaned_species)
                if ability_priors:
                    ability = sorted(ability_priors.items(), key=lambda prior_pair: prior_pair[1], reverse=True)[0][0]
            if not item or item == "none":
                item_priors = meta_priors.get_champions_item_priors(cleaned_species)
                if item_priors:
                    item = sorted(item_priors.items(), key=lambda prior_pair: prior_pair[1], reverse=True)[0][0]

        # Determine moveset
        moves_dictionary = {}
        if revealed_moves:
            for move_id in revealed_moves:
                moves_dictionary[clean_id(move_id)] = SimulatedMove.from_id(move_id)

        if fill_champions_priors and len(moves_dictionary) < 4:
            move_priors = meta_priors.get_champions_move_priors(cleaned_species)
            for move_id, _ in sorted(move_priors.items(), key=lambda prior_pair: prior_pair[1], reverse=True):
                if move_id not in moves_dictionary:
                    moves_dictionary[move_id] = SimulatedMove.from_id(move_id)
                if len(moves_dictionary) >= 4:
                    break

        mega_stones = constants.BASE_TO_MEGA_STONE.get(cleaned_species, [])
        can_mega = (not is_mega and bool(mega_stones) and (item in mega_stones or cleaned_species == "rayquaza"))
        evs_dict = spread.get("evs") or {}
        nature_str = spread.get("nature", "hardy")
        simulated_pokemon = cls(
            species=cleaned_species,
            types=types,
            base_stats=base_stats,
            item=item,
            ability=ability,
            moves=moves_dictionary,
            is_mega=is_mega,
            can_mega_evolve=can_mega,
            side=side,
            slot=slot,
            weight=weight,
            evs=evs_dict,
            nature=nature_str,
        )
        simulated_pokemon.stats = simulated_pokemon._calculate_stats_from_base(
            effort_values_spread=evs_dict, nature=nature_str
        )
        simulated_pokemon.max_hp = float(simulated_pokemon.stats["hp"])
        simulated_pokemon.current_hp = simulated_pokemon.max_hp
        simulated_pokemon.current_hp_fraction = 1.0
        return simulated_pokemon

    def clone(self) -> "SimulatedPokemon":
        cloned_pokemon = SimulatedPokemon.__new__(SimulatedPokemon)
        cloned_pokemon.species = self.species
        cloned_pokemon.name = self.name
        cloned_pokemon.types = self.types
        cloned_pokemon.base_types = self.base_types
        cloned_pokemon.roost_active = self.roost_active
        cloned_pokemon.protean_activated = self.protean_activated
        cloned_pokemon.unburden_active = self.unburden_active
        cloned_pokemon.level = self.level
        cloned_pokemon.current_hp = self.current_hp
        cloned_pokemon.max_hp = self.max_hp
        cloned_pokemon.current_hp_fraction = self.current_hp_fraction
        cloned_pokemon.base_stats = self.base_stats
        cloned_pokemon.stats = self.stats.copy()
        cloned_pokemon.boosts = self.boosts.copy()
        cloned_pokemon.status = self.status
        cloned_pokemon.toxic_turns = self.toxic_turns
        cloned_pokemon.item = self.item
        cloned_pokemon.ability = self.ability
        cloned_pokemon.moves = {move_id: move_object.clone() for move_id, move_object in self.moves.items()}
        cloned_pokemon.protect_counter = self.protect_counter
        cloned_pokemon.is_protected = self.is_protected
        cloned_pokemon.first_turn = self.first_turn
        cloned_pokemon.is_mega = self.is_mega
        cloned_pokemon.can_mega_evolve = self.can_mega_evolve
        cloned_pokemon.fainted = self.fainted
        cloned_pokemon.side = self.side
        cloned_pokemon.slot = self.slot
        cloned_pokemon.flinch = self.flinch
        cloned_pokemon.weight = self.weight
        cloned_pokemon.helping_hand = self.helping_hand
        cloned_pokemon.taunt_turns = self.taunt_turns
        cloned_pokemon.encore_turns = self.encore_turns
        cloned_pokemon.encore_move = self.encore_move
        cloned_pokemon.disabled_move = self.disabled_move
        cloned_pokemon.disable_turns = self.disable_turns
        cloned_pokemon.torment = self.torment
        cloned_pokemon.destiny_bond = self.destiny_bond
        cloned_pokemon.cursed = self.cursed
        cloned_pokemon.last_move_used = self.last_move_used
        cloned_pokemon.just_flinched = self.just_flinched
        cloned_pokemon.perish_turns = self.perish_turns
        cloned_pokemon.is_trapped = self.is_trapped
        cloned_pokemon.imprison = self.imprison
        cloned_pokemon.throat_chop_turns = self.throat_chop_turns
        cloned_pokemon.charging_move = self.charging_move
        cloned_pokemon.charging_target = self.charging_target
        cloned_pokemon.effects = self.effects.copy()
        cloned_pokemon.evs = self.evs.copy()
        cloned_pokemon.nature = self.nature
        cloned_pokemon.base_ability = self.base_ability
        cloned_pokemon.protect_variant = self.protect_variant
        cloned_pokemon.endure_active = self.endure_active
        cloned_pokemon.substitute_hp = self.substitute_hp
        cloned_pokemon.leech_seed = self.leech_seed
        cloned_pokemon.leech_seed_slot = self.leech_seed_slot
        cloned_pokemon.salt_cure = self.salt_cure
        cloned_pokemon.syrup_bomb_turns = self.syrup_bomb_turns
        cloned_pokemon.yawn_turns = self.yawn_turns
        cloned_pokemon.confusion_turns = self.confusion_turns
        cloned_pokemon.sleep_turns = self.sleep_turns
        cloned_pokemon.flash_fire_active = self.flash_fire_active
        cloned_pokemon.charge_active = self.charge_active
        cloned_pokemon.glaive_rush_vulnerable = self.glaive_rush_vulnerable
        cloned_pokemon.disguise_busted = self.disguise_busted
        cloned_pokemon.ice_face_busted = self.ice_face_busted
        cloned_pokemon.paradox_boost_stat = self.paradox_boost_stat
        cloned_pokemon.booster_energy_active = self.booster_energy_active
        cloned_pokemon.once_per_battle_ability_used = self.once_per_battle_ability_used
        cloned_pokemon.hero_form_active = self.hero_form_active
        cloned_pokemon.took_damage_this_turn = self.took_damage_this_turn
        cloned_pokemon.last_move_failed = self.last_move_failed
        cloned_pokemon.moved_this_turn = self.moved_this_turn
        cloned_pokemon.switched_in_this_turn = self.switched_in_this_turn
        cloned_pokemon.cud_chew_berry = self.cud_chew_berry
        cloned_pokemon.last_consumed_berry = self.last_consumed_berry
        cloned_pokemon.crit_stage = self.crit_stage
        cloned_pokemon.minimize_active = self.minimize_active
        cloned_pokemon.must_recharge = self.must_recharge
        cloned_pokemon.locked_move = self.locked_move
        cloned_pokemon.locked_move_turns = self.locked_move_turns
        cloned_pokemon.truant_resting = self.truant_resting
        cloned_pokemon.slow_start_turns = self.slow_start_turns
        cloned_pokemon.consecutive_move_uses = self.consecutive_move_uses
        return cloned_pokemon

    def consume_item(self) -> None:
        """Consumes or removes the Pokemon's held item, activating Unburden, Cheek Pouch, and Cud Chew if applicable."""
        if self.item not in ("", "none", "unknownitem"):
            consumed = self.item
            self.item = ""
            if consumed.endswith("berry"):
                self.last_consumed_berry = consumed
                if self.ability == "cheekpouch" and not self.fainted:
                    self.heal(self.max_hp / 3.0)
                if self.ability == "cudchew":
                    self.cud_chew_berry = consumed
            if self.ability == "unburden":
                self.unburden_active = True

    def take_damage(self, damage_amount: float) -> float:
        """Deducts damage and updates faint status. Returns actual damage taken."""
        actual_damage = min(self.current_hp, max(0.0, damage_amount))
        if actual_damage > 0.0 and self.item == "airballoon":
            self.consume_item()
        self.current_hp = max(0.0, self.current_hp - actual_damage)
        self.current_hp_fraction = self.current_hp / self.max_hp
        if self.current_hp <= 0.0:
            self.fainted = True
        return actual_damage

    def heal(self, heal_amount: float) -> float:
        """Heals HP up to max_hp. Returns actual healed amount."""
        if self.fainted:
            return 0.0
        actual_healing = min(self.max_hp - self.current_hp, max(0.0, heal_amount))
        self.current_hp = min(self.max_hp, self.current_hp + actual_healing)
        self.current_hp_fraction = self.current_hp / self.max_hp
        return actual_healing

    def apply_boost(self, stat_name: str, stages: int) -> None:
        """Applies stat stage modification [-6, +6], accounting for Contrary and Simple."""
        if stat_name in self.boosts:
            if self.ability == "contrary":
                stages = -stages
            elif self.ability == "simple":
                stages *= 2
            current_boost = self.boosts[stat_name]
            self.boosts[stat_name] = max(-6, min(6, current_boost + stages))

    def check_white_herb(self) -> None:
        """Restores any negative stat stage to 0 and consumes White Herb if held."""
        if self.item == "whiteherb" and any(v < 0 for v in self.boosts.values()):
            for stat_key in self.boosts:
                if self.boosts[stat_key] < 0:
                    self.boosts[stat_key] = 0
            self.consume_item()

    def clear_boosts(self) -> None:
        for stat_name in self.boosts:
            self.boosts[stat_name] = 0

    def highest_stat_name(self) -> str:
        """Returns the highest non-HP stat name ('atk', 'def', 'spa', 'spd', 'spe') for Protosynthesis/Quark Drive/Beast Boost."""
        best_stat = "atk"
        best_val = -1.0
        for s_name in ("atk", "def", "spa", "spd", "spe"):
            raw = float(self.stats.get(s_name, 80))
            boost = self.boosts.get(s_name, 0)
            mult = (2.0 + boost) / 2.0 if boost >= 0 else 2.0 / (2.0 - boost)
            val = raw * mult
            if val > best_val:
                best_val = val
                best_stat = s_name
        return best_stat

    def damage_multiplier(
        self,
        attacking_type: Union[str, pkmn_b.PokemonType, "SimulatedMove", pkmn_b.Move],
        has_gravity: bool = False
    ) -> float:
        if isinstance(attacking_type, (SimulatedMove, pkmn_b.Move)):
            raw_type = attacking_type.type
        else:
            raw_type = attacking_type
        return get_type_effectiveness(str(raw_type.name if isinstance(raw_type, pkmn_b.PokemonType) else raw_type), self.types, has_gravity=has_gravity)

    def calculate_effective_speed(
        self,
        weather: dict[str, int],
        fields: dict[str, int],
        side_conditions: dict[str, int]
    ) -> float:
        """Computes true dynamic speed taking into account stages, Tailwind, Paralysis, Scarf, and abilities."""
        if self.fainted:
            return 0.0

        raw_speed = float(self.stats.get("spe", 80))
        speed_boost = self.boosts.get("spe", 0)
        stage_multiplier = (2.0 + speed_boost) / 2.0 if speed_boost >= 0 else 2.0 / (2.0 - speed_boost)
        effective_speed = raw_speed * stage_multiplier

        # Tailwind
        if side_conditions.get("tailwind", 0) > 0:
            effective_speed *= 2.0

        # Paralysis (0.5x, unless Quick Feet)
        if self.status == "par" and self.ability != "quickfeet":
            effective_speed *= 0.5
        elif self.status is not None and self.ability == "quickfeet":
            effective_speed *= 1.5

        # Item modifiers & Unburden
        if self.item == "choicescarf":
            effective_speed *= 1.5
        elif self.item in ("ironball", "machobrace", "poweranklet", "powerband", "powerbelt", "powerbracer", "powerlens", "powerweight"):
            effective_speed *= 0.5
        elif self.item == "quickpowder" and self.species == "ditto":
            effective_speed *= 2.0
        elif self.ability == "unburden" and self.unburden_active and self.item in ("none", ""):
            effective_speed *= 2.0

        # Protosynthesis / Quark Drive speed boost (1.5x)
        if self.paradox_boost_stat == "spe" and self.ability in ("protosynthesis", "quarkdrive"):
            effective_speed *= 1.5

        # Slow Start (0.5x Speed for first 5 turns)
        if self.ability == "slowstart" and self.slow_start_turns > 0:
            effective_speed *= 0.5

        # Weather abilities
        if "raindance" in weather or "rain" in weather or "primordialsea" in weather:
            if self.ability == "swiftswim":
                effective_speed *= 2.0
        elif "sunnyday" in weather or "sun" in weather or "desolateland" in weather:
            if self.ability == "chlorophyll":
                effective_speed *= 2.0
        elif "sandstorm" in weather or "sand" in weather:
            if self.ability == "sandrush":
                effective_speed *= 2.0
        elif "snowscape" in weather or "snow" in weather or "hail" in weather:
            if self.ability == "slushrush":
                effective_speed *= 2.0

        # Electric Terrain
        if "electricterrain" in fields and self.ability == "surgesurfer":
            effective_speed *= 2.0

        return effective_speed

    def calculate_effective_attack(
        self,
        is_special: bool,
        weather: dict[str, int],
        fields: dict[str, int],
        ignore_boosts: bool = False,
        ignore_ability: bool = False,
        ignore_negative_boosts: bool = False,
        ignore_positive_boosts: bool = False,
    ) -> float:
        stat_name = "spa" if is_special else "atk"
        raw_attack = float(self.stats.get(stat_name, 80))
        attack_boost = 0 if ignore_boosts else self.boosts.get(stat_name, 0)
        if ignore_negative_boosts and attack_boost < 0:
            attack_boost = 0
        if ignore_positive_boosts and attack_boost > 0:
            attack_boost = 0
        attack_multiplier = (2.0 + attack_boost) / 2.0 if attack_boost >= 0 else 2.0 / (2.0 - attack_boost)
        eff_ability = "" if ignore_ability else self.ability

        is_sun = ("sunnyday" in weather or "sun" in weather or "desolateland" in weather or eff_ability == "megasol")

        if not is_special:
            if self.status == "brn" and eff_ability != "guts":
                attack_multiplier *= 0.5
            elif self.status is not None and eff_ability == "guts":
                attack_multiplier *= 1.5
            elif self.status in ("psn", "tox") and eff_ability == "toxicboost":
                attack_multiplier *= 1.5
            if self.item == "choiceband":
                attack_multiplier *= 1.5
            elif self.item == "thickclub" and self.species in ("cubone", "marowak", "marowakalola"):
                attack_multiplier *= 2.0
            elif self.item == "lightball" and self.species.startswith("pikachu"):
                attack_multiplier *= 2.0
            if eff_ability in ("hugepower", "purepower"):
                attack_multiplier *= 2.0
            elif eff_ability == "gorillatactics":
                attack_multiplier *= 1.5
            elif eff_ability == "hustle":
                attack_multiplier *= 1.5
            elif eff_ability == "defeatist" and self.current_hp_fraction <= 0.50:
                attack_multiplier *= 0.5
            elif eff_ability == "slowstart" and self.slow_start_turns > 0:
                attack_multiplier *= 0.5
            elif eff_ability == "orichalcumpulse" and is_sun:
                attack_multiplier *= 1.3333
            elif eff_ability in ("protosynthesis", "quarkdrive") and self.paradox_boost_stat == "atk":
                attack_multiplier *= 1.3
        else:
            if self.item == "choicespecs":
                attack_multiplier *= 1.5
            elif self.item == "lightball" and self.species.startswith("pikachu"):
                attack_multiplier *= 2.0
            elif self.item == "deepseatooth" and self.species == "clamperl":
                attack_multiplier *= 2.0
            if self.status == "brn" and eff_ability == "flareboost":
                attack_multiplier *= 1.5
            if eff_ability == "solarpower" and is_sun:
                attack_multiplier *= 1.5
            elif eff_ability == "defeatist" and self.current_hp_fraction <= 0.50:
                attack_multiplier *= 0.5
            elif eff_ability == "hadronengine" and "electricterrain" in fields:
                attack_multiplier *= 1.3333
            elif eff_ability in ("protosynthesis", "quarkdrive") and self.paradox_boost_stat == "spa":
                attack_multiplier *= 1.3

        return raw_attack * attack_multiplier

    def calculate_effective_defense(
        self,
        is_special: bool,
        weather: dict[str, int],
        fields: dict[str, int],
        ignore_boosts: bool = False,
        ignore_ability: bool = False,
        ignore_positive_boosts: bool = False,
        ignore_negative_boosts: bool = False,
    ) -> float:
        stat_name = "spd" if is_special else "def"
        raw_defense = float(self.stats.get(stat_name, 80))
        defense_boost = 0 if ignore_boosts else self.boosts.get(stat_name, 0)
        if ignore_positive_boosts and defense_boost > 0:
            defense_boost = 0
        if ignore_negative_boosts and defense_boost < 0:
            defense_boost = 0
        defense_multiplier = (2.0 + defense_boost) / 2.0 if defense_boost >= 0 else 2.0 / (2.0 - defense_boost)
        eff_ability = "" if ignore_ability else self.ability

        if is_special:
            if self.item == "assaultvest":
                defense_multiplier *= 1.5
            elif self.item == "deepseascale" and self.species == "clamperl":
                defense_multiplier *= 2.0
            if ("sandstorm" in weather or "sand" in weather) and "rock" in self.types:
                defense_multiplier *= 1.5
            if eff_ability == "icescales":
                defense_multiplier *= 2.0
            elif eff_ability in ("protosynthesis", "quarkdrive") and self.paradox_boost_stat == "spd":
                defense_multiplier *= 1.3
        else:
            if self.item == "metalpowder" and self.species == "ditto":
                defense_multiplier *= 1.5
            if ("snowscape" in weather or "snow" in weather or "hail" in weather) and "ice" in self.types:
                defense_multiplier *= 1.5
            if eff_ability == "furcoat":
                defense_multiplier *= 2.0
            elif eff_ability == "grasspelt" and "grassyterrain" in fields:
                defense_multiplier *= 1.5
            elif eff_ability == "marvelscale" and self.status is not None:
                defense_multiplier *= 1.5
            elif eff_ability in ("protosynthesis", "quarkdrive") and self.paradox_boost_stat == "def":
                defense_multiplier *= 1.3

        if self.item == "eviolite":
            defense_multiplier *= 1.5

        return raw_defense * defense_multiplier


def is_pokemon_grounded(
    pokemon: Optional[SimulatedPokemon],
    fields: dict[str, int],
    ignore_ability: bool = False,
) -> bool:
    """Checks if a SimulatedPokemon is grounded on the field."""
    if pokemon is None or pokemon.fainted:
        return False
    if "gravity" in fields or pokemon.item == "ironball":
        return True
    eff_ability = "" if ignore_ability else pokemon.ability
    if ("flying" in pokemon.types and not pokemon.roost_active) or eff_ability in ("levitate", "eelevate") or pokemon.item == "airballoon":
        return False
    return True


def get_effective_move_priority(move: SimulatedMove, attacker: SimulatedPokemon, fields: dict[str, int]) -> int:
    """Returns the effective priority of a move, accounting for Grassy Glide in Grassy Terrain and abilities."""
    prio = move.priority
    if move.id == "grassyglide" and "grassyterrain" in fields and is_pokemon_grounded(attacker, fields):
        prio += 1
    if attacker.ability == "prankster" and move.category == "status":
        prio += 1
    elif attacker.ability == "galewings" and move.type == "flying" and attacker.current_hp_fraction >= 1.0:
        prio += 1
    elif attacker.ability == "triage" and (
        "heal" in move.flags or move.drain > 0.0 or move.heal_fraction > 0.0 or move.id in (
            "drainingkiss", "gigadrain", "drainpunch", "hornleech", "leechlife",
            "matchagotcha", "oblivionwing", "paraboliccharge", "bitterblade",
            "floralhealing", "synthesis", "recover", "roost", "slackoff",
            "softboiled", "milkdrink", "moonlight", "morningsun", "lifedew", "wish",
            "healpulse", "purify", "rest", "shoreup", "strengthsap", "swallow", "lunarblessing", "junglehealing"
        )
    ):
        prio += 3
    return prio


def is_charge_move_instant(move_id: str, attacker: SimulatedPokemon, weather: dict[str, int]) -> bool:
    """Returns True if a two-turn charge move executes in a single turn (already charged, weather, or Power Herb)."""
    if attacker.charging_move == move_id:
        return True
    if move_id in ("solarbeam", "solarblade"):
        if any(w in weather for w in ("sunnyday", "desolateland", "sun")) or attacker.ability == "megasol":
            return True
    elif move_id == "electroshot":
        if any(w in weather for w in ("raindance", "primordialsea", "rain")) and attacker.ability != "megasol":
            return True
    if attacker.item == "powerherb":
        return True
    return False


class SimulatedBattleState:
    """
    Snapshot of battle state for Doubles (2 active per side).
    Supports high-speed deep cloning and forward simulation.
    """
    battle_format: str
    active_pokemon: list[Optional[SimulatedPokemon]]
    opponent_active_pokemon: list[Optional[SimulatedPokemon]]
    bench: list[SimulatedPokemon]
    opponent_bench: list[SimulatedPokemon]
    weather: dict[str, int]
    fields: dict[str, int]
    side_conditions: dict[str, int]
    opponent_side_conditions: dict[str, int]
    turn: int
    hit_counts: dict[str, int]
    force_switch: list[bool]
    trapped: list[bool]
    _raw_battle_reference: Optional[Any]
    redirect_target: dict[int, Optional[tuple[int, str]]]
    wide_guard: dict[int, bool]
    quick_guard: dict[int, bool]
    crafty_shield: dict[int, bool]

    mega_used: bool
    opponent_mega_used: bool
    unrevealed_opponent_count: int
    wish_slots: dict[tuple[int, int], tuple[int, float]]
    future_sight_slots: dict[tuple[int, int], tuple[int, str, SimulatedPokemon]]
    mat_block: dict[int, bool]
    ion_deluge: bool

    __slots__ = (
        "battle_format", "active_pokemon", "opponent_active_pokemon", "bench", "opponent_bench",
        "weather", "fields", "side_conditions", "opponent_side_conditions", "turn",
        "hit_counts", "force_switch", "trapped", "_raw_battle_reference",
        "redirect_target", "wide_guard", "quick_guard", "crafty_shield",
        "mega_used", "opponent_mega_used", "unrevealed_opponent_count",
        "wish_slots", "future_sight_slots", "mat_block", "ion_deluge"
    )

    def __init__(
        self,
        battle_format: str = "doubles",
        active_pokemon: Optional[list[Optional[SimulatedPokemon]]] = None,
        opponent_active_pokemon: Optional[list[Optional[SimulatedPokemon]]] = None,
        bench: Optional[list[SimulatedPokemon]] = None,
        opponent_bench: Optional[list[SimulatedPokemon]] = None,
        weather: Optional[dict[str, int]] = None,
        fields: Optional[dict[str, int]] = None,
        side_conditions: Optional[dict[str, int]] = None,
        opponent_side_conditions: Optional[dict[str, int]] = None,
        turn: int = 1,
        hit_counts: Optional[dict[str, int]] = None,
        unrevealed_opponent_count: int = 0,
    ):
        self.battle_format = "doubles"
        default_active: list[Optional[SimulatedPokemon]] = [None, None]
        self.active_pokemon = list(active_pokemon) if active_pokemon is not None else list(default_active)
        self.opponent_active_pokemon = list(opponent_active_pokemon) if opponent_active_pokemon is not None else list(default_active)
        self.bench = list(bench) if bench is not None else []
        self.opponent_bench = list(opponent_bench) if opponent_bench is not None else []
        self.weather = weather if weather is not None else {}
        self.fields = fields if fields is not None else {}
        self.side_conditions = side_conditions if side_conditions is not None else {}
        self.opponent_side_conditions = opponent_side_conditions if opponent_side_conditions is not None else {}
        self.turn = turn
        self.hit_counts = hit_counts if hit_counts is not None else {}
        self.force_switch = [False, False]
        self.trapped = [False, False]
        self._raw_battle_reference = None
        self.redirect_target = {1: None, 2: None}
        self.wide_guard = {1: False, 2: False}
        self.quick_guard = {1: False, 2: False}
        self.crafty_shield = {1: False, 2: False}
        self.mega_used = False
        self.opponent_mega_used = False
        self.unrevealed_opponent_count = unrevealed_opponent_count
        self.wish_slots = {}
        self.future_sight_slots = {}
        self.mat_block = {1: False, 2: False}
        self.ion_deluge = False

    def clone(self) -> "SimulatedBattleState":
        """Fast deep clone for game tree rollouts."""
        cloned_state = SimulatedBattleState.__new__(SimulatedBattleState)
        cloned_state.battle_format = self.battle_format
        cloned_state.active_pokemon = [pokemon.clone() if pokemon else None for pokemon in self.active_pokemon]
        cloned_state.opponent_active_pokemon = [pokemon.clone() if pokemon else None for pokemon in self.opponent_active_pokemon]
        cloned_state.bench = [pokemon.clone() for pokemon in self.bench]
        cloned_state.opponent_bench = [pokemon.clone() for pokemon in self.opponent_bench]
        cloned_state.weather = self.weather.copy()
        cloned_state.fields = self.fields.copy()
        cloned_state.side_conditions = self.side_conditions.copy()
        cloned_state.opponent_side_conditions = self.opponent_side_conditions.copy()
        cloned_state.turn = self.turn
        cloned_state.hit_counts = self.hit_counts.copy()
        cloned_state.force_switch = self.force_switch.copy()
        cloned_state.trapped = self.trapped.copy()
        cloned_state._raw_battle_reference = self._raw_battle_reference
        cloned_state.redirect_target = self.redirect_target.copy()
        cloned_state.wide_guard = self.wide_guard.copy()
        cloned_state.quick_guard = self.quick_guard.copy()
        cloned_state.crafty_shield = self.crafty_shield.copy()
        cloned_state.mega_used = self.mega_used
        cloned_state.opponent_mega_used = self.opponent_mega_used
        cloned_state.unrevealed_opponent_count = self.unrevealed_opponent_count
        cloned_state.wish_slots = self.wish_slots.copy()
        cloned_state.future_sight_slots = {
            k: (v[0], v[1], v[2].clone()) for k, v in self.future_sight_slots.items()
        }
        cloned_state.mat_block = self.mat_block.copy()
        cloned_state.ion_deluge = self.ion_deluge
        return cloned_state

    def all_team_pokemon(self) -> list[SimulatedPokemon]:
        return [pokemon for pokemon in self.active_pokemon if pokemon] + self.bench

    def all_opponent_pokemon(self) -> list[SimulatedPokemon]:
        return [pokemon for pokemon in self.opponent_active_pokemon if pokemon] + self.opponent_bench

    def alive_team_count(self) -> int:
        return sum(1 for pokemon in self.all_team_pokemon() if not pokemon.fainted)

    def alive_opponent_count(self) -> int:
        return sum(1 for pokemon in self.all_opponent_pokemon() if not pokemon.fainted) + self.unrevealed_opponent_count

    def is_finished(self) -> bool:
        return self.alive_team_count() == 0 or self.alive_opponent_count() == 0

    def winner(self) -> Optional[int]:
        team_alive_count = self.alive_team_count()
        opp_alive_count = self.alive_opponent_count()
        if team_alive_count == 0 and opp_alive_count == 0:
            return 0  # Tie
        if team_alive_count == 0:
            return 2  # Side 2 won
        if opp_alive_count == 0:
            return 1  # Side 1 won
        return None  # Battle ongoing

    def is_pokemon_trapped(self, pokemon: Optional[SimulatedPokemon]) -> bool:
        """Determines if a Pokemon is trapped and prohibited from switching."""
        if not pokemon or pokemon.fainted:
            return False
        if pokemon.charging_move is not None or pokemon.must_recharge or pokemon.locked_move is not None:
            return True
        if "ghost" in pokemon.types:
            return False
        if pokemon.item == "shedshell":
            return False
        if pokemon.is_trapped:
            return True

        opponents = self.opponent_active_pokemon if pokemon.side == 1 else self.active_pokemon
        for opp in opponents:
            if opp and not opp.fainted:
                if opp.ability == "shadowtag" and pokemon.ability != "shadowtag":
                    return True
                if opp.ability == "arenatrap":
                    if is_pokemon_grounded(pokemon, self.fields):
                        return True
                if opp.ability == "magnetpull" and "steel" in pokemon.types:
                    return True
        return False

    @classmethod
    def from_battle(cls, battle: Any, fill_unrevealed: bool = True) -> "SimulatedBattleState":
        """
        Constructs a high-fidelity SimulatedBattleState directly from a poke-env DoubleBattle.
        """
        state = cls(battle_format="doubles", turn=getattr(battle, "turn", 1))

        cur_turn = getattr(battle, "turn", 0)

        # Weather mapping (compute remaining turns from start_turn when available)
        if hasattr(battle, "weather") and isinstance(battle.weather, dict):
            for weather_condition, start_turn in battle.weather.items():
                elapsed = max(0, cur_turn - start_turn) if isinstance(start_turn, int) and cur_turn > 0 else 0
                state.weather[clean_id(weather_condition)] = max(1, 5 - elapsed)

        # Fields mapping (compute remaining turns from start_turn when available)
        if hasattr(battle, "fields") and isinstance(battle.fields, dict):
            for field_condition, start_turn in battle.fields.items():
                elapsed = max(0, cur_turn - start_turn) if isinstance(start_turn, int) and cur_turn > 0 else 0
                state.fields[clean_id(field_condition)] = max(1, 5 - elapsed)

        # Side conditions (compute remaining turns for timed conditions, or layer count for hazards)
        if hasattr(battle, "side_conditions") and isinstance(battle.side_conditions, dict):
            for side_condition, val in battle.side_conditions.items():
                cond_id = clean_id(side_condition)
                if cond_id in ("spikes", "toxicspikes", "stealthrock", "stickyweb") and isinstance(val, int) and val <= 3:
                    state.side_conditions[cond_id] = max(1, val)
                else:
                    max_t = 4 if cond_id == "tailwind" else 5
                    elapsed = max(0, cur_turn - val) if isinstance(val, int) and cur_turn > 0 else 0
                    state.side_conditions[cond_id] = max(1, max_t - elapsed)
        if hasattr(battle, "opponent_side_conditions") and isinstance(battle.opponent_side_conditions, dict):
            for side_condition, val in battle.opponent_side_conditions.items():
                cond_id = clean_id(side_condition)
                if cond_id in ("spikes", "toxicspikes", "stealthrock", "stickyweb") and isinstance(val, int) and val <= 3:
                    state.opponent_side_conditions[cond_id] = max(1, val)
                else:
                    max_t = 4 if cond_id == "tailwind" else 5
                    elapsed = max(0, cur_turn - val) if isinstance(val, int) and cur_turn > 0 else 0
                    state.opponent_side_conditions[cond_id] = max(1, max_t - elapsed)

        # Parse Side 1 (Self) - 2 active slots
        raw_active = battle.active_pokemon if isinstance(battle.active_pokemon, (list, tuple)) else [battle.active_pokemon]
        for slot_index in range(2):
            pokemon_object = raw_active[slot_index] if slot_index < len(raw_active) else None
            if pokemon_object and not pokemon_object.fainted:
                simulated_pokemon = cls._convert_poke_env_mon(
                    pokemon_object, side=1, slot=slot_index, fill_priors=False, battle=battle
                )
                state.active_pokemon[slot_index] = simulated_pokemon

        # Parse Side 1 Bench
        for pokemon_object in battle.team.values():
            if pokemon_object and not pokemon_object.fainted and pokemon_object not in raw_active:
                if getattr(pokemon_object, "selected_in_teampreview", True):
                    simulated_pokemon = cls._convert_poke_env_mon(
                        pokemon_object, side=1, slot=-1, fill_priors=False, battle=battle
                    )
                    state.bench.append(simulated_pokemon)

        # Parse Side 2 (Opponent) - 2 active slots
        raw_opponent_active = list(battle.opponent_active_pokemon) if isinstance(battle.opponent_active_pokemon, (list, tuple)) else [battle.opponent_active_pokemon]
        for slot_index in range(2):
            pokemon_object = raw_opponent_active[slot_index] if slot_index < len(raw_opponent_active) else None
            if pokemon_object and not getattr(pokemon_object, "fainted", False):
                simulated_pokemon = cls._convert_poke_env_mon(
                    pokemon_object, side=2, slot=slot_index, fill_priors=fill_unrevealed, battle=battle
                )
                state.opponent_active_pokemon[slot_index] = simulated_pokemon

        # Parse Side 2 Bench
        for pokemon_object in battle.opponent_team.values():
            if pokemon_object and not getattr(pokemon_object, "fainted", False) and pokemon_object not in raw_opponent_active:
                if getattr(pokemon_object, "revealed", True):
                    simulated_pokemon = cls._convert_poke_env_mon(
                        pokemon_object, side=2, slot=-1, fill_priors=fill_unrevealed, battle=battle
                    )
                    state.opponent_bench.append(simulated_pokemon)

        # Track unrevealed opponent back Pokemon in VGC (Bring 6, Pick 4)
        opp_fainted_count = sum(
            1 for p in battle.opponent_team.values()
            if p and getattr(p, "fainted", False)
        )
        opp_revealed_alive = sum(1 for p in state.all_opponent_pokemon() if not p.fainted)
        opp_brought_cap = min(4, max(len(battle.opponent_team), opp_revealed_alive + opp_fainted_count))
        state.unrevealed_opponent_count = max(0, opp_brought_cap - opp_fainted_count - opp_revealed_alive)

        # Direct hit counts for Rage Fist
        if hasattr(battle, "_hit_counts"):
            for identifier, count in battle._hit_counts.items():
                state.hit_counts[clean_id(identifier)] = count

        # Force switch flags from live battle
        if hasattr(battle, "force_switch") and isinstance(battle.force_switch, list):
            state.force_switch = [bool(flag) for flag in battle.force_switch[:2]]

        # Trapped flags from live battle
        if hasattr(battle, "trapped") and isinstance(battle.trapped, list):
            state.trapped = [bool(flag) for flag in battle.trapped[:2]]
            for slot_idx, is_tr in enumerate(state.trapped):
                if is_tr and slot_idx < len(state.active_pokemon):
                    active_pokemon = state.active_pokemon[slot_idx]
                    if active_pokemon is not None:
                        active_pokemon.is_trapped = True

        state.mega_used = any(p and p.is_mega for p in state.active_pokemon + state.bench)
        state.opponent_mega_used = any(p and p.is_mega for p in state.opponent_active_pokemon + state.opponent_bench)
        state._raw_battle_reference = battle
        return state

    @staticmethod
    def _convert_poke_env_mon(mon: Any, side: int, slot: int, fill_priors: bool, battle: Any) -> SimulatedPokemon:
        species_name = clean_id(mon.species)
        moves_dictionary = {}
        for move_id, move_object in getattr(mon, "moves", {}).items():
            moves_dictionary[clean_id(move_id)] = SimulatedMove.from_id(move_id)

        simulated_pokemon = SimulatedPokemon.from_species(
            species=species_name,
            side=side,
            slot=slot,
            revealed_moves=list(moves_dictionary.keys()),
            revealed_item=clean_id(getattr(mon, "item", None)),
            revealed_ability=clean_id(getattr(mon, "ability", None)),
            fill_champions_priors=fill_priors,
        )

        # Overwrite dynamic state from battle observation
        if getattr(mon, "current_hp_fraction", None) is not None:
            simulated_pokemon.current_hp = simulated_pokemon.max_hp * mon.current_hp_fraction
            simulated_pokemon.current_hp_fraction = mon.current_hp_fraction

        if getattr(mon, "status", None) is not None:
            simulated_pokemon.status = clean_id(mon.status)

        if getattr(mon, "boosts", None):
            for stat_name, boost_value in mon.boosts.items():
                simulated_pokemon.boosts[clean_id(stat_name)] = boost_value

        simulated_pokemon.protect_counter = getattr(mon, "protect_counter", 0)
        simulated_pokemon.is_protected = False
        simulated_pokemon.first_turn = getattr(mon, "first_turn", False)

        # Sync dynamic typing from battle observation (handles Double Shock, Soak, etc.)
        if hasattr(mon, "types") and mon.types:
            simulated_pokemon.types = tuple(
                clean_id(t.name) for t in mon.types
                if hasattr(t, "name") and t.name != "THREE_QUESTION_MARKS"
            )

        # Sync volatile effects from live battle (Taunt, Encore, Disable, Torment, Perish Song, Trapping, Throat Chop, Substitute, Leech Seed, Salt Cure, Yawn, Confusion)
        effects_dict = getattr(mon, "effects", {})
        if not isinstance(effects_dict, dict):
            effects_dict = getattr(mon, "_effects", {})
        if isinstance(effects_dict, dict):
            for eff, count in effects_dict.items():
                if eff == Effect.TAUNT:
                    simulated_pokemon.taunt_turns = max(1, 3 - count) if isinstance(count, int) else 3
                elif eff == Effect.ENCORE:
                    simulated_pokemon.encore_turns = max(1, 3 - count) if isinstance(count, int) else 3
                    if getattr(mon, "last_move", None):
                        simulated_pokemon.encore_move = clean_id(mon.last_move.id)
                elif eff == Effect.DISABLE:
                    simulated_pokemon.disable_turns = max(1, 4 - count) if isinstance(count, int) else 4
                    if getattr(mon, "last_move", None):
                        simulated_pokemon.disabled_move = clean_id(mon.last_move.id)
                elif eff == Effect.TORMENT:
                    simulated_pokemon.torment = True
                elif eff in (Effect.PERISH0, Effect.PERISH1, Effect.PERISH2, Effect.PERISH3):
                    if eff == Effect.PERISH3: simulated_pokemon.perish_turns = 3
                    elif eff == Effect.PERISH2: simulated_pokemon.perish_turns = 2
                    elif eff == Effect.PERISH1: simulated_pokemon.perish_turns = 1
                elif eff in (Effect.TRAPPED, Effect.BIND, Effect.WRAP, Effect.SAND_TOMB):
                    simulated_pokemon.is_trapped = True
                elif eff == Effect.IMPRISON:
                    simulated_pokemon.imprison = True
                elif eff == Effect.THROAT_CHOP:
                    simulated_pokemon.throat_chop_turns = max(1, 2 - count) if isinstance(count, int) else 2
                elif eff == Effect.SUBSTITUTE:
                    simulated_pokemon.substitute_hp = simulated_pokemon.max_hp * 0.25
                elif eff == Effect.LEECH_SEED:
                    simulated_pokemon.leech_seed = True
                elif eff == getattr(Effect, "SALT_CURE", None):
                    simulated_pokemon.salt_cure = True
                elif eff == getattr(Effect, "SYRUP_BOMB", None):
                    simulated_pokemon.syrup_bomb_turns = max(1, 3 - count) if isinstance(count, int) else 3
                elif eff == Effect.YAWN:
                    simulated_pokemon.yawn_turns = 1
                elif eff == Effect.CONFUSION:
                    simulated_pokemon.confusion_turns = max(1, 3 - count) if isinstance(count, int) else 2
                simulated_pokemon.effects[eff] = count

        if getattr(mon, "last_move", None):
            simulated_pokemon.last_move_used = clean_id(mon.last_move.id)
        if getattr(mon, "unburden_active", False):
            simulated_pokemon.unburden_active = True
        if getattr(mon, "preparing", False) and getattr(mon, "preparing_move", None):
            simulated_pokemon.charging_move = clean_id(mon.preparing_move.id)
            target = getattr(mon, "preparing_target", None)
            if target and hasattr(battle, "opponent_active_pokemon"):
                opp_active = battle.opponent_active_pokemon if side == 1 else battle.active_pokemon
                if target in opp_active:
                    simulated_pokemon.charging_target = opp_active.index(target)

        return simulated_pokemon

# ----------------- Action Model -----------------

class SimulatedAction:
    """Action representation: Move, Switch, or Pass."""
    __slots__ = ("type", "slot", "move_id", "target_idx", "switch_idx", "mega")

    def __init__(
        self,
        action_type: str,  # "move", "switch", "pass"
        slot: int = 0,
        move_id: Optional[str] = None,
        target_idx: int = 0,  # 0: left opponent, 1: right opponent, 2: partner/self
        switch_idx: int = 0,
        mega: bool = False,
    ):
        self.type = action_type
        self.slot = slot
        self.move_id = clean_id(move_id) if move_id else None
        self.target_idx = target_idx
        self.switch_idx = switch_idx
        self.mega = mega

    def __repr__(self) -> str:
        if self.type == "move":
            return f"Move({self.move_id}, target={self.target_idx}{', MEGA' if self.mega else ''})"
        elif self.type == "switch":
            return f"Switch(bench={self.switch_idx})"
        return "Pass"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimulatedAction):
            return False
        return (
            self.type == other.type
            and self.slot == other.slot
            and self.move_id == other.move_id
            and self.target_idx == other.target_idx
            and self.switch_idx == other.switch_idx
            and self.mega == other.mega
        )

    def __hash__(self) -> int:
        return hash((self.type, self.slot, self.move_id, self.target_idx, self.switch_idx, self.mega))

# ----------------- Field, Berry, Paradox & Hazard Helpers -----------------

def _is_weather_suppressed(state: SimulatedBattleState) -> bool:
    """Returns True if Cloud Nine or Air Lock is active on the field."""
    for mon in state.active_pokemon + state.opponent_active_pokemon:
        if mon and not mon.fainted and mon.ability in ("cloudnine", "airlock"):
            return True
    return False


def _get_effective_weather(state: SimulatedBattleState) -> dict[str, int]:
    """Returns effective weather dictionary (empty if Cloud Nine / Air Lock suppresses weather)."""
    if _is_weather_suppressed(state):
        return {}
    return state.weather


def _is_neutralizing_gas_active(state: SimulatedBattleState) -> bool:
    """Returns True if an active Pokemon on the field has Neutralizing Gas."""
    for mon in state.active_pokemon + state.opponent_active_pokemon:
        if mon and not mon.fainted and mon.ability == "neutralizinggas":
            return True
    return False


def _sync_neutralizing_gas(state: SimulatedBattleState) -> None:
    """Suppresses or restores suppressible abilities based on whether Neutralizing Gas is active."""
    gas_active = _is_neutralizing_gas_active(state)
    for mon in state.active_pokemon + state.opponent_active_pokemon:
        if not mon or mon.fainted:
            continue
        if gas_active:
            if mon.ability not in _UNSUPPRESSIBLE_ABILITIES:
                if mon.ability != "":
                    mon.base_ability = mon.ability
                mon.ability = ""
        else:
            if mon.ability == "" and mon.base_ability:
                mon.ability = mon.base_ability


def _is_unnerve_active_against(pokemon: SimulatedPokemon, state: SimulatedBattleState) -> bool:
    """Checks if an opposing active Pokemon has Unnerve or As One, preventing berry consumption."""
    opponents = state.opponent_active_pokemon if pokemon.side == 1 else state.active_pokemon
    return any(
        opp and not opp.fainted and opp.ability in ("unnerve", "asone", "asoneglastrier", "asonespectrier")
        for opp in opponents
    )


def _trigger_berry_effect(pokemon: SimulatedPokemon, berry_id: str, consume: bool = True) -> bool:
    """Applies the effect of `berry_id` on `pokemon`. Returns True if the berry triggered."""
    if pokemon.fainted:
        return False
    ripen_mult = 2.0 if pokemon.ability == "ripen" else 1.0

    # 1. Status-curing berries
    if berry_id in _STATUS_CURE_BERRIES:
        cured_list = _STATUS_CURE_BERRIES[berry_id]
        cured = False
        if pokemon.status in cured_list:
            pokemon.status = None
            pokemon.toxic_turns = 0
            pokemon.sleep_turns = 0
            cured = True
        if "confusion" in cured_list and pokemon.confusion_turns > 0:
            pokemon.confusion_turns = 0
            cured = True
        if cured:
            if consume:
                pokemon.consume_item()
            return True
        return False

    # 2. HP-healing berries
    if berry_id == "sitrusberry":
        pokemon.heal(pokemon.max_hp * 0.25 * ripen_mult)
        if consume:
            pokemon.consume_item()
        return True
    elif berry_id == "oranberry":
        pokemon.heal(10.0 * ripen_mult)
        if consume:
            pokemon.consume_item()
        return True
    elif berry_id in _PINCH_HEAL_BERRIES:
        pokemon.heal(pokemon.max_hp * (1.0 / 3.0) * ripen_mult)
        if consume:
            pokemon.consume_item()
        return True

    # 3. Stat-pinch berries
    if berry_id in _PINCH_STAT_BERRIES:
        stat_name, boost_amt = _PINCH_STAT_BERRIES[berry_id]
        pokemon.apply_boost(stat_name, int(boost_amt * ripen_mult))
        if consume:
            pokemon.consume_item()
        return True

    return False


def _check_status_berries(pokemon: SimulatedPokemon, state: SimulatedBattleState) -> None:
    """Checks and triggers Lum, Chesto, Cheri, Pecha, Rawst, Aspear, or Persim Berry if held."""
    if not pokemon or pokemon.fainted or pokemon.item not in _STATUS_CURE_BERRIES:
        return
    if _is_unnerve_active_against(pokemon, state):
        return
    _trigger_berry_effect(pokemon, pokemon.item, consume=True)


def _check_hp_berries(pokemon: SimulatedPokemon, state: SimulatedBattleState) -> None:
    """Checks and triggers Sitrus, Oran, Aguav/Figy/Iapapa/Mago/Wiki, or stat-pinch berries when HP drops."""
    if not pokemon or pokemon.fainted or not pokemon.item.endswith("berry"):
        return
    if _is_unnerve_active_against(pokemon, state):
        return
    hp_frac = pokemon.current_hp / pokemon.max_hp
    if pokemon.item in ("sitrusberry", "oranberry") and hp_frac <= 0.50:
        _trigger_berry_effect(pokemon, pokemon.item, consume=True)
    elif pokemon.item in _PINCH_HEAL_BERRIES or pokemon.item in _PINCH_STAT_BERRIES:
        threshold = 0.50 if pokemon.ability == "gluttony" else 0.25
        if hp_frac <= threshold:
            _trigger_berry_effect(pokemon, pokemon.item, consume=True)


def _check_paradox_abilities(state: SimulatedBattleState) -> None:
    """Updates Protosynthesis and Quark Drive stat boosts across all active Pokemon."""
    eff_weather = _get_effective_weather(state)
    is_sun = any(w in eff_weather for w in ("sunnyday", "desolateland", "sun"))
    is_elec_terrain = "electricterrain" in state.fields

    for mon in state.active_pokemon + state.opponent_active_pokemon:
        if not mon or mon.fainted:
            continue
        if mon.ability == "protosynthesis":
            if is_sun:
                if not mon.paradox_boost_stat:
                    mon.paradox_boost_stat = mon.highest_stat_name()
            elif mon.booster_energy_active:
                if not mon.paradox_boost_stat:
                    mon.paradox_boost_stat = mon.highest_stat_name()
            elif mon.item == "boosterenergy":
                mon.consume_item()
                mon.booster_energy_active = True
                mon.paradox_boost_stat = mon.highest_stat_name()
            else:
                mon.paradox_boost_stat = None
        elif mon.ability == "quarkdrive":
            if is_elec_terrain:
                if not mon.paradox_boost_stat:
                    mon.paradox_boost_stat = mon.highest_stat_name()
            elif mon.booster_energy_active:
                if not mon.paradox_boost_stat:
                    mon.paradox_boost_stat = mon.highest_stat_name()
            elif mon.item == "boosterenergy":
                mon.consume_item()
                mon.booster_energy_active = True
                mon.paradox_boost_stat = mon.highest_stat_name()
            else:
                mon.paradox_boost_stat = None


def _check_room_service(state: SimulatedBattleState) -> None:
    """Triggers Room Service (-1 Spe) on any active Pokemon holding it while Trick Room is active."""
    if "trickroom" not in state.fields:
        return
    for mon in state.active_pokemon + state.opponent_active_pokemon:
        if mon and not mon.fainted and mon.item == "roomservice":
            mon.apply_boost("spe", -1)
            mon.consume_item()
            mon.check_white_herb()


def _apply_entry_hazards(pokemon: SimulatedPokemon, state: SimulatedBattleState) -> None:
    """Applies Stealth Rock, Spikes, Toxic Spikes, and Sticky Web upon switch-in."""
    if not pokemon or pokemon.fainted:
        return
    side_conds = state.side_conditions if pokemon.side == 1 else state.opponent_side_conditions
    if not side_conds:
        return

    grounded = is_pokemon_grounded(pokemon, state.fields)
    has_boots = (pokemon.item == "heavydutyboots")

    # Grounded Poison-types absorb Toxic Spikes even if holding Heavy-Duty Boots
    if grounded and "toxicspikes" in side_conds and "poison" in pokemon.types:
        del side_conds["toxicspikes"]

    if has_boots:
        return

    if "stealthrock" in side_conds and pokemon.ability != "magicguard":
        rock_eff = get_type_effectiveness("rock", pokemon.types)
        pokemon.take_damage(pokemon.max_hp * 0.125 * rock_eff)
        if pokemon.fainted:
            return

    if grounded and "spikes" in side_conds and pokemon.ability != "magicguard":
        layers = side_conds["spikes"]
        spike_frac = 0.125 if layers == 1 else (1.0 / 6.0 if layers == 2 else 0.25)
        pokemon.take_damage(pokemon.max_hp * spike_frac)
        if pokemon.fainted:
            return

    if grounded and "toxicspikes" in side_conds and pokemon.status is None:
        if "steel" not in pokemon.types and "poison" not in pokemon.types:
            if "mistyterrain" not in state.fields and pokemon.ability not in ("immunity", "pastelveil", "purifyingsalt", "comatose", "goodasgold"):
                layers = side_conds["toxicspikes"]
                pokemon.status = "tox" if layers >= 2 else "psn"
                _check_status_berries(pokemon, state)

    if grounded and "stickyweb" in side_conds:
        _apply_external_stat_drops(pokemon, [("spe", -1)], source=None, state=state)

    _check_hp_berries(pokemon, state)


def _force_bench_switch(
    pokemon: SimulatedPokemon,
    state: SimulatedBattleState,
    baton_pass: bool = False,
    shed_tail_sub_hp: float = 0.0,
) -> bool:
    """Switches `pokemon` out for the first available non-fainted bench Pokemon (used by pivot moves, Red Card, Eject Button/Pack, Emergency Exit)."""
    if not pokemon or pokemon.fainted or pokemon.slot not in (0, 1):
        return False
    bench_list = state.bench if pokemon.side == 1 else state.opponent_bench
    for b_idx, b_mon in enumerate(bench_list):
        if b_mon and not b_mon.fainted:
            saved_boosts = pokemon.boosts.copy() if baton_pass else None
            saved_sub = pokemon.substitute_hp if baton_pass else shed_tail_sub_hp
            saved_seed = pokemon.leech_seed if baton_pass else False
            saved_seed_slot = pokemon.leech_seed_slot if baton_pass else 0
            saved_perish = pokemon.perish_turns if baton_pass else 0

            _execute_switches(
                state,
                [SimulatedAction("switch", slot=pokemon.slot, switch_idx=b_idx)],
                side=pokemon.side,
                ignore_trapping=True,
            )
            active_list = state.active_pokemon if pokemon.side == 1 else state.opponent_active_pokemon
            new_mon = active_list[b_mon.slot] if b_mon.slot in (0, 1) else None
            if new_mon and not new_mon.fainted:
                if saved_boosts is not None:
                    new_mon.boosts = saved_boosts
                if saved_sub > 0.0:
                    new_mon.substitute_hp = saved_sub
                if saved_seed:
                    new_mon.leech_seed = True
                    new_mon.leech_seed_slot = saved_seed_slot
                if saved_perish > 0:
                    new_mon.perish_turns = saved_perish
            return True
    return False

# ----------------- Forward Step Simulation -----------------

def step_simulation(
    state: SimulatedBattleState,
    action_player1: Union[SimulatedAction, tuple[SimulatedAction, ...], list[SimulatedAction]],
    action_player2: Union[SimulatedAction, tuple[SimulatedAction, ...], list[SimulatedAction]],
) -> SimulatedBattleState:
    """
    Advances the battle state by 1 full turn following competitive Pokemon battle mechanics:
    1. Switches happen first (+6 priority).
    2. Mega evolutions activate.
    3. Moves sort by priority bracket (+4 Protect, +3 Fake Out, +2 E-Speed, +1 Priority, 0 Normal, -7 TR)
    and dynamic effective speed within brackets (inverted if Trick Room active), with dynamic mid-turn speed recalculation.
    4. Moves execute with KO checks, Protect blocks, damage calculations, stat modifications.
    5. End of turn status damage, weather damage, passive recovery, field timer countdowns.
    """
    next_state = state.clone()
    next_state.turn += 1
    next_state.redirect_target = {1: None, 2: None}
    next_state.wide_guard = {1: False, 2: False}
    next_state.quick_guard = {1: False, 2: False}
    next_state.crafty_shield = {1: False, 2: False}
    next_state.mat_block = {1: False, 2: False}
    next_state.ion_deluge = False

    # Reset per-turn defensive shields and volatile turn trackers for all active Pokemon
    for active_pkmn in next_state.active_pokemon + next_state.opponent_active_pokemon:
        if active_pkmn:
            active_pkmn.is_protected = False
            active_pkmn.protect_variant = None
            active_pkmn.endure_active = False
            active_pkmn.helping_hand = False
            active_pkmn.just_flinched = False
            active_pkmn.took_damage_this_turn = False
            active_pkmn.moved_this_turn = False
            active_pkmn.switched_in_this_turn = False

    player1_actions = [action_player1] if isinstance(action_player1, SimulatedAction) else list(action_player1)
    player2_actions = [action_player2] if isinstance(action_player2, SimulatedAction) else list(action_player2)

    # Reset protect streak for slots that pass
    for action in player1_actions:
        if action.type == "pass" and action.slot < len(next_state.active_pokemon):
            pkmn = next_state.active_pokemon[action.slot]
            if pkmn:
                pkmn.protect_counter = 0
                pkmn.is_protected = False
                pkmn.protect_variant = None
    for action in player2_actions:
        if action.type == "pass" and action.slot < len(next_state.opponent_active_pokemon):
            pkmn = next_state.opponent_active_pokemon[action.slot]
            if pkmn:
                pkmn.protect_counter = 0
                pkmn.is_protected = False
                pkmn.protect_variant = None

    # 1. Switch Phase
    _execute_switches(next_state, player1_actions, side=1)
    _execute_switches(next_state, player2_actions, side=2)

    # 2. Mega Evolution Phase
    for action in player1_actions:
        if action.type == "move" and action.mega and action.slot < len(next_state.active_pokemon):
            _trigger_mega(next_state.active_pokemon[action.slot], next_state)
    for action in player2_actions:
        if action.type == "move" and action.mega and action.slot < len(next_state.opponent_active_pokemon):
            _trigger_mega(next_state.opponent_active_pokemon[action.slot], next_state)

    _sync_neutralizing_gas(next_state)
    _check_paradox_abilities(next_state)

    # 3. Collect Move Executions
    move_execution_queue: list[tuple[int, float, int, int, SimulatedAction]] = []

    for side_idx, side_actions, side_active, side_conds in (
        (1, player1_actions, next_state.active_pokemon, next_state.side_conditions),
        (2, player2_actions, next_state.opponent_active_pokemon, next_state.opponent_side_conditions),
    ):
        for action in side_actions:
            if action.slot < len(side_active):
                user_pokemon = side_active[action.slot]
                if user_pokemon and not user_pokemon.fainted:
                    if user_pokemon.charging_move:
                        action = SimulatedAction(
                            action_type="move",
                            slot=action.slot,
                            move_id=user_pokemon.charging_move,
                            target_idx=user_pokemon.charging_target,
                        )
                    elif user_pokemon.locked_move and user_pokemon.locked_move_turns > 0:
                        action = SimulatedAction(
                            action_type="move",
                            slot=action.slot,
                            move_id=user_pokemon.locked_move,
                            target_idx=action.target_idx,
                        )
            if action.type == "move" and action.move_id and action.slot < len(side_active):
                user_pokemon = side_active[action.slot]
                if user_pokemon and not user_pokemon.fainted:
                    if action.move_id == "struggle":
                        move_object = SimulatedMove(
                            move_id="struggle",
                            name="Struggle",
                            move_type="normal",
                            category="physical",
                            base_power=50.0,
                            accuracy=100.0,
                            priority=0,
                        )
                    elif action.move_id in user_pokemon.moves:
                        move_object = user_pokemon.moves[action.move_id]
                    elif action.move_id in (user_pokemon.charging_move, user_pokemon.locked_move):
                        move_object = SimulatedMove.from_id(action.move_id)
                    else:
                        continue
                    eff_weather = _get_effective_weather(next_state)
                    effective_speed = user_pokemon.calculate_effective_speed(
                        eff_weather, next_state.fields, side_conds
                    )
                    effective_priority = get_effective_move_priority(move_object, user_pokemon, next_state.fields)
                    move_execution_queue.append((effective_priority, effective_speed, side_idx, action.slot, action))

    def _resort_move_queue(queue: list[tuple[int, float, int, int, SimulatedAction]]) -> None:
        """Dynamically recalculates speed & priority mid-turn (Gen 8/9 VGC mechanics)."""
        eff_weather = _get_effective_weather(next_state)
        is_tr = "trickroom" in next_state.fields
        for idx, (prio, spd, s_idx, sl_idx, act) in enumerate(queue):
            # Preserve explicit After You (+100) / Quash (-100) priority overrides
            if prio >= 50 or prio <= -50:
                continue
            active_list = next_state.active_pokemon if s_idx == 1 else next_state.opponent_active_pokemon
            s_conds = next_state.side_conditions if s_idx == 1 else next_state.opponent_side_conditions
            mon = active_list[sl_idx] if sl_idx < len(active_list) else None
            if mon and not mon.fainted:
                new_spd = mon.calculate_effective_speed(eff_weather, next_state.fields, s_conds)
                mv = mon.moves.get(act.move_id) if act.move_id else None
                new_prio = get_effective_move_priority(mv, mon, next_state.fields) if mv else prio
                queue[idx] = (new_prio, new_spd, s_idx, sl_idx, act)

        def sort_key(item: tuple) -> tuple:
            p, sp, s_i, _, _ = item
            adj_sp = -sp if is_tr else sp
            return (p, adj_sp, -s_i)

        queue.sort(key=sort_key, reverse=True)

    # 4. Dynamic Move Execution Loop
    while move_execution_queue:
        _resort_move_queue(move_execution_queue)
        _, _, side_index, slot_index, action = move_execution_queue.pop(0)

        active_pokemon_list = next_state.active_pokemon if side_index == 1 else next_state.opponent_active_pokemon
        opponent_pokemon_list = next_state.opponent_active_pokemon if side_index == 1 else next_state.active_pokemon

        user_pokemon = active_pokemon_list[slot_index] if slot_index < len(active_pokemon_list) else None
        if not user_pokemon or user_pokemon.fainted:
            continue

        # Must recharge turn (Hyper Beam, Giga Impact, etc.)
        if user_pokemon.must_recharge:
            user_pokemon.must_recharge = False
            user_pokemon.moved_this_turn = True
            continue

        # Truant loafing turn
        if user_pokemon.ability == "truant":
            if user_pokemon.truant_resting:
                user_pokemon.truant_resting = False
                user_pokemon.moved_this_turn = True
                continue
            else:
                user_pokemon.truant_resting = True

        # Flinch check
        if user_pokemon.flinch:
            user_pokemon.flinch = False
            user_pokemon.just_flinched = True
            user_pokemon.charging_move = None
            user_pokemon.moved_this_turn = True
            if user_pokemon.ability == "steadfast":
                user_pokemon.apply_boost("spe", 1)
            continue

        # Sleep check (deterministic: wakes up after 2 turns of sleep; Sleep Talk / Snore work while asleep)
        if user_pokemon.status == "slp" and user_pokemon.ability != "comatose":
            _check_status_berries(user_pokemon, next_state)
            if user_pokemon.status == "slp":
                if user_pokemon.sleep_turns >= 2:
                    user_pokemon.status = None
                    user_pokemon.sleep_turns = 0
                else:
                    user_pokemon.sleep_turns += 1
                    if action.move_id not in ("sleeptalk", "snore"):
                        user_pokemon.charging_move = None
                        user_pokemon.moved_this_turn = True
                        continue

        # Freeze check (Fire-type thawing moves thaw user immediately)
        if user_pokemon.status == "frz":
            _check_status_berries(user_pokemon, next_state)
            if user_pokemon.status == "frz":
                if action.move_id in ("flareblitz", "fusionflare", "flamewheel", "sacredfire", "scald", "steameruption", "burnup", "pyroball", "scorchingSands", "scorchingsands", "matchagotcha"):
                    user_pokemon.status = None
                else:
                    user_pokemon.charging_move = None
                    user_pokemon.moved_this_turn = True
                    continue

        # Confusion decrement
        if user_pokemon.confusion_turns > 0:
            _check_status_berries(user_pokemon, next_state)
            if user_pokemon.confusion_turns > 0:
                user_pokemon.confusion_turns -= 1

        if not action.move_id:
            continue

        if action.move_id == "struggle":
            current_move = SimulatedMove(
                move_id="struggle",
                name="Struggle",
                move_type="normal",
                category="physical",
                base_power=50.0,
                accuracy=100.0,
                priority=0,
            )
        else:
            current_move = user_pokemon.moves.get(action.move_id)
            if not current_move and action.move_id in (user_pokemon.charging_move, user_pokemon.locked_move):
                current_move = SimulatedMove.from_id(action.move_id)
        if not current_move:
            continue

        # Check attacker volatile & item restrictions before charging or executing
        if _is_move_blocked_by_attacker_restrictions(next_state, user_pokemon, current_move):
            user_pokemon.charging_move = None
            user_pokemon.last_move_failed = True
            user_pokemon.moved_this_turn = True
            continue

        # Focus Punch fails if user took damage this turn before moving
        if current_move.id == "focuspunch" and user_pokemon.took_damage_this_turn:
            user_pokemon.last_move_failed = True
            user_pokemon.moved_this_turn = True
            continue

        # Protect & Endure handling
        if current_move.id in constants.PROTECT_MOVES or current_move.id == "endure":
            protect_streak = user_pokemon.protect_counter
            protect_probability = 1.0 / (3.0 ** protect_streak)
            if protect_probability > 0.5:
                if current_move.id == "endure":
                    user_pokemon.endure_active = True
                else:
                    user_pokemon.is_protected = True
                    user_pokemon.protect_variant = current_move.id
                user_pokemon.protect_counter += 1
                user_pokemon.last_move_failed = False
            else:
                user_pokemon.is_protected = False
                user_pokemon.protect_variant = None
                user_pokemon.endure_active = False
                user_pokemon.protect_counter = 0
                user_pokemon.last_move_failed = True
            user_pokemon.last_move_used = current_move.id
            user_pokemon.moved_this_turn = True
            continue
        else:
            # Any non-Protect move breaks the consecutive protect streak
            user_pokemon.is_protected = False
            user_pokemon.protect_variant = None
            user_pokemon.endure_active = False
            user_pokemon.protect_counter = 0

        # Two-Turn Charge Move Handling (Electro Shot, Solar Beam, Meteor Beam, Geomancy, etc.)
        if current_move.id in _TWO_TURN_CHARGE_MOVES or "charge" in current_move.flags:
            if user_pokemon.charging_move == current_move.id:
                # Turn 2: Charge complete, execute the move now
                user_pokemon.charging_move = None
            else:
                # Turn 1: Apply charge-phase boosts that occur when charging starts
                if current_move.id in ("electroshot", "meteorbeam"):
                    user_pokemon.apply_boost("spa", 1)
                elif current_move.id == "skullbash":
                    user_pokemon.apply_boost("def", 1)

                eff_weather = _get_effective_weather(next_state)
                is_sunny = (
                    any(w in eff_weather for w in ("sunnyday", "desolateland", "sun"))
                    or user_pokemon.ability == "megasol"
                )
                is_rainy = (
                    any(w in eff_weather for w in ("raindance", "primordialsea", "rain"))
                    and user_pokemon.ability != "megasol"
                )
                weather_instant = (
                    (current_move.id in ("solarbeam", "solarblade") and is_sunny)
                    or (current_move.id == "electroshot" and is_rainy)
                )
                if weather_instant:
                    pass  # Executes immediately on Turn 1 without consuming Power Herb
                elif user_pokemon.item == "powerherb":
                    user_pokemon.consume_item()  # Consumes Power Herb (triggers Unburden) and executes on Turn 1
                else:
                    user_pokemon.charging_move = current_move.id
                    user_pokemon.charging_target = action.target_idx
                    user_pokemon.last_move_used = current_move.id
                    user_pokemon.moved_this_turn = True
                    continue

        # Target Resolution
        targets = []
        partner = active_pokemon_list[1 - slot_index] if len(active_pokemon_list) > 1 else None

        # Check self-targeting / field-targeting moves
        is_self_move = current_move.id in (
            "swordsdance", "nastyplot", "calmmind", "dragondance", "bulkup", "quiverdance",
            "irondefense", "amnesia", "agility", "rockpolish", "autotomize", "shiftgear",
            "shellsmash", "bellydrum", "tidyup", "victorydance", "clangoroussoul", "geomancy",
            "coil", "growth", "workup", "cosmicpower", "defendorder", "cottonguard", "tailglow",
            "acidarmor", "barrier", "tailwind", "trickroom", "reflect", "lightscreen", "auroraveil",
            "sunnyday", "raindance", "sandstorm", "snowscape", "hail", "psychicterrain",
            "electricterrain", "grassyterrain", "mistyterrain", "haze", "recover", "roost",
            "slackoff", "softboiled", "milkdrink", "healorder", "wish", "synthesis", "moonlight",
            "morningsun", "shoreup", "rest", "lifedew", "howl", "wideguard", "quickguard",
            "craftyshield", "matblock", "followme", "ragepowder", "allyswitch", "gravity",
            "chillyreception", "destinybond", "conversion", "camouflage", "perishsong",
            "imprison", "substitute", "shedtail", "batonpass", "teleport", "focusenergy",
            "dragoncheer", "charge", "safeguard", "mist", "stealthrock", "spikes",
            "toxicspikes", "stickyweb", "courtchange", "lunarblessing", "junglehealing"
        ) or (current_move.id == "curse" and "ghost" not in user_pokemon.types)

        is_expanding_force_spread = (
            current_move.id == "expandingforce"
            and "psychicterrain" in next_state.fields
            and is_pokemon_grounded(user_pokemon, next_state.fields)
        )

        if is_self_move:
            targets = [user_pokemon]
        elif action.move_id == "struggle":
            opp = opponent_pokemon_list[action.target_idx] if action.target_idx < len(opponent_pokemon_list) else None
            if not opp or opp.fainted:
                for o in opponent_pokemon_list:
                    if o and not o.fainted:
                        opp = o
                        break
            if opp:
                targets = [opp]
            else:
                continue
        elif action.target_idx == 2:
            # Explicit ally target
            if partner and not partner.fainted:
                targets = [partner]
            else:
                user_pokemon.last_move_failed = True
                user_pokemon.moved_this_turn = True
                continue
        elif current_move.is_spread or is_expanding_force_spread:
            targets = [opp for opp in opponent_pokemon_list if opp and not opp.fainted]
            if current_move.id in ("earthquake", "surf", "discharge", "bulldoze", "explosion", "selfdestruct", "boomburst", "paraboliccharge", "sludgewave", "lavaPlume", "lavaplume", "petalblizzard", "magnitude", "searingshot", "teeterdance") and partner and not partner.fainted:
                targets.append(partner)
        else:
            chosen_opponent = opponent_pokemon_list[action.target_idx] if action.target_idx < len(opponent_pokemon_list) else None
            if chosen_opponent and not chosen_opponent.fainted:
                targets = [chosen_opponent]
            else:
                # Retarget to remaining living opponent in doubles
                living_opponents = [opp for opp in opponent_pokemon_list if opp and not opp.fainted]
                if living_opponents:
                    targets = [living_opponents[0]]

            # Redirection check (Follow Me / Rage Powder / Storm Drain / Lightning Rod)
            # Bypassed by Stalwart, Propeller Tail, or moves with tracksTarget (Snipe Shot)
            ignores_redirection = (
                user_pokemon.ability in ("stalwart", "propellertail")
                or current_move.tracks_target
                or current_move.id == "snipeshot"
            )
            if not ignores_redirection and targets:
                target_side = 2 if side_index == 1 else 1
                redir_info = next_state.redirect_target.get(target_side)
                if redir_info is not None:
                    redir_slot, redir_move_id = redir_info
                    redirector = opponent_pokemon_list[redir_slot] if redir_slot < len(opponent_pokemon_list) else None
                    if redirector and not redirector.fainted:
                        is_rage_powder = (redir_move_id == "ragepowder")
                        immune_to_powder = (
                            is_rage_powder and (
                                "grass" in user_pokemon.types
                                or user_pokemon.item == "safetygoggles"
                                or user_pokemon.ability == "overcoat"
                            )
                        )
                        if not immune_to_powder:
                            targets = [redirector]
                else:
                    # Storm Drain (Water) & Lightning Rod (Electric) single-target redirection
                    eff_move_type = current_move.type
                    if eff_move_type == "normal":
                        if user_pokemon.ability == "pixilate": eff_move_type = "fairy"
                        elif user_pokemon.ability == "aerilate": eff_move_type = "flying"
                        elif user_pokemon.ability == "refrigerate": eff_move_type = "ice"
                        elif user_pokemon.ability == "galvanize": eff_move_type = "electric"
                        elif user_pokemon.ability == "dragonize": eff_move_type = "dragon"
                    elif user_pokemon.ability == "liquidvoice" and ("sound" in current_move.flags or current_move.id in _SOUND_MOVES):
                        eff_move_type = "water"

                    if eff_move_type in ("water", "electric"):
                        draw_ability = "stormdrain" if eff_move_type == "water" else "lightningrod"
                        for candidate_mon in ([partner] if partner else []) + opponent_pokemon_list:
                            if candidate_mon and not candidate_mon.fainted and candidate_mon != user_pokemon and candidate_mon.ability == draw_ability:
                                if candidate_mon not in targets:
                                    targets = [candidate_mon]
                                break

        valid_targets = [target for target in targets if target is not None]
        if not valid_targets:
            user_pokemon.last_move_failed = True
            user_pokemon.moved_this_turn = True
            continue

        prev_last_move = user_pokemon.last_move_used
        any_target_hit = False
        for target_pokemon in valid_targets:
            hit_resolved = _execute_single_move(
                next_state,
                user_pokemon,
                target_pokemon,
                current_move,
                is_spread=len(valid_targets) > 1,
                move_execution_queue=move_execution_queue
            )
            if hit_resolved:
                any_target_hit = True

        user_pokemon.moved_this_turn = True
        user_pokemon.last_move_failed = not any_target_hit

        if any_target_hit:
            if prev_last_move == current_move.id:
                user_pokemon.consecutive_move_uses += 1
            else:
                user_pokemon.consecutive_move_uses = 1

            # Throat Spray: +1 SpA after using a sound move
            if ("sound" in current_move.flags or current_move.id in _SOUND_MOVES) and user_pokemon.item == "throatspray" and not user_pokemon.fainted:
                user_pokemon.apply_boost("spa", 1)
                user_pokemon.consume_item()

            # Recharge moves (Hyper Beam, Giga Impact, etc.)
            if current_move.id in _RECHARGE_MOVES and not user_pokemon.fainted:
                user_pokemon.must_recharge = True

            # Rampage moves (Outrage, Thrash, Petal Dance, Raging Fury)
            if current_move.id in _RAMPAGE_MOVES and not user_pokemon.fainted:
                if user_pokemon.locked_move == current_move.id:
                    user_pokemon.locked_move_turns -= 1
                    if user_pokemon.locked_move_turns <= 0:
                        user_pokemon.locked_move = None
                        if user_pokemon.ability != "owntempo":
                            user_pokemon.confusion_turns = 2
                            _check_status_berries(user_pokemon, next_state)
                else:
                    user_pokemon.locked_move = current_move.id
                    user_pokemon.locked_move_turns = 1
        else:
            user_pokemon.consecutive_move_uses = 0
            if user_pokemon.locked_move == current_move.id:
                user_pokemon.locked_move = None
                user_pokemon.locked_move_turns = 0

        if current_move.category in ("physical", "special"):
            _apply_attacker_post_move_effects(user_pokemon, current_move, any_target_hit, next_state)

        # Pivot moves that switch the user out after hitting (U-turn, Volt Switch, Flip Turn, Parting Shot)
        if any_target_hit and not user_pokemon.fainted and current_move.id in ("uturn", "voltswitch", "flipturn", "partingshot"):
            _force_bench_switch(user_pokemon, next_state)

    # 5. End of Turn Effects
    _execute_end_of_turn(next_state)

    return next_state


def _is_move_blocked_by_attacker_restrictions(
    state: SimulatedBattleState,
    attacker: SimulatedPokemon,
    move: SimulatedMove,
) -> bool:
    """Checks if the attacker is prevented from using `move` due to volatile or item restrictions."""
    if move.id == "struggle":
        return False
    if attacker.taunt_turns > 0 and move.category == "status":
        return True
    if attacker.item == "assaultvest" and move.category == "status":
        return True
    if attacker.disable_turns > 0 and move.id == attacker.disabled_move:
        return True
    if attacker.encore_turns > 0 and attacker.encore_move and move.id != attacker.encore_move:
        return True
    if attacker.torment and attacker.last_move_used and move.id == attacker.last_move_used:
        return True
    if attacker.throat_chop_turns > 0 and ("sound" in move.flags or move.id in _SOUND_MOVES):
        return True
    if attacker.item in ("choiceband", "choicespecs", "choicescarf") or attacker.ability == "gorillatactics":
        if attacker.last_move_used and attacker.last_move_used in attacker.moves and move.id != attacker.last_move_used:
            return True

    # Imprison restriction: cannot use any move that an active opposing Imprison user knows
    opponents = state.opponent_active_pokemon if attacker.side == 1 else state.active_pokemon
    for opp in opponents:
        if opp and not opp.fainted and opp.imprison and move.id in opp.moves:
            return True

    return False


def _apply_attacker_post_move_effects(
    attacker: SimulatedPokemon,
    move: SimulatedMove,
    any_target_hit: bool,
    state: Optional[SimulatedBattleState] = None,
) -> None:
    """Applies once-per-move-use attacker recoil, self-stat drops, and self-stat boosts after all targets are hit."""
    # Self-destructing / fixed max-HP recoil moves that sacrifice HP even if blocked by Protect
    if move.id in ("mistyexplosion", "explosion", "selfdestruct") and not attacker.fainted:
        attacker.take_damage(attacker.current_hp)
        return
    if move.id in ("mindblown", "steelbeam") and not attacker.fainted:
        if attacker.ability != "magicguard":
            attacker.take_damage(attacker.max_hp * 0.50)

    if not any_target_hit or attacker.fainted:
        return

    # Chloroblast (loses 50% max HP on successful hit unless Magic Guard)
    if move.self_destruct_recoil > 0.0 and move.id not in ("mindblown", "steelbeam"):
        if attacker.ability != "magicguard":
            attacker.take_damage(attacker.max_hp * move.self_destruct_recoil)

    # Struggle recoil (25% of max HP once per use)
    if move.id == "struggle" and not attacker.fainted:
        attacker.take_damage(attacker.max_hp * 0.25)

    # Life Orb recoil (10% of max HP once per damaging move use, unless Sheer Force boosted the move or Magic Guard)
    if attacker.item == "lifeorb" and not attacker.fainted and attacker.ability != "magicguard":
        if not (attacker.ability == "sheerforce" and bool(move.secondary)):
            attacker.take_damage(attacker.max_hp * 0.10)

    if attacker.fainted:
        return

    if state is not None:
        _check_hp_berries(attacker, state)

    # Glaive Rush vulnerability on user until next turn
    if move.id == "glaiverush":
        attacker.glaive_rush_vulnerable = True

    # Secondary stat boosts on attacker (applied once per move use, unless Sheer Force suppresses secondary)
    if attacker.ability != "sheerforce":
        if move.id == "torchsong":
            attacker.apply_boost("spa", 1)
        elif move.id in ("aquastep", "trailblaze", "flamecharge", "esperwing", "aurawheel", "rapidspin"):
            attacker.apply_boost("spe", 1)
        elif move.id in ("poweruppunch", "fellstinger", "metalclaw", "meteorMash", "meteormash"):
            attacker.apply_boost("atk", 1)
        elif move.id in ("fierydance", "chargebeam"):
            attacker.apply_boost("spa", 1)
        elif move.id == "diamondstorm":
            attacker.apply_boost("def", 2)
        elif move.id == "clangoroussoulblaze":
            for stat_key in ("atk", "def", "spa", "spd", "spe"):
                attacker.apply_boost(stat_key, 1)

    # Self stat drops from powerful attacks (applied once after all targets are hit; NOT suppressed by Sheer Force)
    had_self_drop = False
    if move.id in ("closecombat", "headlongrush", "armorcannon", "dragonascent"):
        attacker.apply_boost("def", -1)
        attacker.apply_boost("spd", -1)
        had_self_drop = True
    elif move.id in ("dracometeor", "overheat", "leafstorm", "fleurcannon", "psychoboost"):
        attacker.apply_boost("spa", -2)
        had_self_drop = True
    elif move.id == "makeitrain":
        attacker.apply_boost("spa", -1)
        had_self_drop = True
    elif move.id == "superpower":
        attacker.apply_boost("atk", -1)
        attacker.apply_boost("def", -1)
        had_self_drop = True
    elif move.id == "vcreate":
        attacker.apply_boost("def", -1)
        attacker.apply_boost("spd", -1)
        attacker.apply_boost("spe", -1)
        had_self_drop = True
    elif move.id in ("hammerarm", "icehammer"):
        attacker.apply_boost("spe", -1)
        had_self_drop = True
    elif move.id == "spinout":
        attacker.apply_boost("spe", -2)
        had_self_drop = True
    elif move.id in ("clangingscales", "hyperspacefury"):
        attacker.apply_boost("def", -1)
        had_self_drop = True
    elif move.id == "scaleshot":
        attacker.apply_boost("def", -1)
        attacker.apply_boost("spe", 1)
        had_self_drop = True

    attacker.check_white_herb()
    if had_self_drop and attacker.item == "ejectpack" and state is not None and attacker.ability != "contrary":
        attacker.consume_item()
        _force_bench_switch(attacker, state)


def _check_terrain_seeds(state: SimulatedBattleState) -> None:
    """Checks if any active Pokemon holds a Terrain Seed matching the active terrain and consumes it."""
    for mon in state.active_pokemon + state.opponent_active_pokemon:
        if not mon or mon.fainted:
            continue
        if "psychicterrain" in state.fields and mon.item == "psychicseed":
            mon.apply_boost("spd", 1)
            mon.consume_item()
        elif "grassyterrain" in state.fields and mon.item == "grassyseed":
            mon.apply_boost("def", 1)
            mon.consume_item()
        elif "electricterrain" in state.fields and mon.item == "electricseed":
            mon.apply_boost("def", 1)
            mon.consume_item()
        elif "mistyterrain" in state.fields and mon.item == "mistyseed":
            mon.apply_boost("spd", 1)
            mon.consume_item()
    _check_paradox_abilities(state)


def _set_terrain(state: SimulatedBattleState, terrain_id: str, turns: int = 5) -> None:
    """Sets the active terrain (replacing any existing terrain) and checks Terrain Seeds and Quark Drive."""
    for existing in ("electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"):
        state.fields.pop(existing, None)
    state.fields[terrain_id] = turns
    _check_terrain_seeds(state)


def _apply_external_stat_drops(
    target: SimulatedPokemon,
    drops: list[tuple[str, int]],
    source: Optional[SimulatedPokemon],
    state: SimulatedBattleState,
    is_intimidate: bool = False,
    ignore_ability: bool = False,
    is_reflected: bool = False,
) -> bool:
    """
    Applies one or more negative stat stage changes caused by another Pokemon (`source`) to `target`.
    Handles Intimidate-specific immunities (`guarddog`, `innerfocus`, `oblivious`, `owntempo`, `scrappy`),
    `flowerveil`, `mirrorarmor`, `clearamulet`, `clearbody`/`whitesmoke`/`fullmetalbody`,
    `hypercutter`, `bigpecks`, `keeneye`/`illuminate`/`mindseye`, `contrary`/`simple`,
    `defiant`/`competitive` (+2 per stat lowered by an opponent), `rattled`/`adrenalineorb` (Intimidate),
    `whiteherb`, and `ejectpack`.
    Returns True if at least one stat stage on `target` was modified.
    """
    if target.fainted:
        return False

    # Substitute blocks external stat drops (unless source has Infiltrator)
    if target.substitute_hp > 0.0 and not is_intimidate and (source is None or source.ability != "infiltrator"):
        return False

    # Mist blocks external stat drops from opponents
    target_side_conds = state.side_conditions if target.side == 1 else state.opponent_side_conditions
    if "mist" in target_side_conds and (source is None or source.side != target.side and source.ability != "infiltrator"):
        return False

    unignorable = ("fullmetalbody", "shadowshield", "prismarmor", "comatose")
    target_ability = "" if (ignore_ability and target.ability not in unignorable) else target.ability

    # 1. Intimidate-specific immunities and Guard Dog
    if is_intimidate:
        if target_ability == "guarddog":
            before_atk = target.boosts.get("atk", 0)
            target.apply_boost("atk", 1)
            return target.boosts.get("atk", 0) != before_atk
        if target_ability in ("innerfocus", "oblivious", "owntempo", "scrappy"):
            return False

    # 2. Flower Veil (protects Grass-type Pokemon on the Flower Veil user's side)
    if "grass" in target.types:
        if target_ability == "flowerveil":
            return False
        ally_list = state.active_pokemon if target.side == 1 else state.opponent_active_pokemon
        if not ignore_ability and any(
            ally and not ally.fainted and ally != target and ally.ability == "flowerveil"
            for ally in ally_list
        ):
            return False

    # 3. Mirror Armor (reflects external stat drops back onto source)
    if target_ability == "mirrorarmor" and not is_reflected:
        if source is not None and not source.fainted and source != target:
            _apply_external_stat_drops(
                source,
                drops,
                source=target,
                state=state,
                is_intimidate=False,
                ignore_ability=False,
                is_reflected=True,
            )
        return False

    # 4. Full-stat-drop immunities (Clear Amulet, Clear Body, White Smoke, Full Metal Body)
    if target_ability != "contrary":
        if target.item == "clearamulet" or target_ability in ("clearbody", "whitesmoke", "fullmetalbody"):
            return False

    # 5. Apply each individual stat drop and trigger Defiant / Competitive per stat lowered by foe
    is_foe_drop = (source is None or source.side != target.side)
    any_dropped = False
    any_changed = False

    for stat_name, stages in drops:
        if stages >= 0 or stat_name not in target.boosts:
            continue
        if target_ability != "contrary":
            if stat_name == "atk" and target_ability == "hypercutter":
                continue
            if stat_name == "def" and target_ability == "bigpecks":
                continue
            if stat_name == "accuracy" and target_ability in ("keeneye", "illuminate", "mindseye"):
                continue

        before_stage = target.boosts.get(stat_name, 0)
        if ignore_ability and target.ability in ("contrary", "simple"):
            target.boosts[stat_name] = max(-6, min(6, before_stage + stages))
        else:
            target.apply_boost(stat_name, stages)
        after_stage = target.boosts.get(stat_name, 0)

        if after_stage != before_stage:
            any_changed = True
        if after_stage < before_stage:
            any_dropped = True
            if is_foe_drop:
                if target.ability == "defiant":
                    target.apply_boost("atk", 2)
                elif target.ability == "competitive":
                    target.apply_boost("spa", 2)

    # 6. Post-Intimidate triggers (Rattled, Adrenaline Orb) & White Herb / Eject Pack
    if is_intimidate and any_dropped:
        if target.ability == "rattled":
            target.apply_boost("spe", 1)
        if target.item == "adrenalineorb" and target.boosts.get("spe", 0) < 6:
            target.apply_boost("spe", 1)
            target.consume_item()

    if any_dropped:
        target.check_white_herb()
        if target.item == "ejectpack":
            target.consume_item()
            _force_bench_switch(target, state)

    return any_changed


def _trigger_entry_abilities(pokemon: Optional[SimulatedPokemon], state: SimulatedBattleState) -> None:
    """Triggers switch-in / Mega Evolution entry abilities (Intimidate, Surges, Weather, Hospitality, Download, etc.)."""
    if not pokemon or pokemon.fainted:
        return
    _sync_neutralizing_gas(state)
    opponent_pokemon_list = state.opponent_active_pokemon if pokemon.side == 1 else state.active_pokemon
    ally_pokemon_list = state.active_pokemon if pokemon.side == 1 else state.opponent_active_pokemon
    partner = next((p for p in ally_pokemon_list if p is not None and p != pokemon and not p.fainted), None)
    ability = pokemon.ability

    # Trace: copies ability of first valid opposing active Pokemon on entry
    if ability == "trace":
        for opp in opponent_pokemon_list:
            if opp and not opp.fainted and opp.ability and opp.ability not in _UNSUPPRESSIBLE_ABILITIES and opp.ability != "trace":
                pokemon.ability = opp.ability
                ability = pokemon.ability
                break

    # Imposter: transforms into the opposite slot's Pokemon
    if ability == "imposter":
        opp_target = opponent_pokemon_list[pokemon.slot] if pokemon.slot < len(opponent_pokemon_list) else None
        if not opp_target or opp_target.fainted:
            opp_target = next((o for o in opponent_pokemon_list if o and not o.fainted), None)
        if opp_target and opp_target.substitute_hp <= 0.0:
            pokemon.types = opp_target.types
            pokemon.base_types = opp_target.base_types
            for st_key in ("atk", "def", "spa", "spd", "spe"):
                pokemon.stats[st_key] = opp_target.stats[st_key]
            pokemon.boosts = opp_target.boosts.copy()
            pokemon.moves = {m_id: m_obj.clone() for m_id, m_obj in opp_target.moves.items()}
            pokemon.weight = opp_target.weight
            if opp_target.ability not in _UNSUPPRESSIBLE_ABILITIES and opp_target.ability != "imposter":
                pokemon.ability = opp_target.ability
                ability = pokemon.ability

    # 1. Intimidate & Supersweet Syrup
    if ability == "intimidate":
        for opponent_pokemon in opponent_pokemon_list:
            if opponent_pokemon and not opponent_pokemon.fainted:
                _apply_external_stat_drops(
                    opponent_pokemon,
                    [("atk", -1)],
                    source=pokemon,
                    state=state,
                    is_intimidate=True,
                )
    elif ability == "supersweetsyrup" and not pokemon.once_per_battle_ability_used:
        pokemon.once_per_battle_ability_used = True
        for opponent_pokemon in opponent_pokemon_list:
            if opponent_pokemon and not opponent_pokemon.fainted:
                _apply_external_stat_drops(
                    opponent_pokemon,
                    [("evasion", -1)],
                    source=pokemon,
                    state=state,
                )

    # 1b. Once-per-battle and stat-boosting entry abilities (Intrepid Sword, Dauntless Shield, Embody Aspect, Download)
    if ability == "intrepidsword" and not pokemon.once_per_battle_ability_used:
        pokemon.once_per_battle_ability_used = True
        pokemon.apply_boost("atk", 1)
    elif ability == "dauntlessshield" and not pokemon.once_per_battle_ability_used:
        pokemon.once_per_battle_ability_used = True
        pokemon.apply_boost("def", 1)
    elif ability.startswith("embodyaspect") and not pokemon.once_per_battle_ability_used:
        pokemon.once_per_battle_ability_used = True
        if "hearthflame" in pokemon.species or pokemon.item == "hearthflamemask":
            pokemon.apply_boost("atk", 1)
        elif "wellspring" in pokemon.species or pokemon.item == "wellspringmask":
            pokemon.apply_boost("spd", 1)
        elif "cornerstone" in pokemon.species or pokemon.item == "cornerstonemask":
            pokemon.apply_boost("def", 1)
        else:
            pokemon.apply_boost("spe", 1)
    elif ability == "download":
        living_opps = [o for o in opponent_pokemon_list if o and not o.fainted]
        if living_opps:
            avg_def = sum(o.calculate_effective_defense(False, state.weather, state.fields) for o in living_opps) / len(living_opps)
            avg_spd = sum(o.calculate_effective_defense(True, state.weather, state.fields) for o in living_opps) / len(living_opps)
            pokemon.apply_boost("atk" if avg_def < avg_spd else "spa", 1)

    # 1c. Partner & Field support entry abilities (Hospitality, Curious Medicine, Screen Cleaner, Wind Rider)
    if ability == "hospitality" and partner and not partner.fainted:
        partner.heal(partner.max_hp * 0.25)
    elif ability == "curiousmedicine" and partner and not partner.fainted:
        partner.clear_boosts()
    elif ability == "screencleaner":
        for cond_dict in (state.side_conditions, state.opponent_side_conditions):
            cond_dict.pop("reflect", None)
            cond_dict.pop("lightscreen", None)
            cond_dict.pop("auroraveil", None)
    elif ability == "windrider":
        own_side_conds = state.side_conditions if pokemon.side == 1 else state.opponent_side_conditions
        if own_side_conds.get("tailwind", 0) > 0:
            pokemon.apply_boost("atk", 1)

    # 2. Weather-setting abilities
    has_primal_weather = ("desolateland" in state.weather or "primordialsea" in state.weather or "deltastream" in state.weather)
    if ability == "desolateland":
        state.weather = {"desolateland": 5}
    elif ability == "primordialsea":
        state.weather = {"primordialsea": 5}
    elif ability == "deltastream":
        state.weather = {"deltastream": 5}
    elif ability == "teraformzero":
        state.weather.clear()
        for existing in ("electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"):
            state.fields.pop(existing, None)
    elif not has_primal_weather:
        if ability in ("drought", "orichalcumpulse"):
            state.weather = {"sunnyday": 8 if pokemon.item == "heatrock" else 5}
        elif ability == "drizzle":
            state.weather = {"raindance": 8 if pokemon.item == "damprock" else 5}
        elif ability == "sandstream":
            state.weather = {"sandstorm": 8 if pokemon.item == "smoothrock" else 5}
        elif ability == "snowwarning":
            state.weather = {"snowscape": 8 if pokemon.item == "icyrock" else 5}
            for mon in state.active_pokemon + state.opponent_active_pokemon:
                if mon and not mon.fainted and mon.ability == "iceface" and mon.ice_face_busted:
                    mon.ice_face_busted = False

    if ability == "iceface" and pokemon.ice_face_busted and any(w in state.weather for w in ("snowscape", "hail", "snow")):
        pokemon.ice_face_busted = False

    # 3. Terrain-setting abilities (Surges & Hadron Engine)
    ext_turns = 8 if pokemon.item == "terrainextender" else 5
    if ability == "psychicsurge":
        _set_terrain(state, "psychicterrain", ext_turns)
    elif ability == "grassysurge":
        _set_terrain(state, "grassyterrain", ext_turns)
    elif ability in ("electricsurge", "hadronengine"):
        _set_terrain(state, "electricterrain", ext_turns)
    elif ability == "mistysurge":
        _set_terrain(state, "mistyterrain", ext_turns)
    else:
        _check_terrain_seeds(state)

    _check_paradox_abilities(state)
    _check_room_service(state)


def _execute_switches(
    state: SimulatedBattleState,
    actions: list[SimulatedAction],
    side: int,
    ignore_trapping: bool = False,
) -> None:
    active_pokemon_list = state.active_pokemon if side == 1 else state.opponent_active_pokemon
    bench_pokemon_list = state.bench if side == 1 else state.opponent_bench

    for action in actions:
        if action.type == "switch" and action.slot < len(active_pokemon_list) and action.switch_idx < len(bench_pokemon_list):
            switched_out_pokemon = active_pokemon_list[action.slot]
            switched_in_pokemon = bench_pokemon_list[action.switch_idx]
            if switched_out_pokemon and not ignore_trapping and state.is_pokemon_trapped(switched_out_pokemon):
                continue  # Blocked from switching due to being trapped!

            if switched_in_pokemon and not switched_in_pokemon.fainted:
                # Apply switch-out abilities & reset volatile conditions on switch out
                if switched_out_pokemon:
                    if not switched_out_pokemon.fainted:
                        if switched_out_pokemon.ability == "regenerator":
                            switched_out_pokemon.heal(switched_out_pokemon.max_hp / 3.0)
                        elif switched_out_pokemon.ability == "naturalcure":
                            switched_out_pokemon.status = None
                            switched_out_pokemon.toxic_turns = 0
                            switched_out_pokemon.sleep_turns = 0
                        elif switched_out_pokemon.ability == "zerotohero" and not switched_out_pokemon.hero_form_active:
                            switched_out_pokemon.hero_form_active = True
                            switched_out_pokemon.species = "palafinhero"
                            switched_out_pokemon.base_stats = {"hp": 100, "atk": 160, "def": 97, "spa": 106, "spd": 87, "spe": 100}
                            switched_out_pokemon.stats = switched_out_pokemon._calculate_stats_from_base(
                                effort_values_spread=switched_out_pokemon.evs, nature=switched_out_pokemon.nature
                            )

                    switched_out_pokemon.clear_boosts()
                    switched_out_pokemon.first_turn = False
                    switched_out_pokemon.taunt_turns = 0
                    switched_out_pokemon.encore_turns = 0
                    switched_out_pokemon.encore_move = None
                    switched_out_pokemon.disabled_move = None
                    switched_out_pokemon.disable_turns = 0
                    switched_out_pokemon.torment = False
                    switched_out_pokemon.throat_chop_turns = 0
                    switched_out_pokemon.charging_move = None
                    switched_out_pokemon.charging_target = 0
                    switched_out_pokemon.destiny_bond = False
                    switched_out_pokemon.cursed = False
                    switched_out_pokemon.last_move_used = None
                    switched_out_pokemon.protect_counter = 0
                    switched_out_pokemon.is_protected = False
                    switched_out_pokemon.protect_variant = None
                    switched_out_pokemon.endure_active = False
                    switched_out_pokemon.types = switched_out_pokemon.base_types
                    switched_out_pokemon.ability = switched_out_pokemon.base_ability
                    switched_out_pokemon.roost_active = False
                    switched_out_pokemon.protean_activated = False
                    switched_out_pokemon.flinch = False
                    switched_out_pokemon.just_flinched = False
                    switched_out_pokemon.perish_turns = 0
                    switched_out_pokemon.is_trapped = False
                    switched_out_pokemon.imprison = False
                    switched_out_pokemon.unburden_active = False
                    switched_out_pokemon.substitute_hp = 0.0
                    switched_out_pokemon.leech_seed = False
                    switched_out_pokemon.salt_cure = False
                    switched_out_pokemon.syrup_bomb_turns = 0
                    switched_out_pokemon.yawn_turns = 0
                    switched_out_pokemon.confusion_turns = 0
                    switched_out_pokemon.flash_fire_active = False
                    switched_out_pokemon.charge_active = False
                    switched_out_pokemon.glaive_rush_vulnerable = False
                    switched_out_pokemon.paradox_boost_stat = None
                    switched_out_pokemon.crit_stage = 0
                    switched_out_pokemon.minimize_active = False
                    switched_out_pokemon.must_recharge = False
                    switched_out_pokemon.locked_move = None
                    switched_out_pokemon.locked_move_turns = 0
                    switched_out_pokemon.truant_resting = False
                    switched_out_pokemon.consecutive_move_uses = 0
                    switched_out_pokemon.slot = -1
                    bench_pokemon_list[action.switch_idx] = switched_out_pokemon

                switched_in_pokemon.first_turn = True
                switched_in_pokemon.switched_in_this_turn = True
                switched_in_pokemon.protect_counter = 0
                switched_in_pokemon.is_protected = False
                switched_in_pokemon.protect_variant = None
                switched_in_pokemon.endure_active = False
                switched_in_pokemon.protean_activated = False
                switched_in_pokemon.flinch = False
                switched_in_pokemon.just_flinched = False
                switched_in_pokemon.throat_chop_turns = 0
                switched_in_pokemon.charging_move = None
                switched_in_pokemon.charging_target = 0
                switched_in_pokemon.unburden_active = False
                switched_in_pokemon.slow_start_turns = 5 if switched_in_pokemon.ability == "slowstart" else 0
                switched_in_pokemon.slot = action.slot
                active_pokemon_list[action.slot] = switched_in_pokemon

                _apply_entry_hazards(switched_in_pokemon, state)
                if not switched_in_pokemon.fainted:
                    _trigger_entry_abilities(switched_in_pokemon, state)


def _trigger_mega(pokemon: Optional[SimulatedPokemon], state: Optional[SimulatedBattleState] = None) -> None:
    if not pokemon or pokemon.is_mega or pokemon.fainted:
        return
    if state is not None:
        if pokemon.side == 1 and state.mega_used:
            return
        if pokemon.side == 2 and state.opponent_mega_used:
            return
    cleaned_species = pokemon.species
    mega_stones = constants.BASE_TO_MEGA_STONE.get(cleaned_species, [])
    if not mega_stones and cleaned_species != "rayquaza":
        return
    if cleaned_species != "rayquaza" and pokemon.item not in mega_stones:
        return

    mega_species = None
    # Match exact held Mega Stone first (crucial for Charizard X/Y, Raichu X/Y, Garchomp/Lucario/Absol Z)
    for candidate_mega_species, stone_name in constants.MEGA_TO_MEGA_STONE.items():
        if stone_name == pokemon.item and (candidate_mega_species.startswith(cleaned_species) or cleaned_species in candidate_mega_species):
            mega_species = candidate_mega_species
            break
    if mega_species is None:
        for candidate_mega_species, stone_name in constants.MEGA_TO_MEGA_STONE.items():
            if stone_name in mega_stones and (candidate_mega_species.startswith(cleaned_species) or cleaned_species in candidate_mega_species):
                mega_species = candidate_mega_species
                break
    if mega_species is None and cleaned_species == "rayquaza":
        mega_species = "rayquazamega"

    if mega_species and mega_species in constants.MEGA_SPECIES_STATS:
        old_ability = pokemon.ability
        mega_data = constants.MEGA_SPECIES_STATS[mega_species]
        pokemon.species = mega_species
        pokemon.types = tuple(clean_id(type_name) for type_name in mega_data["types"])
        pokemon.base_stats = mega_data["base_stats"]
        pokemon.base_types = pokemon.types
        pokemon.ability = clean_id(mega_data["ability"])
        pokemon.base_ability = pokemon.ability
        pokemon.stats = pokemon._calculate_stats_from_base(
            effort_values_spread=pokemon.evs, nature=pokemon.nature
        )
        pokemon.is_mega = True
        pokemon.can_mega_evolve = False
        if state is not None:
            if pokemon.side == 1:
                state.mega_used = True
            else:
                state.opponent_mega_used = True
            if pokemon.ability != old_ability:
                _trigger_entry_abilities(pokemon, state)

def _execute_single_move( # pyright: ignore[reportGeneralTypeIssues]
    state: SimulatedBattleState,
    attacker: SimulatedPokemon,
    target: SimulatedPokemon,
    move: SimulatedMove,
    is_spread: bool = False,
    move_execution_queue: Optional[list[Any]] = None,
    is_magic_bounced: bool = False,
) -> bool:
    if attacker.fainted or target.fainted:
        return False

    if attacker.slot not in (0, 1):
        active_list = state.active_pokemon if attacker.side == 1 else state.opponent_active_pokemon
        if attacker in active_list:
            attacker.slot = active_list.index(attacker)

    is_ally = (target.side == attacker.side and target != attacker)
    ignores_ability = attacker.ability in ("moldbreaker", "teravolt", "turboblaze", "myceliummight")
    unignorable_abilities = ("fullmetalbody", "shadowshield", "prismarmor", "comatose")
    target_eff_ability = (
        target.ability
        if (not ignores_ability or target.ability in unignorable_abilities)
        else ""
    )
    eff_weather = _get_effective_weather(state)
    has_protective_contact = (
        attacker.item == "protectivepads"
        or attacker.ability == "longreach"
        or (attacker.item == "punchingglove" and "punch" in move.flags)
    )

    # 1. Attacker volatile & item restrictions (Taunt, Disable, Encore, Torment, Choice lock, Assault Vest, Throat Chop, Imprison)
    if _is_move_blocked_by_attacker_restrictions(state, attacker, move):
        return False

    # First Impression & Mat Block only succeed on the user's first turn on the field
    if move.id in ("firstimpression", "matblock") and not attacker.first_turn:
        return False

    # Sucker Punch & Thunderclap fail if target is ally or has already moved or is not queued to use a damaging move
    if move.id in ("suckerpunch", "thunderclap"):
        if target.moved_this_turn:
            return False
        target_queued_damaging = False
        if move_execution_queue:
            for _, _, q_side, q_slot, q_act in move_execution_queue:
                if q_side == target.side and q_slot == target.slot and q_act.type == "move" and q_act.move_id:
                    q_mv = target.moves.get(q_act.move_id) or SimulatedMove.from_id(q_act.move_id)
                    if q_mv and q_mv.category in ("physical", "special"):
                        target_queued_damaging = True
                    break
        if not target_queued_damaging:
            return False

    # Upper Hand (+3 priority): succeeds ONLY if target has not moved and is queued to use a priority (+1 to +3) damaging move
    if move.id == "upperhand":
        if target.moved_this_turn:
            return False
        target_queued_prio_damaging = False
        if move_execution_queue:
            for q_prio, _, q_side, q_slot, q_act in move_execution_queue:
                if q_side == target.side and q_slot == target.slot and q_act.type == "move" and q_act.move_id:
                    q_mv = target.moves.get(q_act.move_id) or SimulatedMove.from_id(q_act.move_id)
                    if q_mv and q_mv.category in ("physical", "special") and q_prio > 0:
                        target_queued_prio_damaging = True
                    break
        if not target_queued_prio_damaging:
            return False

    # Poltergeist fails if target has no held item
    if move.id == "poltergeist" and target.item in ("", "none"):
        return False

    effective_priority = get_effective_move_priority(move, attacker, state.fields)
    is_move_spread = is_spread or move.is_spread or (
        move.id == "expandingforce" and "psychicterrain" in state.fields and is_pokemon_grounded(attacker, state.fields)
    )

    # Record last move used once attacker commits to the move
    attacker.last_move_used = move.id

    # 2. Defensive shield protections (Protect variants, Feint, Unseen Fist/Piercing Drill, Wide Guard, Quick Guard, Crafty Shield, Mat Block)
    bypasses_protect = (
        move.breaks_protect
        or move.id in _PROTECT_BREAKING_MOVES
        or (attacker.ability in ("unseenfist", "piercingdrill") and "contact" in move.flags)
    )
    if move.id == "feint" and not is_ally:
        target.is_protected = False
        target.protect_variant = None
        state.wide_guard[target.side] = False
        state.quick_guard[target.side] = False
        state.crafty_shield[target.side] = False
        state.mat_block[target.side] = False

    if target.is_protected and "protect" in move.flags and not is_ally and not bypasses_protect:
        p_var = target.protect_variant or "protect"
        # King's Shield, Silk Trap, Burning Bulwark, Obstruct only block DAMAGING moves (status moves pass through!)
        if p_var in ("kingsshield", "silktrap", "burningbulwark", "obstruct") and move.category == "status":
            pass
        else:
            if "contact" in move.flags and not has_protective_contact and not attacker.fainted:
                if p_var == "spikyshield" and attacker.ability != "magicguard":
                    attacker.take_damage(attacker.max_hp / 8.0)
                elif p_var == "banefulbunker" and attacker.status is None:
                    if not any(t in attacker.types for t in ("poison", "steel")) and attacker.ability not in ("immunity", "pastelveil", "purifyingsalt", "comatose"):
                        attacker.status = "psn"
                        _check_status_berries(attacker, state)
                elif p_var == "kingsshield":
                    _apply_external_stat_drops(attacker, [("atk", -1)], source=target, state=state)
                elif p_var == "silktrap":
                    _apply_external_stat_drops(attacker, [("spe", -1)], source=target, state=state)
                elif p_var == "burningbulwark" and attacker.status is None:
                    if "fire" not in attacker.types and attacker.ability not in ("waterveil", "waterbubble", "thermalexchange", "purifyingsalt", "comatose"):
                        attacker.status = "brn"
                        _check_status_berries(attacker, state)
                elif p_var == "obstruct":
                    _apply_external_stat_drops(attacker, [("def", -2)], source=target, state=state)
            if move.has_crash_damage and attacker.ability != "magicguard":
                attacker.take_damage(attacker.max_hp * 0.50)
            return False

    if is_move_spread and not is_ally and state.wide_guard.get(target.side, False) and not bypasses_protect:
        return False
    if effective_priority > 0 and not is_ally and state.quick_guard.get(target.side, False) and not bypasses_protect:
        return False
    if move.category == "status" and not is_ally and state.crafty_shield.get(target.side, False) and target != attacker:
        return False
    if move.category in ("physical", "special") and not is_ally and state.mat_block.get(target.side, False) and not bypasses_protect:
        return False

    # 3. Psychic Terrain & Priority-Blocking Abilities (Dazzling, Queenly Majesty, Armor Tail)
    if effective_priority > 0 and not is_ally and target != attacker:
        if "psychicterrain" in state.fields and is_pokemon_grounded(target, state.fields, ignore_ability=ignores_ability):
            return False
        target_side_active = state.opponent_active_pokemon if attacker.side == 1 else state.active_pokemon
        if not ignores_ability and any(
            opp and not opp.fainted and opp.ability in ("dazzling", "queenlymajesty", "armortail")
            for opp in target_side_active
        ):
            return False
        if attacker.ability == "prankster" and move.category == "status" and "dark" in target.types:
            return False

    # 4. Fake Out / Upper Hand flinch logic
    if move.id == "fakeout":
        if not attacker.first_turn:
            return False
        if not is_ally and "ghost" not in target.types and target_eff_ability not in ("innerfocus", "shielddust") and target.item != "covertcloak" and target.substitute_hp <= 0.0:
            target.flinch = True
    elif move.id == "upperhand":
        if not is_ally and "ghost" not in target.types and target_eff_ability not in ("innerfocus", "shielddust") and target.item != "covertcloak" and target.substitute_hp <= 0.0:
            target.flinch = True

    # 5. Good as Gold & Magic Bounce on Status Moves
    if (
        move.category == "status"
        and target != attacker
        and target_eff_ability == "goodasgold"
        and move.id not in (
            "haze", "trickroom", "tailwind", "reflect", "lightscreen", "auroraveil",
            "sunnyday", "raindance", "sandstorm", "snowscape", "hail",
            "psychicterrain", "electricterrain", "grassyterrain", "mistyterrain",
            "gravity", "wideguard", "quickguard", "craftyshield", "matblock", "followme", "ragepowder",
            "allyswitch", "chillyreception", "imprison", "perishsong",
            "lifedew", "howl", "lunarblessing", "junglehealing", "stealthrock", "spikes", "toxicspikes", "stickyweb"
        )
    ):
        return False

    if (
        move.category == "status"
        and target != attacker
        and not is_ally
        and target_eff_ability == "magicbounce"
        and not is_magic_bounced
        and ("reflectable" in move.flags or move.id in (
            "taunt", "encore", "disable", "torment", "thunderwave", "glare", "stunspore",
            "willowisp", "spore", "hypnosis", "sleeppowder", "darkvoid", "sing", "yawn",
            "lovelykiss", "toxic", "poisonpowder", "poisongas", "leechseed", "stealthrock",
            "spikes", "toxicspikes", "stickyweb", "partingshot", "screech", "faketears",
            "charm", "featherdance", "scaryface", "growl", "leer", "sweetscent", "defog",
            "roar", "whirlwind", "soak", "entrain", "strengthsap"
        ))
    ):
        return _execute_single_move(
            state,
            target,
            attacker,
            move,
            is_spread=False,
            move_execution_queue=move_execution_queue,
            is_magic_bounced=True,
        )

    # 1. Ally Support, Turn-Order Control & Targeting Moves
    if move.id == "helpinghand":
        target.helping_hand = True
        return True
    elif move.id == "coaching":
        target.apply_boost("atk", 1)
        target.apply_boost("def", 1)
        return True
    elif move.id == "decorate":
        target.apply_boost("atk", 2)
        target.apply_boost("spa", 2)
        return True
    elif move.id == "aromaticmist":
        target.apply_boost("spd", 1)
        return True
    elif move.id == "healpulse":
        heal_fraction = 0.75 if attacker.ability == "megalauncher" else 0.50
        target.heal(target.max_hp * heal_fraction)
        return True
    elif move.id == "floralhealing":
        heal_fraction = 0.66 if "grassyterrain" in state.fields else 0.50
        target.heal(target.max_hp * heal_fraction)
        return True
    elif move.id == "lifedew":
        attacker.heal(attacker.max_hp * 0.25)
        active_list = state.active_pokemon if attacker.side == 1 else state.opponent_active_pokemon
        partner = next((p for p in active_list if p is not None and p != attacker), None)
        if partner and not partner.fainted:
            partner.heal(partner.max_hp * 0.25)
        return True
    elif move.id in ("lunarblessing", "junglehealing"):
        for ally_mon in (state.active_pokemon if attacker.side == 1 else state.opponent_active_pokemon):
            if ally_mon and not ally_mon.fainted:
                ally_mon.heal(ally_mon.max_hp * 0.25)
                ally_mon.status = None
                ally_mon.toxic_turns = 0
                ally_mon.sleep_turns = 0
        return True
    elif move.id == "howl":
        attacker.apply_boost("atk", 1)
        active_list = state.active_pokemon if attacker.side == 1 else state.opponent_active_pokemon
        partner = next((p for p in active_list if p is not None and p != attacker), None)
        if partner and not partner.fainted and partner.ability != "soundproof":
            partner.apply_boost("atk", 1)
        return True
    elif move.id == "dragoncheer":
        active_list = state.active_pokemon if attacker.side == 1 else state.opponent_active_pokemon
        partner = next((p for p in active_list if p is not None and p != attacker), None)
        if partner and not partner.fainted:
            partner.crit_stage = max(partner.crit_stage, 2 if "dragon" in partner.types else 1)
        return True
    elif move.id == "focusenergy":
        attacker.crit_stage = max(attacker.crit_stage, 2)
        return True
    elif move.id == "charge":
        attacker.apply_boost("spd", 1)
        attacker.charge_active = True
        return True
    elif move.id == "pollenpuff" and is_ally:
        target.heal(target.max_hp * 0.50)
        return True
    elif move.id == "spicyextract":
        target.apply_boost("atk", 2)
        _apply_external_stat_drops(target, [("def", -2)], source=attacker, state=state, ignore_ability=ignores_ability)
        return True
    elif move.id == "swagger":
        target.apply_boost("atk", 2)
        if target_eff_ability != "owntempo" and not ("mistyterrain" in state.fields and is_pokemon_grounded(target, state.fields)):
            target.confusion_turns = 2
            _check_status_berries(target, state)
        return True
    elif move.id == "flatter":
        target.apply_boost("spa", 1)
        if target_eff_ability != "owntempo" and not ("mistyterrain" in state.fields and is_pokemon_grounded(target, state.fields)):
            target.confusion_turns = 2
            _check_status_berries(target, state)
        return True
    elif move.id == "afteryou":
        if move_execution_queue and not target.moved_this_turn:
            for idx, (prio, spd, s_idx, sl_idx, act) in enumerate(move_execution_queue):
                if s_idx == target.side and sl_idx == target.slot:
                    move_execution_queue[idx] = (100, spd, s_idx, sl_idx, act)
                    return True
        return False
    elif move.id == "quash":
        if move_execution_queue and not target.moved_this_turn:
            for idx, (prio, spd, s_idx, sl_idx, act) in enumerate(move_execution_queue):
                if s_idx == target.side and sl_idx == target.slot:
                    move_execution_queue[idx] = (-100, spd, s_idx, sl_idx, act)
                    return True
        return False
    elif move.id == "instruct":
        if (
            target and not target.fainted and target.last_move_used
            and target.last_move_used not in constants.PROTECT_MOVES
            and target.last_move_used not in ("instruct", "fakeout", "firstimpression", "focuspunch", "beakblast", "shelltrap", "sketch", "transform")
            and target.charging_move is None
            and not target.must_recharge
        ):
            inst_move = target.moves.get(target.last_move_used) or SimulatedMove.from_id(target.last_move_used)
            if inst_move and not _is_move_blocked_by_attacker_restrictions(state, target, inst_move):
                opp_list = state.opponent_active_pokemon if target.side == 1 else state.active_pokemon
                inst_targets = [o for o in opp_list if o and not o.fainted]
                if not inst_move.is_spread and inst_targets:
                    inst_targets = [inst_targets[0]]
                any_inst_hit = False
                for i_tgt in inst_targets:
                    if _execute_single_move(state, target, i_tgt, inst_move, is_spread=len(inst_targets) > 1, move_execution_queue=move_execution_queue):
                        any_inst_hit = True
                if inst_move.category in ("physical", "special"):
                    _apply_attacker_post_move_effects(target, inst_move, any_inst_hit, state)
                return True
        return False

    # 2. Status Moves (Self & Opponent)
    if move.category == "status":
        # Target debuffs
        if move.id in ("partingshot", "tearfullook"):
            _apply_external_stat_drops(
                target,
                [("atk", -1), ("spa", -1)],
                source=attacker,
                state=state,
                ignore_ability=ignores_ability,
            )
        elif move.id == "screech":
            _apply_external_stat_drops(
                target, [("def", -2)], source=attacker, state=state, ignore_ability=ignores_ability
            )
        elif move.id in ("faketears", "metalsound"):
            _apply_external_stat_drops(
                target, [("spd", -2)], source=attacker, state=state, ignore_ability=ignores_ability
            )
        elif move.id in ("charm", "featherdance"):
            _apply_external_stat_drops(
                target, [("atk", -2)], source=attacker, state=state, ignore_ability=ignores_ability
            )
        elif move.id == "eerieimpulse":
            _apply_external_stat_drops(
                target, [("spa", -2)], source=attacker, state=state, ignore_ability=ignores_ability
            )
        elif move.id in ("scaryface", "stringshot", "cottonspore"):
            is_powder = (move.id == "cottonspore")
            if not (is_powder and ("grass" in target.types or target_eff_ability == "overcoat" or target.item == "safetygoggles")):
                _apply_external_stat_drops(
                    target, [("spe", -2)], source=attacker, state=state, ignore_ability=ignores_ability
                )
        elif move.id in ("growl", "babydolleyes"):
            _apply_external_stat_drops(
                target, [("atk", -1)], source=attacker, state=state, ignore_ability=ignores_ability
            )
        elif move.id in ("leer", "tailwhip"):
            _apply_external_stat_drops(
                target, [("def", -1)], source=attacker, state=state, ignore_ability=ignores_ability
            )
        elif move.id == "confide":
            _apply_external_stat_drops(
                target, [("spa", -1)], source=attacker, state=state, ignore_ability=ignores_ability
            )
        elif move.id == "nobleroar":
            _apply_external_stat_drops(
                target, [("atk", -1), ("spa", -1)], source=attacker, state=state, ignore_ability=ignores_ability
            )
        elif move.id in ("sandattack", "smokescreen", "flash", "kinesis"):
            _apply_external_stat_drops(
                target, [("accuracy", -1)], source=attacker, state=state, ignore_ability=ignores_ability
            )
        elif move.id == "sweetscent":
            _apply_external_stat_drops(
                target, [("evasion", -2)], source=attacker, state=state, ignore_ability=ignores_ability
            )
        elif move.id == "tickle":
            _apply_external_stat_drops(
                target, [("atk", -1), ("def", -1)], source=attacker, state=state, ignore_ability=ignores_ability
            )
        elif move.id == "haze":
            for active_pkmn in state.active_pokemon + state.opponent_active_pokemon:
                if active_pkmn and not active_pkmn.fainted:
                    active_pkmn.clear_boosts()
        elif move.id == "clearsmog":
            target.clear_boosts()
        elif move.id == "topsyturvy":
            for k in target.boosts:
                target.boosts[k] = -target.boosts[k]
        elif move.id == "heartswap":
            attacker.boosts, target.boosts = target.boosts.copy(), attacker.boosts.copy()
        elif move.id == "psychup":
            attacker.boosts = target.boosts.copy()

        # Self stat boosts
        elif move.id == "swordsdance":
            attacker.apply_boost("atk", 2)
        elif move.id == "nastyplot":
            attacker.apply_boost("spa", 2)
        elif move.id == "calmmind":
            attacker.apply_boost("spa", 1)
            attacker.apply_boost("spd", 1)
        elif move.id == "dragondance":
            attacker.apply_boost("atk", 1)
            attacker.apply_boost("spe", 1)
        elif move.id == "bulkup":
            attacker.apply_boost("atk", 1)
            attacker.apply_boost("def", 1)
        elif move.id == "quiverdance":
            attacker.apply_boost("spa", 1)
            attacker.apply_boost("spd", 1)
            attacker.apply_boost("spe", 1)
        elif move.id in ("irondefense", "acidarmor"):
            attacker.apply_boost("def", 2)
        elif move.id == "amnesia":
            attacker.apply_boost("spd", 2)
        elif move.id in ("agility", "rockpolish", "autotomize"):
            attacker.apply_boost("spe", 2)
        elif move.id == "shiftgear":
            attacker.apply_boost("atk", 1)
            attacker.apply_boost("spe", 2)
        elif move.id == "shellsmash":
            attacker.apply_boost("atk", 2)
            attacker.apply_boost("spa", 2)
            attacker.apply_boost("spe", 2)
            attacker.apply_boost("def", -1)
            attacker.apply_boost("spd", -1)
            attacker.check_white_herb()
            if attacker.item == "ejectpack" and attacker.ability != "contrary":
                attacker.consume_item()
                _force_bench_switch(attacker, state)
        elif move.id == "bellydrum":
            if attacker.current_hp > attacker.max_hp * 0.50:
                attacker.take_damage(attacker.max_hp * 0.50)
                attacker.boosts["atk"] = 6
                _check_hp_berries(attacker, state)
        elif move.id == "filletaway":
            if attacker.current_hp > attacker.max_hp * 0.50:
                attacker.take_damage(attacker.max_hp * 0.50)
                attacker.apply_boost("atk", 2)
                attacker.apply_boost("spa", 2)
                attacker.apply_boost("spe", 2)
                _check_hp_berries(attacker, state)
        elif move.id == "tidyup":
            for mon in state.active_pokemon + state.opponent_active_pokemon:
                if mon:
                    mon.substitute_hp = 0.0
            for cond_dict in (state.side_conditions, state.opponent_side_conditions):
                for h_key in ("stealthrock", "spikes", "toxicspikes", "stickyweb"):
                    cond_dict.pop(h_key, None)
            attacker.apply_boost("atk", 1)
            attacker.apply_boost("spe", 1)
        elif move.id == "victorydance":
            attacker.apply_boost("atk", 1)
            attacker.apply_boost("def", 1)
            attacker.apply_boost("spe", 1)
        elif move.id == "clangoroussoul":
            if attacker.current_hp > attacker.max_hp * 0.33:
                attacker.take_damage(attacker.max_hp * 0.33)
                for stat_key in ("atk", "def", "spa", "spd", "spe"):
                    attacker.apply_boost(stat_key, 1)
                _check_hp_berries(attacker, state)
        elif move.id == "geomancy":
            attacker.apply_boost("spa", 2)
            attacker.apply_boost("spd", 2)
            attacker.apply_boost("spe", 2)
        elif move.id == "coil":
            attacker.apply_boost("atk", 1)
            attacker.apply_boost("def", 1)
            attacker.apply_boost("accuracy", 1)
        elif move.id == "growth":
            boost_amount = 2 if (any(w in eff_weather for w in ("sunnyday", "desolateland", "sun")) or attacker.ability == "megasol") else 1
            attacker.apply_boost("atk", boost_amount)
            attacker.apply_boost("spa", boost_amount)
        elif move.id == "workup":
            attacker.apply_boost("atk", 1)
            attacker.apply_boost("spa", 1)
        elif move.id in ("cosmicpower", "defendorder"):
            attacker.apply_boost("def", 1)
            attacker.apply_boost("spd", 1)
        elif move.id in ("cottonguard", "barrier"):
            attacker.apply_boost("def", 3)
        elif move.id == "tailglow":
            attacker.apply_boost("spa", 3)
        elif move.id == "minimize":
            attacker.apply_boost("evasion", 2)
            attacker.minimize_active = True

        # Field, Side Conditions & Entry Hazards
        elif move.id == "tailwind":
            side_conditions = state.side_conditions if attacker.side == 1 else state.opponent_side_conditions
            if side_conditions.get("tailwind", 0) == 0:
                side_conditions["tailwind"] = 4
                for ally_mon in (state.active_pokemon if attacker.side == 1 else state.opponent_active_pokemon):
                    if ally_mon and not ally_mon.fainted:
                        if ally_mon.ability == "windrider":
                            ally_mon.apply_boost("atk", 1)
                        elif ally_mon.ability == "windpower":
                            ally_mon.charge_active = True
        elif move.id == "trickroom":
            if "trickroom" in state.fields:
                del state.fields["trickroom"]
            else:
                state.fields["trickroom"] = 5
                _check_room_service(state)
        elif move.id == "reflect":
            side_conditions = state.side_conditions if attacker.side == 1 else state.opponent_side_conditions
            side_conditions["reflect"] = 8 if attacker.item == "lightclay" else 5
        elif move.id == "lightscreen":
            side_conditions = state.side_conditions if attacker.side == 1 else state.opponent_side_conditions
            side_conditions["lightscreen"] = 8 if attacker.item == "lightclay" else 5
        elif move.id == "auroraveil":
            if any(w in eff_weather for w in ("snowscape", "hail", "snow")):
                side_conditions = state.side_conditions if attacker.side == 1 else state.opponent_side_conditions
                side_conditions["auroraveil"] = 8 if attacker.item == "lightclay" else 5
        elif move.id == "safeguard":
            side_conditions = state.side_conditions if attacker.side == 1 else state.opponent_side_conditions
            side_conditions["safeguard"] = 5
        elif move.id == "mist":
            side_conditions = state.side_conditions if attacker.side == 1 else state.opponent_side_conditions
            side_conditions["mist"] = 5
        elif move.id == "stealthrock":
            opp_conds = state.opponent_side_conditions if attacker.side == 1 else state.side_conditions
            opp_conds["stealthrock"] = 1
        elif move.id == "spikes":
            opp_conds = state.opponent_side_conditions if attacker.side == 1 else state.side_conditions
            opp_conds["spikes"] = min(3, opp_conds.get("spikes", 0) + 1)
        elif move.id == "toxicspikes":
            opp_conds = state.opponent_side_conditions if attacker.side == 1 else state.side_conditions
            opp_conds["toxicspikes"] = min(2, opp_conds.get("toxicspikes", 0) + 1)
        elif move.id == "stickyweb":
            opp_conds = state.opponent_side_conditions if attacker.side == 1 else state.side_conditions
            opp_conds["stickyweb"] = 1
        elif move.id == "courtchange":
            state.side_conditions, state.opponent_side_conditions = (
                state.opponent_side_conditions.copy(),
                state.side_conditions.copy(),
            )
        elif move.id == "defog":
            for cond_dict in (state.side_conditions, state.opponent_side_conditions):
                for key in ("reflect", "lightscreen", "auroraveil", "safeguard", "mist", "stealthrock", "spikes", "toxicspikes", "stickyweb"):
                    cond_dict.pop(key, None)
            for t_key in ("electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"):
                state.fields.pop(t_key, None)
            _apply_external_stat_drops(target, [("evasion", -1)], source=attacker, state=state, ignore_ability=ignores_ability)
        elif move.id == "sunnyday":
            state.weather = {"sunnyday": 8 if attacker.item == "heatrock" else 5}
            _check_paradox_abilities(state)
        elif move.id == "raindance":
            state.weather = {"raindance": 8 if attacker.item == "damprock" else 5}
            _check_paradox_abilities(state)
        elif move.id == "sandstorm":
            state.weather = {"sandstorm": 8 if attacker.item == "smoothrock" else 5}
            _check_paradox_abilities(state)
        elif move.id in ("snowscape", "hail"):
            state.weather = {"snowscape": 8 if attacker.item == "icyrock" else 5}
            for mon in state.active_pokemon + state.opponent_active_pokemon:
                if mon and not mon.fainted and mon.ability == "iceface" and mon.ice_face_busted:
                    mon.ice_face_busted = False
            _check_paradox_abilities(state)
        elif move.id == "psychicterrain":
            _set_terrain(state, "psychicterrain", 8 if attacker.item == "terrainextender" else 5)
        elif move.id == "electricterrain":
            _set_terrain(state, "electricterrain", 8 if attacker.item == "terrainextender" else 5)
        elif move.id == "grassyterrain":
            _set_terrain(state, "grassyterrain", 8 if attacker.item == "terrainextender" else 5)
        elif move.id == "mistyterrain":
            _set_terrain(state, "mistyterrain", 8 if attacker.item == "terrainextender" else 5)

        # Status afflictions (Blocked by Substitute unless Infiltrator)
        elif move.id in ("thunderwave", "glare", "stunspore") and target.status is None:
            if target.substitute_hp > 0.0 and attacker.ability != "infiltrator":
                return False
            is_powder = move.id == "stunspore"
            target_side_conds = state.opponent_side_conditions if target.side == 2 else state.side_conditions
            target_allies = state.active_pokemon if target.side == 1 else state.opponent_active_pokemon
            has_flower_veil = "grass" in target.types and (
                target_eff_ability == "flowerveil"
                or (not ignores_ability and any(a and not a.fainted and a != target and a.ability == "flowerveil" for a in target_allies))
            )
            if not (is_powder and ("grass" in target.types or target_eff_ability == "overcoat" or target.item == "safetygoggles")):
                if not ("safeguard" in target_side_conds and attacker.ability != "infiltrator") and not has_flower_veil:
                    if not ("mistyterrain" in state.fields and is_pokemon_grounded(target, state.fields, ignore_ability=ignores_ability)):
                        if "electric" not in target.types and (move.id != "thunderwave" or "ground" not in target.types) and target_eff_ability not in ("limber", "purifyingsalt", "comatose", "goodasgold"):
                            target.status = "par"
                            if target.ability == "synchronize" and attacker.status is None and "electric" not in attacker.types:
                                attacker.status = "par"
                                _check_status_berries(attacker, state)
                            _check_status_berries(target, state)
        elif move.id in ("willowisp",) and target.status is None:
            if target.substitute_hp > 0.0 and attacker.ability != "infiltrator":
                return False
            if target_eff_ability in ("flashfire", "wellbakedbody"):
                if target_eff_ability == "wellbakedbody":
                    target.apply_boost("def", 2)
                else:
                    target.flash_fire_active = True
                return False
            target_side_conds = state.opponent_side_conditions if target.side == 2 else state.side_conditions
            target_allies = state.active_pokemon if target.side == 1 else state.opponent_active_pokemon
            has_flower_veil = "grass" in target.types and (
                target_eff_ability == "flowerveil"
                or (not ignores_ability and any(a and not a.fainted and a != target and a.ability == "flowerveil" for a in target_allies))
            )
            if not ("safeguard" in target_side_conds and attacker.ability != "infiltrator") and not has_flower_veil:
                if not ("mistyterrain" in state.fields and is_pokemon_grounded(target, state.fields, ignore_ability=ignores_ability)):
                    if "fire" not in target.types and target_eff_ability not in ("waterveil", "waterbubble", "thermalexchange", "purifyingsalt", "comatose", "goodasgold"):
                        target.status = "brn"
                        if target.ability == "synchronize" and attacker.status is None and "fire" not in attacker.types:
                            attacker.status = "brn"
                            _check_status_berries(attacker, state)
                        _check_status_berries(target, state)
        elif move.id in ("spore", "hypnosis", "sleeppowder", "darkvoid", "sing", "yawn", "lovelykiss") and target.status is None:
            if target.substitute_hp > 0.0 and attacker.ability != "infiltrator":
                return False
            is_powder = move.id in ("spore", "sleeppowder")
            target_side_conds = state.opponent_side_conditions if target.side == 2 else state.side_conditions
            target_allies = state.active_pokemon if target.side == 1 else state.opponent_active_pokemon
            has_flower_veil = "grass" in target.types and (
                target_eff_ability == "flowerveil"
                or (not ignores_ability and any(a and not a.fainted and a != target and a.ability == "flowerveil" for a in target_allies))
            )
            has_sweet_veil = target_eff_ability == "sweetveil" or (
                not ignores_ability and any(a and not a.fainted and a != target and a.ability == "sweetveil" for a in target_allies)
            )
            if not (is_powder and ("grass" in target.types or target_eff_ability == "overcoat" or target.item == "safetygoggles")):
                if not ("safeguard" in target_side_conds and attacker.ability != "infiltrator") and not has_flower_veil and not has_sweet_veil:
                    if not (("electricterrain" in state.fields or "mistyterrain" in state.fields) and is_pokemon_grounded(target, state.fields, ignore_ability=ignores_ability)):
                        if target_eff_ability not in ("vitalspirit", "insomnia", "purifyingsalt", "sweetveil", "comatose", "goodasgold"):
                            if move.id == "yawn":
                                if target.yawn_turns == 0:
                                    target.yawn_turns = 2
                            else:
                                target.status = "slp"
                                target.sleep_turns = 0
                                _check_status_berries(target, state)
        elif move.id in ("toxic", "poisonpowder", "poisongas") and target.status is None:
            if target.substitute_hp > 0.0 and attacker.ability != "infiltrator":
                return False
            is_powder = move.id == "poisonpowder"
            target_side_conds = state.opponent_side_conditions if target.side == 2 else state.side_conditions
            target_allies = state.active_pokemon if target.side == 1 else state.opponent_active_pokemon
            has_flower_veil = "grass" in target.types and (
                target_eff_ability == "flowerveil"
                or (not ignores_ability and any(a and not a.fainted and a != target and a.ability == "flowerveil" for a in target_allies))
            )
            has_pastel_veil = target_eff_ability == "pastelveil" or (
                not ignores_ability and any(a and not a.fainted and a != target and a.ability == "pastelveil" for a in target_allies)
            )
            can_poison_type = (attacker.ability == "corrosion") or not any(type_name in target.types for type_name in ("poison", "steel"))
            if not (is_powder and ("grass" in target.types or target_eff_ability == "overcoat" or target.item == "safetygoggles")):
                if not ("safeguard" in target_side_conds and attacker.ability != "infiltrator") and not has_flower_veil and not has_pastel_veil:
                    if not ("mistyterrain" in state.fields and is_pokemon_grounded(target, state.fields, ignore_ability=ignores_ability)):
                        if can_poison_type and target_eff_ability not in ("immunity", "pastelveil", "purifyingsalt", "comatose", "goodasgold"):
                            target.status = "tox" if move.id == "toxic" else "psn"
                            target.toxic_turns = 0
                            if target.ability == "synchronize" and attacker.status is None and not any(t in attacker.types for t in ("poison", "steel")):
                                attacker.status = target.status
                                _check_status_berries(attacker, state)
                            _check_status_berries(target, state)
        elif move.id in ("confuseray", "supersonic", "sweetkiss", "teeterdance"):
            if target.substitute_hp <= 0.0 or attacker.ability == "infiltrator":
                if target_eff_ability != "owntempo" and not ("mistyterrain" in state.fields and is_pokemon_grounded(target, state.fields)):
                    target.confusion_turns = 2
                    _check_status_berries(target, state)
        elif move.id == "leechseed":
            if "grass" not in target.types and target.substitute_hp <= 0.0 and not target.leech_seed:
                target.leech_seed = True
                target.leech_seed_slot = attacker.slot if attacker.slot in (0, 1) else 0

        # Direct Healing, Wish, Rest, Substitute, Shed Tail, Baton Pass
        elif move.id == "wish":
            state.wish_slots[(attacker.side, attacker.slot)] = (2, attacker.max_hp * 0.50)
            attacker.heal(attacker.max_hp * 0.50)
        elif move.id == "rest":
            if not (("electricterrain" in state.fields or "mistyterrain" in state.fields) and is_pokemon_grounded(attacker, state.fields)):
                if attacker.ability not in ("insomnia", "vitalspirit", "comatose", "purifyingsalt"):
                    attacker.current_hp = attacker.max_hp
                    attacker.current_hp_fraction = 1.0
                    attacker.status = "slp"
                    attacker.sleep_turns = 0
                    attacker.toxic_turns = 0
                    _check_status_berries(attacker, state)
        elif move.id in ("recover", "slackoff", "softboiled", "milkdrink", "healorder"):
            attacker.heal(attacker.max_hp * 0.50)
        elif move.id == "shoreup":
            heal_rate = 0.66 if any(w in eff_weather for w in ("sandstorm", "sand")) else 0.50
            attacker.heal(attacker.max_hp * heal_rate)
        elif move.id == "roost":
            attacker.heal(attacker.max_hp * 0.50)
            if "flying" in attacker.types:
                stripped = tuple(t for t in attacker.types if t != "flying")
                attacker.types = stripped if stripped else ("normal",)
                attacker.roost_active = True
        elif move.id in ("synthesis", "moonlight", "morningsun"):
            is_sunny = any(w in eff_weather for w in ("sunnyday", "desolateland", "sun")) or attacker.ability == "megasol"
            is_bad_weather = any(w in eff_weather for w in ("raindance", "primordialsea", "rain", "sandstorm", "sand", "snowscape", "hail", "snow")) and attacker.ability != "megasol"
            heal_rate = 0.66 if is_sunny else (0.25 if is_bad_weather else 0.50)
            attacker.heal(attacker.max_hp * heal_rate)
        elif move.id == "floralhealing":
            heal_rate = 0.66 if "grassyterrain" in state.fields else 0.50
            target.heal(target.max_hp * heal_rate)
        elif move.id == "strengthsap":
            if not is_ally and target_eff_ability != "goodasgold" and target.substitute_hp <= 0.0 and target.boosts.get("atk", 0) > -6:
                target_atk = target.calculate_effective_attack(is_special=False, weather=eff_weather, fields=state.fields)
                _apply_external_stat_drops(
                    target, [("atk", -1)], source=attacker, state=state, ignore_ability=ignores_ability
                )
                if target_eff_ability == "liquidooze":
                    if attacker.ability != "magicguard":
                        attacker.take_damage(target_atk)
                else:
                    attacker.heal(target_atk)
        elif move.id == "painsplit":
            if target.substitute_hp <= 0.0:
                avg_hp = (attacker.current_hp + target.current_hp) / 2.0
                attacker.current_hp = min(attacker.max_hp, max(1.0, avg_hp))
                attacker.current_hp_fraction = attacker.current_hp / attacker.max_hp
                target.current_hp = min(target.max_hp, max(1.0, avg_hp))
                target.current_hp_fraction = target.current_hp / target.max_hp
        elif move.id == "substitute":
            sub_cost = math.floor(attacker.max_hp * 0.25)
            if attacker.current_hp > sub_cost and attacker.substitute_hp <= 0.0:
                attacker.take_damage(sub_cost)
                attacker.substitute_hp = float(sub_cost)
                _check_hp_berries(attacker, state)
        elif move.id == "shedtail":
            shed_cost = math.ceil(attacker.max_hp * 0.50)
            if attacker.current_hp > shed_cost:
                attacker.take_damage(shed_cost)
                _force_bench_switch(attacker, state, shed_tail_sub_hp=math.floor(attacker.max_hp * 0.25))
        elif move.id == "batonpass":
            _force_bench_switch(attacker, state, baton_pass=True)
        elif move.id == "teleport":
            _force_bench_switch(attacker, state)

        # Team Mitigation & Redirection
        elif move.id == "wideguard":
            state.wide_guard[attacker.side] = True
        elif move.id == "quickguard":
            state.quick_guard[attacker.side] = True
        elif move.id == "craftyshield":
            state.crafty_shield[attacker.side] = True
        elif move.id == "matblock":
            state.mat_block[attacker.side] = True
        elif move.id in ("followme", "ragepowder"):
            state.redirect_target[attacker.side] = (attacker.slot, move.id)
        elif move.id == "allyswitch":
            active_list = state.active_pokemon if attacker.side == 1 else state.opponent_active_pokemon
            if len(active_list) >= 2:
                p0, p1 = active_list[0], active_list[1]
                if p0 and p1 and not p0.fainted and not p1.fainted:
                    p0.slot, p1.slot = p1.slot, p0.slot
                    active_list[0], active_list[1] = p1, p0
                    if move_execution_queue:
                        for i, q_item in enumerate(move_execution_queue):
                            prio, spd, s_idx, sl_idx, act = q_item
                            if s_idx == attacker.side:
                                move_execution_queue[i] = (prio, spd, s_idx, 1 - sl_idx, act)
        elif move.id == "destinybond":
            attacker.destiny_bond = True

        # Disruption, Item Swap, Phazing & Debuffs
        elif move.id in ("trick", "switcheroo"):
            if target.substitute_hp <= 0.0 and target_eff_ability not in ("stickyhold", "goodasgold"):
                a_unswappable = attacker.item.endswith("ite") or attacker.item in ("boosterenergy", "hearthflamemask", "wellspringmask", "cornerstonemask", "rustedshield", "rustedsword")
                t_unswappable = target.item.endswith("ite") or target.item in ("boosterenergy", "hearthflamemask", "wellspringmask", "cornerstonemask", "rustedshield", "rustedsword")
                if not a_unswappable and not t_unswappable:
                    attacker.item, target.item = target.item, attacker.item
        elif move.id in ("whirlwind", "roar"):
            if not is_ally and target_eff_ability not in ("suctioncups", "guarddog", "goodasgold") and not (move.id == "roar" and target_eff_ability == "soundproof"):
                _force_bench_switch(target, state)
        elif move.id == "taunt":
            if not is_ally:
                target_allies = state.active_pokemon if target.side == 1 else state.opponent_active_pokemon
                has_aroma_veil = target_eff_ability == "aromaveil" or (
                    not ignores_ability and any(a and not a.fainted and a != target and a.ability == "aromaveil" for a in target_allies)
                )
                if target.item == "mentalherb":
                    target.consume_item()
                elif not has_aroma_veil and target_eff_ability not in ("aromaveil", "oblivious", "goodasgold"):
                    target.taunt_turns = 3
        elif move.id == "encore":
            if not is_ally and target.last_move_used and target.last_move_used in target.moves:
                target_allies = state.active_pokemon if target.side == 1 else state.opponent_active_pokemon
                has_aroma_veil = target_eff_ability == "aromaveil" or (
                    not ignores_ability and any(a and not a.fainted and a != target and a.ability == "aromaveil" for a in target_allies)
                )
                if target.item == "mentalherb":
                    target.consume_item()
                elif not has_aroma_veil and target_eff_ability not in ("aromaveil", "goodasgold"):
                    target.encore_turns = 3
                    target.encore_move = target.last_move_used
        elif move.id == "disable":
            if not is_ally and target.last_move_used and target.last_move_used in target.moves:
                target_allies = state.active_pokemon if target.side == 1 else state.opponent_active_pokemon
                has_aroma_veil = target_eff_ability == "aromaveil" or (
                    not ignores_ability and any(a and not a.fainted and a != target and a.ability == "aromaveil" for a in target_allies)
                )
                if target.item == "mentalherb":
                    target.consume_item()
                elif not has_aroma_veil and target_eff_ability not in ("aromaveil", "goodasgold"):
                    target.disabled_move = target.last_move_used
                    target.disable_turns = 4
        elif move.id == "torment":
            if not is_ally:
                target_allies = state.active_pokemon if target.side == 1 else state.opponent_active_pokemon
                has_aroma_veil = target_eff_ability == "aromaveil" or (
                    not ignores_ability and any(a and not a.fainted and a != target and a.ability == "aromaveil" for a in target_allies)
                )
                if target.item == "mentalherb":
                    target.consume_item()
                elif not has_aroma_veil and target_eff_ability not in ("aromaveil", "goodasgold"):
                    target.torment = True
        elif move.id == "gravity":
            state.fields["gravity"] = 5
            for mon in state.active_pokemon + state.opponent_active_pokemon:
                if mon and not mon.fainted:
                    mon.roost_active = False
        elif move.id == "chillyreception":
            state.weather = {"snowscape": 8 if attacker.item == "icyrock" else 5}
            for mon in state.active_pokemon + state.opponent_active_pokemon:
                if mon and not mon.fainted and mon.ability == "iceface" and mon.ice_face_busted:
                    mon.ice_face_busted = False
            _force_bench_switch(attacker, state)
        elif move.id == "curse":
            if "ghost" in attacker.types:
                attacker.take_damage(attacker.max_hp * 0.50)
                target.cursed = True
                _check_hp_berries(attacker, state)
            else:
                attacker.apply_boost("spe", -1)
                attacker.apply_boost("atk", 1)
                attacker.apply_boost("def", 1)
                attacker.check_white_herb()
        elif move.id == "soak":
            if target.substitute_hp <= 0.0 and not target.is_mega:
                target.types = ("water",)
        elif move.id == "magicpowder":
            if target.substitute_hp <= 0.0 and not target.is_mega:
                target.types = ("psychic",)
        elif move.id == "forestscurse":
            if target.substitute_hp <= 0.0 and "grass" not in target.types:
                target.types = target.types + ("grass",)
        elif move.id == "trickortreat":
            if target.substitute_hp <= 0.0 and "ghost" not in target.types:
                target.types = target.types + ("ghost",)
        elif move.id == "reflecttype":
            if target and not target.fainted and target.types:
                attacker.types = target.types
        elif move.id == "conversion":
            if attacker.moves:
                first_move = next(iter(attacker.moves.values()))
                if first_move.type:
                    attacker.types = (first_move.type,)
        elif move.id == "camouflage":
            if "electricterrain" in state.fields: attacker.types = ("electric",)
            elif "grassyterrain" in state.fields: attacker.types = ("grass",)
            elif "psychicterrain" in state.fields: attacker.types = ("psychic",)
            elif "mistyterrain" in state.fields: attacker.types = ("fairy",)
            else: attacker.types = ("normal",)
        elif move.id in ("skillswap", "entrainment", "doodle", "worryseed", "simplebeam", "gastroacid"):
            if target.ability not in _UNSUPPRESSIBLE_ABILITIES:
                if move.id == "skillswap" and attacker.ability not in _UNSUPPRESSIBLE_ABILITIES:
                    attacker.ability, target.ability = target.ability, attacker.ability
                elif move.id == "entrainment":
                    target.ability = attacker.ability
                elif move.id == "worryseed":
                    target.ability = "insomnia"
                elif move.id == "simplebeam":
                    target.ability = "simple"
                elif move.id == "gastroacid":
                    target.ability = ""
        elif move.id == "perishsong":
            for active_pkmn in state.active_pokemon + state.opponent_active_pokemon:
                if active_pkmn and not active_pkmn.fainted and active_pkmn.ability not in ("soundproof", "goodasgold"):
                    if active_pkmn.perish_turns == 0:
                        active_pkmn.perish_turns = 3
        elif move.id in ("meanlook", "block", "spiderweb"):
            if not is_ally and "ghost" not in target.types and target.item != "shedshell" and target.substitute_hp <= 0.0:
                target.is_trapped = True
        elif move.id == "imprison":
            attacker.imprison = True
        return True

    # 3. Fixed-Damage & HP-Proportional Damaging Moves (Super Fang, Ruination, Nature's Madness, Final Gambit, Seismic Toss, Night Shade, Endeavor)
    if move.id in ("superfang", "ruination", "naturesmadness", "finalgambit", "seismictoss", "nightshade", "endeavor"):
        if target_eff_ability == "wonderguard":
            return False
        if move.id in ("superfang", "finalgambit", "seismictoss", "endeavor") and "ghost" in target.types and attacker.ability not in ("scrappy", "mindseye"):
            return False
        if move.id == "nightshade" and "normal" in target.types:
            return False
        if move.id in ("superfang", "ruination", "naturesmadness"):
            fixed_dmg = max(1.0, math.floor(target.current_hp * 0.50))
        elif move.id == "finalgambit":
            fixed_dmg = attacker.current_hp
            attacker.take_damage(attacker.current_hp)
        elif move.id in ("seismictoss", "nightshade"):
            fixed_dmg = float(attacker.level)
        else:  # endeavor
            if target.current_hp <= attacker.current_hp:
                return False
            fixed_dmg = target.current_hp - attacker.current_hp
        if target.substitute_hp > 0.0 and attacker.ability != "infiltrator":
            target.substitute_hp = max(0.0, target.substitute_hp - fixed_dmg)
        else:
            target.take_damage(fixed_dmg)
            target.took_damage_this_turn = True
            _check_hp_berries(target, state)
        return True

    # 4. Dynamic Base Power & Move Typing
    base_power = move.base_power
    effective_type = move.type

    if move.id in ("waterspout", "eruption", "dragonenergy"):
        base_power = max(1.0, math.floor(150.0 * attacker.current_hp_fraction))
    elif move.id in ("flail", "reversal"):
        r = math.floor(attacker.current_hp * 48.0 / max(1.0, attacker.max_hp))
        if r <= 1: base_power = 200.0
        elif r <= 4: base_power = 150.0
        elif r <= 9: base_power = 100.0
        elif r <= 16: base_power = 80.0
        elif r <= 32: base_power = 40.0
        else: base_power = 20.0
    elif move.id in ("crushgrip", "hardpress"):
        max_bp = 120.0 if move.id == "crushgrip" else 100.0
        base_power = max(1.0, math.floor(max_bp * target.current_hp_fraction))
    elif move.id == "brine":
        base_power = 130.0 if target.current_hp_fraction <= 0.50 else 65.0
    elif move.id in ("venoshock", "barbbarrage"):
        if target.status in ("psn", "tox"):
            base_power *= 2.0
    elif move.id == "lastrespects":
        fainted_allies = sum(
            1 for allied_pokemon in (state.all_team_pokemon() if attacker.side == 1 else state.all_opponent_pokemon()) if allied_pokemon.fainted
        )
        base_power = 50.0 + (50.0 * fainted_allies)
    elif move.id == "ragefist":
        hits = state.hit_counts.get(attacker.species, 0)
        base_power = min(350.0, 50.0 + (50.0 * hits))
    elif move.id in ("lowkick", "grassknot"):
        target_weight = target.weight
        if target_weight < 10.0: base_power = 20.0
        elif target_weight < 25.0: base_power = 40.0
        elif target_weight < 50.0: base_power = 60.0
        elif target_weight < 100.0: base_power = 80.0
        elif target_weight < 200.0: base_power = 100.0
        else: base_power = 120.0
    elif move.id in ("heavyslam", "heatcrash"):
        weight_ratio = attacker.weight / max(0.1, target.weight)
        if weight_ratio >= 5.0: base_power = 120.0
        elif weight_ratio >= 4.0: base_power = 100.0
        elif weight_ratio >= 3.0: base_power = 80.0
        elif weight_ratio >= 2.0: base_power = 60.0
        else: base_power = 40.0
    elif move.id == "gyroball":
        a_conds = state.side_conditions if attacker.side == 1 else state.opponent_side_conditions
        t_conds = state.side_conditions if target.side == 1 else state.opponent_side_conditions
        a_spe = max(1.0, attacker.calculate_effective_speed(eff_weather, state.fields, a_conds))
        t_spe = target.calculate_effective_speed(eff_weather, state.fields, t_conds)
        base_power = min(150.0, math.floor(25.0 * t_spe / a_spe) + 1.0)
    elif move.id == "electroball":
        a_conds = state.side_conditions if attacker.side == 1 else state.opponent_side_conditions
        t_conds = state.side_conditions if target.side == 1 else state.opponent_side_conditions
        a_spe = attacker.calculate_effective_speed(eff_weather, state.fields, a_conds)
        t_spe = max(1.0, target.calculate_effective_speed(eff_weather, state.fields, t_conds))
        ratio = a_spe / t_spe
        if ratio >= 4.0: base_power = 150.0
        elif ratio >= 3.0: base_power = 120.0
        elif ratio >= 2.0: base_power = 80.0
        elif ratio >= 1.0: base_power = 60.0
        else: base_power = 40.0
    elif move.id == "knockoff":
        if target.item not in ("none", "") and not target.item.endswith("ite") and target.item not in ("boosterenergy", "hearthflamemask", "wellspringmask", "cornerstonemask"):
            base_power = 97.5
    elif move.id == "facade":
        if attacker.status in ("brn", "par", "psn", "tox"):
            base_power = 140.0
    elif move.id == "acrobatics":
        if attacker.item in ("none", ""):
            base_power = 110.0
    elif move.id in ("hex", "infernalparade"):
        if target.status is not None:
            base_power = 130.0 if move.id == "hex" else 120.0
    elif move.id == "assurance":
        if target.took_damage_this_turn:
            base_power *= 2.0
    elif move.id in ("avalanche", "revenge"):
        if attacker.took_damage_this_turn:
            base_power *= 2.0
    elif move.id in ("stompingtantrum", "temperflare"):
        if attacker.last_move_failed:
            base_power *= 2.0
    elif move.id == "gravapple":
        if "gravity" in state.fields:
            base_power = 120.0
    elif move.id == "weatherball":
        if any(w in eff_weather for w in ("sunnyday", "desolateland", "sun")) or attacker.ability == "megasol":
            base_power, effective_type = 100.0, "fire"
        elif any(w in eff_weather for w in ("raindance", "primordialsea", "rain")):
            base_power, effective_type = 100.0, "water"
        elif any(w in eff_weather for w in ("sandstorm", "sand")):
            base_power, effective_type = 100.0, "rock"
        elif any(w in eff_weather for w in ("snowscape", "hail", "snow")):
            base_power, effective_type = 100.0, "ice"
    elif move.id == "terrainpulse":
        if is_pokemon_grounded(attacker, state.fields):
            if "electricterrain" in state.fields: base_power, effective_type = 100.0, "electric"
            elif "grassyterrain" in state.fields: base_power, effective_type = 100.0, "grass"
            elif "psychicterrain" in state.fields: base_power, effective_type = 100.0, "psychic"
            elif "mistyterrain" in state.fields: base_power, effective_type = 100.0, "fairy"
    elif move.id == "expandingforce":
        if "psychicterrain" in state.fields and is_pokemon_grounded(attacker, state.fields):
            base_power = 120.0
    elif move.id == "risingvoltage":
        if "electricterrain" in state.fields and is_pokemon_grounded(target, state.fields, ignore_ability=ignores_ability):
            base_power = 140.0
    elif move.id == "psyblade":
        if "electricterrain" in state.fields:
            base_power = 120.0
    elif move.id == "mistyexplosion":
        if "mistyterrain" in state.fields and is_pokemon_grounded(attacker, state.fields):
            base_power = 150.0
    elif move.id in ("storedpower", "powertrip"):
        pos_boosts = sum(max(0, val) for val in attacker.boosts.values())
        base_power = 20.0 + (20.0 * pos_boosts)
    elif move.id == "revelationdance":
        if attacker.types:
            effective_type = attacker.types[0]
    elif move.id == "aurawheel":
        effective_type = "dark" if "hangry" in attacker.species else "electric"
    elif move.id == "ragingbull":
        if "fire" in attacker.species or "blaze" in attacker.species:
            effective_type = "fire"
        elif "water" in attacker.species or "aqua" in attacker.species:
            effective_type = "water"
        else:
            effective_type = "fighting"
    elif move.id == "ivycudgel":
        if "hearthflame" in attacker.species or attacker.item == "hearthflamemask":
            effective_type = "fire"
        elif "wellspring" in attacker.species or attacker.item == "wellspringmask":
            effective_type = "water"
        elif "cornerstone" in attacker.species or attacker.item == "cornerstonemask":
            effective_type = "rock"
        else:
            effective_type = "grass"

    if base_power <= 0:
        return False

    # Double Shock & Burn Up: fail if user lacks corresponding type; strip type on use
    had_electric_stab = False
    had_fire_stab = False
    if move.id == "doubleshock":
        if "electric" not in attacker.types:
            return False  # Move fails if user is not Electric type
        had_electric_stab = True
        attacker.types = tuple(t for t in attacker.types if t != "electric")
    elif move.id == "burnup":
        if "fire" not in attacker.types:
            return False  # Move fails if user is not Fire type
        had_fire_stab = True
        attacker.types = tuple(t for t in attacker.types if t != "fire")

    # Normal-type conversion abilities & move-type altering abilities
    base_power_multiplier = 1.0
    if effective_type == "normal":
        if state.ion_deluge:
            effective_type = "electric"
        elif attacker.ability == "pixilate": effective_type, base_power_multiplier = "fairy", 1.2
        elif attacker.ability == "aerilate": effective_type, base_power_multiplier = "flying", 1.2
        elif attacker.ability == "refrigerate": effective_type, base_power_multiplier = "ice", 1.2
        elif attacker.ability == "galvanize": effective_type, base_power_multiplier = "electric", 1.2
        elif attacker.ability == "dragonize": effective_type, base_power_multiplier = "dragon", 1.2
    if attacker.ability == "normalize":
        effective_type, base_power_multiplier = "normal", 1.2
    elif attacker.ability == "liquidvoice" and ("sound" in move.flags or move.id in _SOUND_MOVES):
        effective_type = "water"

    # Protean / Libero (changes user's type to move type before STAB/damage)
    if attacker.ability in ("protean", "libero") and not attacker.protean_activated:
        attacker.types = (effective_type,)
        attacker.protean_activated = True

    # Soundproof, Bulletproof & Wind Rider immunities
    if target_eff_ability == "soundproof" and ("sound" in move.flags or move.id in _SOUND_MOVES):
        return False
    if target_eff_ability == "bulletproof" and ("bullet" in move.flags or move.id in _BULLET_MOVES):
        return False
    if target_eff_ability == "windrider" and ("wind" in move.flags or move.id in _WIND_MOVES):
        target.apply_boost("atk", 1)
        return False

    has_gravity = ("gravity" in state.fields)
    # Type effectiveness & Scrappy / Mind's Eye
    if move.id == "struggle":
        type_multiplier = 1.0  # Struggle deals typeless damage, hits Ghost-types neutrally
    else:
        type_multiplier = target.damage_multiplier(effective_type, has_gravity=has_gravity)
    if move.id == "freezedry" and "water" in target.types:
        type_multiplier *= 4.0  # Ice vs Water becomes 2.0x instead of 0.5x
    elif move.id == "flyingpress":
        type_multiplier *= target.damage_multiplier("flying", has_gravity=has_gravity)
    elif move.id == "thousandarrows" and "flying" in target.types and type_multiplier == 0.0:
        type_multiplier = 1.0
    if type_multiplier == 0.0:
        if attacker.ability in ("scrappy", "mindseye") and effective_type in ("normal", "fighting"):
            type_multiplier = 1.0
        else:
            if move.has_crash_damage and attacker.ability != "magicguard":
                attacker.take_damage(attacker.max_hp * 0.50)
            return False

    # Tera Shell: at full HP, forces all non-immune damaging hits to 0.5x effectiveness
    if target_eff_ability == "terashell" and target.current_hp >= target.max_hp - 1e-6 and type_multiplier > 0.0:
        type_multiplier = 0.5

    # Defensive ability immunities
    if effective_type == "ground" and target_eff_ability in ("levitate", "eelevate") and not has_gravity and target.item != "ironball" and move.id != "thousandarrows":
        if move.has_crash_damage and attacker.ability != "magicguard":
            attacker.take_damage(attacker.max_hp * 0.50)
        return False
    if effective_type == "ground" and target_eff_ability == "eartheater":
        target.heal(target.max_hp * 0.25)
        if move.has_crash_damage and attacker.ability != "magicguard":
            attacker.take_damage(attacker.max_hp * 0.50)
        return False
    if effective_type == "fire" and target_eff_ability in ("flashfire", "wellbakedbody"):
        if target_eff_ability == "wellbakedbody":
            target.apply_boost("def", 2)
        else:
            target.flash_fire_active = True
        return False
    if effective_type == "water" and target_eff_ability in ("waterabsorb", "stormdrain", "dryskin"):
        if target_eff_ability == "stormdrain":
            target.apply_boost("spa", 1)
        else:
            target.heal(target.max_hp * 0.25)
        return False
    if effective_type == "electric" and target_eff_ability in ("voltabsorb", "lightningrod", "motordrive"):
        if target_eff_ability == "lightningrod":
            target.apply_boost("spa", 1)
        elif target_eff_ability == "motordrive":
            target.apply_boost("spe", 1)
        else:
            target.heal(target.max_hp * 0.25)
        if move.has_crash_damage and attacker.ability != "magicguard":
            attacker.take_damage(attacker.max_hp * 0.50)
        return False
    if effective_type == "grass" and target_eff_ability == "sapsipper":
        target.apply_boost("atk", 1)
        return False
    if is_ally and target_eff_ability == "telepathy":
        return False
    if target_eff_ability == "wonderguard" and type_multiplier <= 1.0 and move.id != "struggle":
        return False

    # Screen-breaking moves (Brick Break, Psychic Fangs, Raging Bull) shatter screens before damage calculation
    target_side_conditions = state.opponent_side_conditions if target.side == 2 else state.side_conditions
    if move.id in ("brickbreak", "psychicfangs", "ragingbull"):
        target_side_conditions.pop("reflect", None)
        target_side_conditions.pop("lightscreen", None)
        target_side_conditions.pop("auroraveil", None)

    # STAB calculation
    same_type_attack_bonus = 1.0
    if effective_type in attacker.types or (move.id == "doubleshock" and had_electric_stab) or (move.id == "burnup" and had_fire_stab):
        same_type_attack_bonus = 2.0 if attacker.ability == "adaptability" else 1.5

    # Critical Hit Resolution
    is_crit = False
    if target_eff_ability not in ("battlearmor", "shellarmor") and "luckychant" not in target_side_conditions:
        if move.id in _GUARANTEED_CRIT_MOVES or attacker.crit_stage >= 3:
            is_crit = True
        elif attacker.ability == "merciless" and target.status in ("psn", "tox"):
            is_crit = True
        else:
            crit_stage = attacker.crit_stage
            if move.id in _HIGH_CRIT_MOVES:
                crit_stage += 1
            if attacker.ability == "superluck":
                crit_stage += 1
            if attacker.item in ("scopelens", "razorclaw"):
                crit_stage += 1
            if crit_stage >= 3:
                is_crit = True

    # Attack and Defense Stat Resolution (accounting for Unaware, Mold Breaker, and Critical Hits)
    is_special = (move.category == "special")
    ignore_target_def_boosts = (attacker.ability == "unaware")
    ignore_attacker_atk_boosts = (target_eff_ability == "unaware")
    eff_weather = _get_effective_weather(state)

    if move.id == "bodypress":
        effective_attack_stat = attacker.calculate_effective_defense(
            False, eff_weather, state.fields,
            ignore_boosts=ignore_attacker_atk_boosts,
            ignore_negative_boosts=is_crit,
        )
        effective_defense_stat = target.calculate_effective_defense(
            False, eff_weather, state.fields,
            ignore_boosts=ignore_target_def_boosts,
            ignore_ability=ignores_ability,
            ignore_positive_boosts=is_crit,
        )
        atk_stat_used, def_stat_used, atk_holder = "def", "def", attacker
    elif move.id == "foulplay":
        effective_attack_stat = target.calculate_effective_attack(
            False, eff_weather, state.fields,
            ignore_boosts=ignore_attacker_atk_boosts,
            ignore_negative_boosts=is_crit,
        )
        effective_defense_stat = target.calculate_effective_defense(
            False, eff_weather, state.fields,
            ignore_boosts=ignore_target_def_boosts,
            ignore_ability=ignores_ability,
            ignore_positive_boosts=is_crit,
        )
        atk_stat_used, def_stat_used, atk_holder = "atk", "def", target
    elif move.id in ("psyshock", "psystrike", "secretsword"):
        effective_attack_stat = attacker.calculate_effective_attack(
            True, eff_weather, state.fields,
            ignore_boosts=ignore_attacker_atk_boosts,
            ignore_negative_boosts=is_crit,
        )
        effective_defense_stat = target.calculate_effective_defense(
            False, eff_weather, state.fields,
            ignore_boosts=ignore_target_def_boosts,
            ignore_ability=ignores_ability,
            ignore_positive_boosts=is_crit,
        )
        atk_stat_used, def_stat_used, atk_holder = "spa", "def", attacker
    elif move.id == "photongeyser":
        atk_val = attacker.calculate_effective_attack(False, eff_weather, state.fields, ignore_boosts=ignore_attacker_atk_boosts, ignore_negative_boosts=is_crit)
        spa_val = attacker.calculate_effective_attack(True, eff_weather, state.fields, ignore_boosts=ignore_attacker_atk_boosts, ignore_negative_boosts=is_crit)
        is_special = (spa_val >= atk_val)
        effective_attack_stat = spa_val if is_special else atk_val
        effective_defense_stat = target.calculate_effective_defense(
            is_special, eff_weather, state.fields,
            ignore_boosts=ignore_target_def_boosts,
            ignore_ability=ignores_ability,
            ignore_positive_boosts=is_crit,
        )
        atk_stat_used = "spa" if is_special else "atk"
        def_stat_used = "spd" if is_special else "def"
        atk_holder = attacker
    else:
        effective_attack_stat = attacker.calculate_effective_attack(
            is_special, eff_weather, state.fields,
            ignore_boosts=ignore_attacker_atk_boosts,
            ignore_negative_boosts=is_crit,
        )
        effective_defense_stat = target.calculate_effective_defense(
            is_special, eff_weather, state.fields,
            ignore_boosts=ignore_target_def_boosts,
            ignore_ability=ignores_ability,
            ignore_positive_boosts=is_crit,
        )
        atk_stat_used = "spa" if is_special else "atk"
        def_stat_used = "spd" if is_special else "def"
        atk_holder = attacker

    # Wonder Room swaps Def and SpD
    if "wonderroom" in state.fields:
        effective_defense_stat = target.calculate_effective_defense(
            not is_special, eff_weather, state.fields,
            ignore_boosts=ignore_target_def_boosts,
            ignore_ability=ignores_ability,
            ignore_positive_boosts=is_crit,
        )

    # Ruinous Quartet field stat reductions (Tablets of Ruin, Sword of Ruin, Vessel of Ruin, Beads of Ruin)
    all_active = [p for p in state.active_pokemon + state.opponent_active_pokemon if p and not p.fainted]
    if atk_stat_used == "atk" and atk_holder.ability != "tabletsofruin" and any(p.ability == "tabletsofruin" for p in all_active):
        effective_attack_stat *= 0.75
    elif atk_stat_used == "spa" and atk_holder.ability != "vesselofruin" and any(p.ability == "vesselofruin" for p in all_active):
        effective_attack_stat *= 0.75

    if def_stat_used == "def" and target.ability != "swordofruin" and any(p.ability == "swordofruin" for p in all_active):
        effective_defense_stat *= 0.75
    elif def_stat_used == "spd" and target.ability != "beadsofruin" and any(p.ability == "beadsofruin" for p in all_active):
        effective_defense_stat *= 0.75

    # Facade ignores burn penalty
    if move.id == "facade" and attacker.status == "brn" and attacker.ability != "guts":
        effective_attack_stat *= 2.0

    # Aura Abilities (Fairy Aura, Dark Aura, Aura Break) - affect all active Pokemon on the field
    has_aura_break = any(p.ability == "aurabreak" for p in all_active)
    if effective_type == "fairy" and any(p.ability == "fairyaura" for p in all_active):
        base_power_multiplier *= 0.75 if has_aura_break else (4.0 / 3.0)
    elif effective_type == "dark" and any(p.ability == "darkaura" for p in all_active):
        base_power_multiplier *= 0.75 if has_aura_break else (4.0 / 3.0)

    # Offensive Ability Multipliers (including Champions Regulation M-C custom abilities)
    if attacker.ability in ("transistor", "dragonsmaw") and effective_type == ("electric" if attacker.ability == "transistor" else "dragon"):
        base_power_multiplier *= 1.3 if attacker.ability == "transistor" else 1.5
    elif attacker.ability == "steelworker" and effective_type == "steel":
        base_power_multiplier *= 1.5
    elif attacker.ability == "rockypayload" and effective_type == "rock":
        base_power_multiplier *= 1.5
    elif attacker.ability == "firemane" and effective_type == "fire":
        base_power_multiplier *= 1.5
    elif attacker.ability == "waterbubble" and effective_type == "water":
        base_power_multiplier *= 2.0
    elif attacker.ability == "sandforce" and any(w in eff_weather for w in ("sandstorm", "sand")) and effective_type in ("rock", "ground", "steel"):
        base_power_multiplier *= 1.3
    elif attacker.ability == "toughclaws" and "contact" in move.flags:
        base_power_multiplier *= 1.3
    elif attacker.ability == "sharpness" and "slicing" in move.flags:
        base_power_multiplier *= 1.5
    elif attacker.ability == "strongjaw" and "bite" in move.flags:
        base_power_multiplier *= 1.5
    elif attacker.ability == "megalauncher" and "pulse" in move.flags:
        base_power_multiplier *= 1.5
    elif attacker.ability == "ironfist" and "punch" in move.flags:
        base_power_multiplier *= 1.2
    elif attacker.ability == "punkrock" and ("sound" in move.flags or move.id in _SOUND_MOVES):
        base_power_multiplier *= 1.3
    elif attacker.ability == "technician" and base_power <= 60.0:
        base_power_multiplier *= 1.5
    elif attacker.ability == "sheerforce" and bool(move.secondary):
        base_power_multiplier *= 1.3
    elif attacker.ability == "reckless" and (move.recoil > 0.0 or move.has_crash_damage):
        base_power_multiplier *= 1.2
    elif attacker.ability == "supremeoverlord":
        fainted_count = sum(
            1 for p in (state.all_team_pokemon() if attacker.side == 1 else state.all_opponent_pokemon()) if p.fainted
        )
        base_power_multiplier *= (1.0 + 0.10 * min(5, fainted_count))
    elif attacker.ability == "analytic" and target.moved_this_turn:
        base_power_multiplier *= 1.3
    elif attacker.ability == "stakeout" and target.first_turn:
        base_power_multiplier *= 2.0
    elif attacker.ability == "toxicboost" and attacker.status in ("psn", "tox") and not is_special:
        base_power_multiplier *= 1.5
    elif attacker.ability == "flareboost" and attacker.status == "brn" and is_special:
        base_power_multiplier *= 1.5

    # Starter Pinch Abilities (Blaze, Torrent, Overgrow, Swarm)
    if attacker.current_hp <= attacker.max_hp / 3.0 + 1e-6:
        if (
            (attacker.ability == "blaze" and effective_type == "fire")
            or (attacker.ability == "torrent" and effective_type == "water")
            or (attacker.ability == "overgrow" and effective_type == "grass")
            or (attacker.ability == "swarm" and effective_type == "bug")
        ):
            base_power_multiplier *= 1.5

    # Flash Fire & Charge boost
    if attacker.flash_fire_active and effective_type == "fire" and attacker.ability == "flashfire":
        base_power_multiplier *= 1.5
    if attacker.charge_active and effective_type == "electric":
        base_power_multiplier *= 2.0
        attacker.charge_active = False

    if effective_type == "steel":
        attacker_side_active = state.active_pokemon if attacker.side == 1 else state.opponent_active_pokemon
        for ally_mon in attacker_side_active:
            if ally_mon and not ally_mon.fainted and ally_mon.ability == "steelyspirit":
                base_power_multiplier *= 1.5

    # Ally Battery / Power Spot boosts
    attacker_side_active = state.active_pokemon if attacker.side == 1 else state.opponent_active_pokemon
    for ally_mon in attacker_side_active:
        if ally_mon and not ally_mon.fainted and ally_mon != attacker:
            if ally_mon.ability == "powerspot":
                base_power_multiplier *= 1.3
            if ally_mon.ability == "battery" and is_special:
                base_power_multiplier *= 1.3

    # Offensive Held Item Base Power Multipliers
    if attacker.item in constants.TYPE_BOOSTING_ITEMS and clean_id(constants.TYPE_BOOSTING_ITEMS[attacker.item]) == effective_type:
        base_power_multiplier *= 1.2
    elif attacker.item == "muscleband" and not is_special:
        base_power_multiplier *= 1.1
    elif attacker.item == "wiseglasses" and is_special:
        base_power_multiplier *= 1.1
    elif attacker.item == "punchingglove" and "punch" in move.flags:
        base_power_multiplier *= 1.1
    elif attacker.item == "normalgem" and effective_type == "normal":
        base_power_multiplier *= 1.3
        attacker.consume_item()
    elif attacker.item in ("wellspringmask", "hearthflamemask", "cornerstonemask") and "ogerpon" in attacker.species:
        base_power_multiplier *= 1.2
    elif attacker.item == "souldew" and attacker.species in ("latios", "latias") and effective_type in ("dragon", "psychic"):
        base_power_multiplier *= 1.2
    elif attacker.item in ("adamantorb", "adamantcrystal") and "dialga" in attacker.species and effective_type in ("dragon", "steel"):
        base_power_multiplier *= 1.2
    elif attacker.item in ("lustrousorb", "lustrousglobe") and "palkia" in attacker.species and effective_type in ("dragon", "water"):
        base_power_multiplier *= 1.2
    elif attacker.item in ("griseousorb", "griseouscore") and "giratina" in attacker.species and effective_type in ("dragon", "ghost"):
        base_power_multiplier *= 1.2

    total_damage_modifier = base_power_multiplier * same_type_attack_bonus * type_multiplier

    if is_crit:
        total_damage_modifier *= 2.25 if attacker.ability == "sniper" else 1.5

    if attacker.ability == "tintedlens" and 0.0 < type_multiplier < 1.0:
        total_damage_modifier *= 2.0
    if attacker.ability == "neuroforce" and type_multiplier > 1.0:
        total_damage_modifier *= 1.25

    # Held Item Final Damage Modifiers (Life Orb, Expert Belt, Metronome)
    if attacker.item == "lifeorb":
        total_damage_modifier *= 1.3
    elif attacker.item == "expertbelt" and type_multiplier > 1.0:
        total_damage_modifier *= 1.2
    elif attacker.item == "metronome" and attacker.consecutive_move_uses > 0:
        total_damage_modifier *= min(2.0, 1.0 + 0.2 * attacker.consecutive_move_uses)

    # Super-effective moves bonus (Collision Course, Electro Drift)
    if move.id in ("collisioncourse", "electrodrift") and type_multiplier > 1.0:
        total_damage_modifier *= (4.0 / 3.0)

    # Glaive Rush vulnerability on target (takes 2x damage)
    if target.glaive_rush_vulnerable:
        total_damage_modifier *= 2.0

    # Defensive Ability Multipliers (Multiscale, Shadow Shield, Thick Fat, Heatproof, Purifying Salt, Water Bubble, Fluffy, Aura Guard, Filter/Solid Rock/Prism Armor, Punk Rock, Ice Scales, Friend Guard)
    if target_eff_ability in ("multiscale", "shadowshield") and target.current_hp >= target.max_hp - 1e-6:
        total_damage_modifier *= 0.5
    if target_eff_ability == "thickfat" and effective_type in ("fire", "ice"):
        total_damage_modifier *= 0.5
    elif target_eff_ability in ("heatproof", "waterbubble") and effective_type == "fire":
        total_damage_modifier *= 0.5
    elif target_eff_ability == "dryskin" and effective_type == "fire":
        total_damage_modifier *= 1.25
    elif target_eff_ability == "purifyingsalt" and effective_type == "ghost":
        total_damage_modifier *= 0.5
    if target_eff_ability == "icescales" and is_special:
        total_damage_modifier *= 0.5
    if target_eff_ability == "auraguard" and "contact" in move.flags and attacker.item != "protectivepads":
        total_damage_modifier *= 0.5
    if target_eff_ability == "fluffy":
        if "contact" in move.flags and attacker.item != "protectivepads":
            total_damage_modifier *= 0.5
        if effective_type == "fire":
            total_damage_modifier *= 2.0
    if target_eff_ability == "punkrock" and ("sound" in move.flags or move.id in _SOUND_MOVES):
        total_damage_modifier *= 0.5
    if target_eff_ability in ("filter", "solidrock", "prismarmor") and type_multiplier > 1.0:
        total_damage_modifier *= 0.75

    target_allies = state.active_pokemon if target.side == 1 else state.opponent_active_pokemon
    if not ignores_ability and any(
        ally and not ally.fainted and ally != target and ally.ability == "friendguard"
        for ally in target_allies
    ):
        total_damage_modifier *= 0.75

    if is_spread:
        total_damage_modifier *= 0.75

    # Weather multipliers
    is_sunny = any(w in eff_weather for w in ("sunnyday", "desolateland", "sun")) or attacker.ability == "megasol"
    is_rainy = any(w in eff_weather for w in ("raindance", "primordialsea", "rain")) and attacker.ability != "megasol"
    is_bad_solar_weather = (
        any(w in eff_weather for w in ("raindance", "primordialsea", "rain", "sandstorm", "sand", "snowscape", "hail", "snow"))
        and attacker.ability != "megasol"
    )
    if move.id in ("solarbeam", "solarblade") and is_bad_solar_weather:
        total_damage_modifier *= 0.5
    if effective_type == "fire":
        if is_sunny: total_damage_modifier *= 1.5
        elif is_rainy: total_damage_modifier *= 0.5
    elif effective_type == "water":
        if move.id == "hydrosteam" and is_sunny:
            total_damage_modifier *= 1.5
        elif is_rainy: total_damage_modifier *= 1.5
        elif is_sunny: total_damage_modifier *= 0.5

    # Terrain multipliers
    if "psychicterrain" in state.fields and effective_type == "psychic" and is_pokemon_grounded(attacker, state.fields):
        total_damage_modifier *= 1.3
    elif "electricterrain" in state.fields and effective_type == "electric" and is_pokemon_grounded(attacker, state.fields):
        total_damage_modifier *= 1.3
    elif "grassyterrain" in state.fields and effective_type == "grass" and is_pokemon_grounded(attacker, state.fields):
        total_damage_modifier *= 1.3
    elif "grassyterrain" in state.fields and move.id in ("earthquake", "bulldoze", "magnitude") and is_pokemon_grounded(target, state.fields, ignore_ability=ignores_ability):
        total_damage_modifier *= 0.5
    elif "mistyterrain" in state.fields and effective_type == "dragon" and is_pokemon_grounded(target, state.fields, ignore_ability=ignores_ability):
        total_damage_modifier *= 0.5

    # Screens reduction (bypassed by Infiltrator or Critical Hit)
    if attacker.ability != "infiltrator" and not is_crit:
        screen_factor = 0.66
        if not is_special and ("reflect" in target_side_conditions or "auroraveil" in target_side_conditions):
            total_damage_modifier *= screen_factor
        elif is_special and ("lightscreen" in target_side_conditions or "auroraveil" in target_side_conditions):
            total_damage_modifier *= screen_factor

    # Helping Hand multiplier
    if attacker.helping_hand:
        total_damage_modifier *= 1.5

    # Type-Resist Berries (Occa, Passho, Wacan, Rindo, Yache, Chople, etc. + Chilan Berry for Normal)
    if (
        target.item in constants.TYPE_RESIST_BERRIES
        and not _is_unnerve_active_against(target, state)
        and target.substitute_hp <= 0.0
    ):
        berry_type = clean_id(constants.TYPE_RESIST_BERRIES[target.item])
        if (berry_type == "normal" and effective_type == "normal") or (berry_type == effective_type and type_multiplier > 1.0):
            resist_factor = 0.25 if target.ability == "ripen" else 0.5
            total_damage_modifier *= resist_factor
            consumed_resist_berry = target.item
            target.consume_item()
            if target.ability == "cudchew":
                target.cud_chew_berry = consumed_resist_berry
            if target.ability == "cheekpouch":
                target.heal(target.max_hp / 3.0)

    # Determine Multi-Hit Count
    hit_multipliers: list[float] = [1.0]
    if move.id in ("surgingstrikes", "watershuriken", "triplearrows"):
        hit_multipliers = [1.0, 1.0, 1.0] if move.id == "surgingstrikes" else ([1.0, 1.0, 1.0, 1.0] if attacker.item == "loadeddice" else [1.0, 1.0, 1.0])
    elif move.id in ("tripleaxel", "triplekick"):
        hit_multipliers = [1.0, 2.0, 3.0]
    elif move.id in ("dualwingbeat", "doublehit", "bonemerang", "doubleironbash", "dragonDarts", "dragondarts", "twinbeam", "tachyoncutter", "twineedle"):
        hit_multipliers = [1.0, 1.0]
    elif move.id == "populationbomb":
        hit_multipliers = [1.0] * 10
    elif move.id in ("bulletseed", "iciclespear", "rockblast", "scaleshot", "tailslap", "pinmissile", "armthrust", "watershuriken", "bonesrush", "bonerush"):
        if attacker.ability == "skilllink":
            hit_multipliers = [1.0] * 5
        elif attacker.item == "loadeddice":
            hit_multipliers = [1.0] * 4
        else:
            hit_multipliers = [1.0] * 3
    elif move.id == "beatup":
        team_list = state.all_team_pokemon() if attacker.side == 1 else state.all_opponent_pokemon()
        alive_count = sum(1 for p in team_list if p and not p.fainted and p.status in ("none", ""))
        hit_multipliers = [1.0] * max(1, alive_count)
    elif attacker.ability == "parentalbond" and not is_spread and len(hit_multipliers) == 1:
        hit_multipliers = [1.0, 0.25]

    single_hit_base_damage = (
        ((22.0 * base_power * (effective_attack_stat / max(1.0, effective_defense_stat))) / 50.0) + 2.0
    ) * total_damage_modifier

    actual_damage_dealt = 0.0
    hp_before_move = target.current_hp
    substitute_absorbed_all = False

    for hit_mult in hit_multipliers:
        if target.fainted or attacker.fainted:
            break

        hit_damage = single_hit_base_damage * hit_mult

        # Disguise / Ice Face damage busting (absorbs first hit before Substitute if not bypassed by Mold Breaker)
        if not ignores_ability and target.substitute_hp <= 0.0:
            if target.ability == "disguise" and not target.disguise_busted:
                target.disguise_busted = True
                target.take_damage(target.max_hp / 8.0)
                continue
            if target.ability == "iceface" and not is_special and not target.ice_face_busted:
                target.ice_face_busted = True
                continue

        # Substitute damage absorption (unless sound move or Infiltrator)
        bypasses_sub = (
            attacker.ability == "infiltrator"
            or "sound" in move.flags
            or move.id in _SOUND_MOVES
        )
        if target.substitute_hp > 0.0 and not bypasses_sub:
            sub_dmg = min(target.substitute_hp, hit_damage)
            target.substitute_hp = max(0.0, target.substitute_hp - sub_dmg)
            substitute_absorbed_all = True
            continue
        else:
            substitute_absorbed_all = False

        # Full-HP OHKO Survival (Sturdy & Focus Sash) and Endure
        was_at_full_hp = (target.current_hp >= target.max_hp - 1e-6)
        if hit_damage >= target.current_hp and target.current_hp >= 1.0:
            if target.endure_active:
                hit_damage = target.current_hp - 1.0
            elif was_at_full_hp and target.max_hp > 1.0:
                if target_eff_ability == "sturdy":
                    hit_damage = target.current_hp - 1.0
                elif target.item == "focussash":
                    hit_damage = target.current_hp - 1.0
                    target.consume_item()

        dealt_this_hit = target.take_damage(hit_damage)
        actual_damage_dealt += dealt_this_hit
        target.took_damage_this_turn = True

        # Air Balloon pops when hit by any direct damage
        if dealt_this_hit > 0 and target.item == "airballoon":
            target.consume_item()

        # Rage Fist hit counter tracking per hit
        state.hit_counts[target.species] = state.hit_counts.get(target.species, 0) + 1

        # Per-hit Stamina / Weak Armor / Justified / Contact procs
        if dealt_this_hit > 0:
            if not target.fainted:
                if target.ability == "stamina":
                    target.apply_boost("def", 1)
                elif target.ability == "weakarmor" and not is_special:
                    target.apply_boost("def", -1)
                    target.apply_boost("spe", 2)
                    target.check_white_herb()
                elif target.ability == "justified" and effective_type == "dark":
                    target.apply_boost("atk", 1)
                elif target.ability == "rattled" and effective_type in ("bug", "ghost", "dark"):
                    target.apply_boost("spe", 1)
                elif target.ability == "thermalexchange" and effective_type == "fire":
                    target.apply_boost("atk", 1)
                elif target.ability == "steamengine" and effective_type in ("fire", "water"):
                    target.apply_boost("spe", 6)
                elif target.ability == "watercompaction" and effective_type == "water":
                    target.apply_boost("def", 2)
                elif target.ability == "angerpoint" and is_crit:
                    target.boosts["atk"] = 6
                elif target.ability == "toxicdebris" and not is_special:
                    attacker_conds = state.side_conditions if attacker.side == 1 else state.opponent_side_conditions
                    attacker_conds["toxicspikes"] = min(2, attacker_conds.get("toxicspikes", 0) + 1)

            if "contact" in move.flags and attacker.item != "protectivepads" and not attacker.fainted:
                if target.ability in ("gooey", "tanglinghair"):
                    _apply_external_stat_drops(attacker, [("spe", -1)], source=target, state=state)
                if target.ability in ("roughskin", "ironbarbs") and attacker.ability != "magicguard":
                    attacker.take_damage(attacker.max_hp / 8.0)
                if target.item == "rockyhelmet" and attacker.ability != "magicguard" and not attacker.fainted:
                    attacker.take_damage(attacker.max_hp / 6.0)
                if target.ability in ("mummy", "lingeringaroma") and attacker.ability not in _UNSUPPRESSIBLE_ABILITIES:
                    attacker.ability = target.ability
                elif target.ability == "wanderingspirit" and attacker.ability not in _UNSUPPRESSIBLE_ABILITIES:
                    attacker.ability, target.ability = target.ability, attacker.ability
                elif target.ability == "perishbody":
                    if attacker.perish_turns == 0:
                        attacker.perish_turns = 4
                    if target.perish_turns == 0:
                        target.perish_turns = 4

            # Check HP pinch berries after each hit
            _check_hp_berries(target, state)

    # Destiny Bond trigger: if target faints from foe's attack, attacker faints too
    if target.destiny_bond and target.fainted and not is_ally and not attacker.fainted:
        attacker.take_damage(attacker.current_hp)

    # Innards Out & Aftermath on KO
    if target.fainted and actual_damage_dealt > 0 and not attacker.fainted and attacker.ability != "magicguard":
        if target.ability == "innardsout":
            attacker.take_damage(hp_before_move)
        elif target.ability == "aftermath" and "contact" in move.flags and attacker.item != "protectivepads":
            if not any(p.ability == "damp" for p in all_active):
                attacker.take_damage(attacker.max_hp / 4.0)

    # Attacker On-KO Abilities (Moxie, Chilling Neigh, Grim Neigh, As One, Beast Boost) & Soul-Heart
    if target.fainted and not is_ally:
        if not attacker.fainted:
            if attacker.ability in ("moxie", "chillingneigh", "asoneglastrier"):
                attacker.apply_boost("atk", 1)
            elif attacker.ability in ("grimneigh", "asonespectrier"):
                attacker.apply_boost("spa", 1)
            elif attacker.ability == "beastboost":
                best_stat = max(("atk", "def", "spa", "spd", "spe"), key=lambda s: attacker.stats.get(s, 0.0))
                attacker.apply_boost(best_stat, 1)
        for mon in state.active_pokemon + state.opponent_active_pokemon:
            if mon and not mon.fainted and mon != target and mon.ability == "soulheart":
                mon.apply_boost("spa", 1)

    # Knock Off / Thief / Covet item removal or theft
    if not substitute_absorbed_all:
        if move.id == "knockoff" and not target.fainted and target.item not in ("none", "") and not target.item.endswith("ite") and target.ability != "stickyhold":
            target.consume_item()
        elif move.id in ("thief", "covet") and not attacker.fainted and attacker.item in ("none", "") and target.item not in ("none", "") and not target.item.endswith("ite") and target.ability != "stickyhold":
            stolen_item = target.item
            target.consume_item()
            attacker.item = stolen_item

        # Throat Chop effect: prevents target from using sound moves for 2 turns
        if move.id == "throatchop" and not target.fainted and target_eff_ability != "shielddust" and target.item != "covertcloak":
            target.throat_chop_turns = 2

        # Salt Cure & Syrup Bomb volatile application
        if move.id == "saltcure" and not target.fainted and target_eff_ability != "shielddust" and target.item != "covertcloak":
            target.salt_cure = True
        elif move.id == "syrupbomb" and not target.fainted and target_eff_ability != "shielddust" and target.item != "covertcloak":
            target.syrup_bomb_turns = 3

        # Clear Smog resets all target stat stages to 0
        if move.id == "clearsmog" and not target.fainted:
            for k in target.boosts:
                target.boosts[k] = 0

        # Rapid Spin & Mortal Spin hazard/trap clearing for attacker's side
        if move.id in ("rapidspin", "mortalspin") and not attacker.fainted:
            attacker.leech_seed = False
            attacker.is_trapped = False
            attacker_conds = state.side_conditions if attacker.side == 1 else state.opponent_side_conditions
            for h_key in ("stealthrock", "spikes", "toxicspikes", "stickyweb"):
                attacker_conds.pop(h_key, None)
            if move.id == "mortalspin" and not target.fainted and target.status is None:
                if "poison" not in target.types and "steel" not in target.types and target_eff_ability not in ("immunity", "purifyingsalt", "comatose"):
                    target.status = "psn"
                    _check_status_berries(target, state)

        # Ceaseless Edge / Stone Axe hazard setting
        if move.id == "ceaselessedge" and not attacker.fainted and attacker.ability != "sheerforce":
            target_side_conditions["spikes"] = min(3, target_side_conditions.get("spikes", 0) + 1)
        elif move.id == "stoneaxe" and not attacker.fainted and attacker.ability != "sheerforce":
            target_side_conditions["stealthrock"] = 1

    # Post-Move On-Hit Procs (Weakness Policy, Berserk, Anger Shell, Electromorphosis, Wind Power, Seed Sower, Sand Spit, Cotton Down, Cursed Body)
    if actual_damage_dealt > 0:
        if not target.fainted:
            if target.item == "weaknesspolicy" and type_multiplier > 1.0:
                target.apply_boost("atk", 2)
                target.apply_boost("spa", 2)
                target.consume_item()
            if target.item == "absorbbulb" and effective_type == "water":
                target.apply_boost("spa", 1)
                target.consume_item()
            elif target.item == "cellbattery" and effective_type == "electric":
                target.apply_boost("atk", 1)
                target.consume_item()
            elif target.item == "luminousmoss" and effective_type == "water":
                target.apply_boost("spd", 1)
                target.consume_item()
            elif target.item == "snowball" and effective_type == "ice":
                target.apply_boost("atk", 1)
                target.consume_item()

            # Berserk & Anger Shell (crossed 50% HP threshold from this attack)
            if hp_before_move > target.max_hp * 0.5 and target.current_hp <= target.max_hp * 0.5 and attacker.ability != "sheerforce":
                if target.ability == "berserk":
                    target.apply_boost("spa", 1)
                elif target.ability == "angershell":
                    target.apply_boost("atk", 1)
                    target.apply_boost("spa", 1)
                    target.apply_boost("spe", 1)
                    target.apply_boost("def", -1)
                    target.apply_boost("spd", -1)
                    target.check_white_herb()

            if target.ability in ("electromorphosis", "windpower"):
                if target.ability == "electromorphosis" or move.id in _WIND_MOVES:
                    target.charge_active = True
            elif target.ability == "seedsower":
                _set_terrain(state, "grassyterrain", 5)
            elif target.ability == "sandspit" and not ("desolateland" in state.weather or "primordialsea" in state.weather or "deltastream" in state.weather):
                state.weather = {"sandstorm": 8 if target.item == "smoothrock" else 5}
                _check_paradox_abilities(state)

        if target.ability == "cottondown":
            for other_mon in state.active_pokemon + state.opponent_active_pokemon:
                if other_mon and not other_mon.fainted and other_mon != target:
                    _apply_external_stat_drops(other_mon, [("spe", -1)], source=target, state=state)

        # Color Change ability on target
        if target.ability == "colorchange" and not target.fainted and effective_type:
            target.types = (effective_type,)

    # Trapping binding moves (Fire Spin, Whirlpool, Sand Tomb, Magma Storm, Infestation, Snap Trap, Anchor Shot, Jaw Lock, Thousand Waves, Spirit Shackle)
    if not substitute_absorbed_all and move.id in ("firespin", "whirlpool", "sandtomb", "magmastorm", "infestation", "snaptrap", "anchorshot", "jawlock", "thousandwaves", "spiritshackle"):
        if "ghost" not in target.types and target.item != "shedshell" and not target.fainted:
            target.is_trapped = True

    # Damage-proportional Recoil (Flare Blitz, Brave Bird, Wave Crash, Wood Hammer, Wild Charge, Head Smash, etc.)
    if move.recoil > 0.0 and attacker.ability not in ("rockhead", "magicguard") and not attacker.fainted:
        attacker.take_damage(actual_damage_dealt * move.recoil)
        _check_hp_berries(attacker, state)

    # Shell Bell healing (1/8 of damage dealt)
    if (
        attacker.item == "shellbell"
        and actual_damage_dealt > 0
        and not attacker.fainted
        and not (attacker.ability == "sheerforce" and bool(move.secondary))
    ):
        attacker.heal(actual_damage_dealt / 8.0)

    # Drain moves (affected by target's Liquid Ooze and attacker's Big Root)
    if move.id in ("drainpunch", "gigadrain", "hornleech", "bitterblade", "paraboliccharge", "matchagotcha", "oblivionwing", "drainingkiss", "leechlife"):
        drain_ratio = 0.75 if move.id in ("oblivionwing", "drainingkiss") else 0.50
        if attacker.item == "bigroot":
            drain_ratio *= 1.3
        drain_amt = actual_damage_dealt * drain_ratio
        if target.ability == "liquidooze":
            if attacker.ability != "magicguard":
                attacker.take_damage(drain_amt)
        else:
            attacker.heal(drain_amt)

    # Secondary Effects on Target (blocked by Substitute, Covert Cloak, Shield Dust, or Attacker's Sheer Force)
    if (
        not substitute_absorbed_all
        and not target.fainted
        and target_eff_ability != "shielddust"
        and target.item != "covertcloak"
        and attacker.ability != "sheerforce"
    ):
        if move.id in ("snarl", "skittersmack", "mysticalfire", "strugglebug", "spiritbreak"):
            _apply_external_stat_drops(target, [("spa", -1)], source=attacker, state=state, ignore_ability=ignores_ability)
        elif move.id in ("icywind", "electroweb", "mudshot", "rocktomb", "bulldoze", "lowsweep", "pounce", "glaciate", "drumbeating"):
            _apply_external_stat_drops(target, [("spe", -1)], source=attacker, state=state, ignore_ability=ignores_ability)
        elif move.id in ("lunge", "tropkick", "chillingwater", "breakingswipe"):
            _apply_external_stat_drops(target, [("atk", -1)], source=attacker, state=state, ignore_ability=ignores_ability)
        elif move.id in ("firelash", "gravapple", "thunderouskick", "triplearrows"):
            _apply_external_stat_drops(target, [("def", -1)], source=attacker, state=state, ignore_ability=ignores_ability)
        elif move.id in ("appleacid", "applesacid", "luminaCrash", "luminacrash"):
            drop_amt = -2 if move.id in ("luminacrash", "luminaCrash") else -1
            _apply_external_stat_drops(target, [("spd", drop_amt)], source=attacker, state=state, ignore_ability=ignores_ability)
        elif move.id in ("acidspray", "acidSPRAY", "seedflare"):
            _apply_external_stat_drops(target, [("spd", -2)], source=attacker, state=state, ignore_ability=ignores_ability)
        elif move.id in ("mudslap", "muddywater", "leaftornado", "mirrorshot", "nightdaze", "octazooka"):
            _apply_external_stat_drops(target, [("accuracy", -1)], source=attacker, state=state, ignore_ability=ignores_ability)

        # Guaranteed status-inflicting attacks (Nuzzle, Inferno, Zap Cannon, Dynamic Punch)
        if move.id == "nuzzle" and target.status is None:
            if "electric" not in target.types and target_eff_ability not in ("limber", "comatose", "purifyingsalt"):
                target.status = "par"
                _check_status_berries(target, state)
        elif move.id == "inferno" and target.status is None:
            if "fire" not in target.types and target_eff_ability not in ("waterveil", "waterbubble", "thermalexchange", "comatose", "purifyingsalt"):
                target.status = "brn"
                _check_status_berries(target, state)
        elif move.id == "zapcannon" and target.status is None:
            if "electric" not in target.types and target_eff_ability not in ("limber", "comatose", "purifyingsalt"):
                target.status = "par"
                _check_status_berries(target, state)
        elif move.id == "dynamicpunch" and target.confusion_turns <= 0 and target_eff_ability != "owntempo":
            target.confusion_turns = 2
            _check_status_berries(target, state)

        # Champions Spicy Spray ability: guarantees burn on target when using a damaging move with secondary or pepper/fire move
        if attacker.ability == "spicyspray" and target.status is None and "fire" not in target.types:
            if target_eff_ability not in ("waterveil", "waterbubble", "thermalexchange", "comatose", "purifyingsalt"):
                target.status = "brn"
                _check_status_berries(target, state)

    # Dragon Tail / Circle Throw forced phazing switch
    if (
        not substitute_absorbed_all
        and move.id in ("dragontail", "circlethrow")
        and actual_damage_dealt > 0
        and not target.fainted
        and target_eff_ability not in ("suctioncups", "guarddog")
    ):
        _force_bench_switch(target, state)

    # Red Card, Eject Button, Emergency Exit / Wimp Out
    if actual_damage_dealt > 0 and not substitute_absorbed_all:
        if target.item == "redcard" and not target.fainted and not attacker.fainted and attacker.ability not in ("suctioncups", "guarddog"):
            target.consume_item()
            _force_bench_switch(attacker, state)
        elif target.item == "ejectbutton" and not target.fainted and attacker.ability != "sheerforce":
            target.consume_item()
            _force_bench_switch(target, state)
        elif (
            target.ability in ("emergencyexit", "wimpout")
            and not target.fainted
            and hp_before_move > target.max_hp * 0.5
            and target.current_hp <= target.max_hp * 0.5
            and attacker.ability != "sheerforce"
        ):
            _force_bench_switch(target, state)

    return True


def _execute_end_of_turn(state: SimulatedBattleState) -> None:
    eff_weather = _get_effective_weather(state)

    # 1. Wish delayed healing resolution
    for wish_key in list(state.wish_slots.keys()):
        turns_left, heal_amount = state.wish_slots[wish_key]
        turns_left -= 1
        if turns_left <= 0:
            del state.wish_slots[wish_key]
            if isinstance(wish_key, tuple) and len(wish_key) == 2:
                side_num, slot_num = int(wish_key[0]), int(wish_key[1])
            else:
                side_str, slot_str = str(wish_key).split("_")
                side_num, slot_num = int(side_str), int(slot_str)
            active_list = state.active_pokemon if side_num == 1 else state.opponent_active_pokemon
            wish_recipient = active_list[slot_num] if slot_num < len(active_list) else None
            if wish_recipient is not None and not wish_recipient.fainted:
                wish_recipient.heal(heal_amount)
        else:
            state.wish_slots[wish_key] = (turns_left, heal_amount)

    all_active_pokemon = [pokemon for pokemon in state.active_pokemon + state.opponent_active_pokemon if pokemon and not pokemon.fainted]

    for pokemon in all_active_pokemon:
        if pokemon.fainted:
            continue

        # Weather end-of-turn abilities & chip damage
        if any(w in eff_weather for w in ("sunnyday", "desolateland", "sun")):
            if pokemon.ability in ("solarpower", "dryskin") and pokemon.ability != "magicguard":
                pokemon.take_damage(pokemon.max_hp / 8.0)
        elif any(w in eff_weather for w in ("raindance", "primordialsea", "rain")):
            if pokemon.ability == "raindish":
                pokemon.heal(pokemon.max_hp / 16.0)
            elif pokemon.ability == "dryskin":
                pokemon.heal(pokemon.max_hp / 8.0)
            elif pokemon.ability == "hydration" and pokemon.status is not None:
                pokemon.status = None
                pokemon.toxic_turns = 0
                pokemon.sleep_turns = 0
        elif any(w in eff_weather for w in ("snowscape", "hail", "snow")):
            if pokemon.ability == "icebody":
                pokemon.heal(pokemon.max_hp / 16.0)

        if any(w in eff_weather for w in ("sandstorm", "sand")) and pokemon.ability != "magicguard":
            if (
                not any(type_name in pokemon.types for type_name in ("rock", "ground", "steel"))
                and pokemon.ability not in ("sandveil", "sandrush", "sandforce", "overcoat")
                and pokemon.item != "safetygoggles"
            ):
                pokemon.take_damage(pokemon.max_hp / 16.0)

        if pokemon.fainted:
            continue

        # Grassy Terrain healing
        if "grassyterrain" in state.fields and is_pokemon_grounded(pokemon, state.fields):
            pokemon.heal(pokemon.max_hp / 16.0)

        # Leftovers / Black Sludge
        if pokemon.item == "leftovers":
            pokemon.heal(pokemon.max_hp / 16.0)
        elif pokemon.item == "blacksludge":
            if "poison" in pokemon.types:
                pokemon.heal(pokemon.max_hp / 16.0)
            elif pokemon.ability != "magicguard":
                pokemon.take_damage(pokemon.max_hp / 8.0)

        if pokemon.fainted:
            continue

        # Leech Seed sap (1/8 max HP)
        if pokemon.leech_seed and pokemon.ability != "magicguard":
            opp_active = state.opponent_active_pokemon if pokemon.side == 1 else state.active_pokemon
            seeder = next((o for o in opp_active if o and not o.fainted), None)
            sap_dmg = pokemon.take_damage(pokemon.max_hp / 8.0)
            if seeder and sap_dmg > 0:
                heal_mult = 1.3 if seeder.item == "bigroot" else 1.0
                if pokemon.ability == "liquidooze":
                    if seeder.ability != "magicguard":
                        seeder.take_damage(sap_dmg * heal_mult)
                else:
                    seeder.heal(sap_dmg * heal_mult)

        if pokemon.fainted:
            continue

        # Poison Heal vs Status Damage (Burn, Poison, Toxic)
        if pokemon.ability == "poisonheal" and pokemon.status in ("psn", "tox"):
            pokemon.heal(pokemon.max_hp / 8.0)
        elif pokemon.ability != "magicguard":
            if pokemon.status == "brn":
                brn_divisor = 32.0 if pokemon.ability == "heatproof" else 16.0
                pokemon.take_damage(pokemon.max_hp / brn_divisor)
            elif pokemon.status == "psn":
                pokemon.take_damage(pokemon.max_hp / 8.0)
            elif pokemon.status == "tox":
                pokemon.toxic_turns += 1
                pokemon.take_damage(pokemon.max_hp * (pokemon.toxic_turns / 16.0))

        if pokemon.fainted:
            continue

        # Salt Cure chip damage (1/4 max HP for Water/Steel, 1/8 otherwise)
        if pokemon.salt_cure and pokemon.ability != "magicguard":
            salt_divisor = 4.0 if ("water" in pokemon.types or "steel" in pokemon.types) else 8.0
            pokemon.take_damage(pokemon.max_hp / salt_divisor)

        if pokemon.fainted:
            continue

        # Bad Dreams chip damage from opposing Darkrai
        if pokemon.status == "slp" or pokemon.ability == "comatose":
            opp_active = state.opponent_active_pokemon if pokemon.side == 1 else state.active_pokemon
            if any(o and not o.fainted and o.ability == "baddreams" for o in opp_active) and pokemon.ability != "magicguard":
                pokemon.take_damage(pokemon.max_hp / 8.0)

        if pokemon.fainted:
            continue

        # Syrup Bomb speed drop (-1 Spe each turn for 3 turns)
        if pokemon.syrup_bomb_turns > 0:
            pokemon.syrup_bomb_turns -= 1
            _apply_external_stat_drops(pokemon, [("spe", -1)], source=None, state=state)

        # Yawn delayed sleep resolution
        if pokemon.yawn_turns > 0:
            pokemon.yawn_turns -= 1
            if pokemon.yawn_turns == 0 and pokemon.status is None:
                if not ("electricterrain" in state.fields and is_pokemon_grounded(pokemon, state.fields)) and not ("mistyterrain" in state.fields and is_pokemon_grounded(pokemon, state.fields)):
                    if pokemon.ability not in ("insomnia", "vitalspirit", "comatose", "purifyingsalt", "sweetveil"):
                        pokemon.status = "slp"
                        pokemon.sleep_turns = 0
                        _check_status_berries(pokemon, state)

        # Flame Orb / Toxic Orb self-activation
        if pokemon.status is None:
            if pokemon.item == "flameorb" and "fire" not in pokemon.types and pokemon.ability not in ("waterveil", "waterbubble", "thermalexchange", "comatose", "purifyingsalt"):
                pokemon.status = "brn"
                _check_status_berries(pokemon, state)
            elif pokemon.item == "toxicorb" and "poison" not in pokemon.types and "steel" not in pokemon.types and pokemon.ability not in ("immunity", "comatose", "purifyingsalt", "pastelveil"):
                pokemon.status = "tox"
                pokemon.toxic_turns = 0
                _check_status_berries(pokemon, state)

        # Speed Boost, Moody, Shed Skin, Cud Chew, Harvest, Slow Start
        if pokemon.ability == "speedboost" and not pokemon.first_turn:
            pokemon.apply_boost("spe", 1)
        elif pokemon.ability == "moody":
            lowest_stat = min(("atk", "def", "spa", "spd", "spe"), key=lambda s: pokemon.boosts.get(s, 0))
            pokemon.apply_boost(lowest_stat, 2)
        elif pokemon.ability == "shedskin" and pokemon.status is not None and pokemon.toxic_turns >= 2:
            pokemon.status = None
            pokemon.toxic_turns = 0
            pokemon.sleep_turns = 0

        if pokemon.cud_chew_berry:
            berry_to_eat = pokemon.cud_chew_berry
            pokemon.cud_chew_berry = None
            _trigger_berry_effect(pokemon, berry_to_eat, consume=False)

        if pokemon.ability == "harvest" and pokemon.item in ("none", "") and pokemon.last_consumed_berry:
            if any(w in eff_weather for w in ("sunnyday", "desolateland", "sun")) or pokemon.current_hp <= pokemon.max_hp * 0.5:
                pokemon.item = pokemon.last_consumed_berry
                _check_hp_berries(pokemon, state)
                _check_status_berries(pokemon, state)

        if pokemon.ability == "slowstart" and pokemon.slow_start_turns > 0:
            pokemon.slow_start_turns -= 1

        # Volatile status decrements & cursed / perish song damage
        if pokemon.taunt_turns > 0:
            pokemon.taunt_turns -= 1
        if pokemon.throat_chop_turns > 0:
            pokemon.throat_chop_turns -= 1
        if pokemon.encore_turns > 0:
            pokemon.encore_turns -= 1
            if pokemon.encore_turns == 0:
                pokemon.encore_move = None
        if pokemon.disable_turns > 0:
            pokemon.disable_turns -= 1
            if pokemon.disable_turns == 0:
                pokemon.disabled_move = None
        if pokemon.cursed and pokemon.ability != "magicguard":
            pokemon.take_damage(pokemon.max_hp * 0.25)
        if pokemon.perish_turns > 0 and not pokemon.fainted:
            pokemon.perish_turns -= 1
            if pokemon.perish_turns == 0:
                pokemon.take_damage(pokemon.max_hp)

        _check_hp_berries(pokemon, state)
        _check_status_berries(pokemon, state)

        pokemon.first_turn = False
        pokemon.flinch = False
        pokemon.helping_hand = False
        pokemon.destiny_bond = False
        pokemon.endure_active = False
        pokemon.moved_this_turn = False
        pokemon.took_damage_this_turn = False
        if pokemon.roost_active:
            pokemon.types = pokemon.base_types
            pokemon.roost_active = False

    # Decrement Field and Condition Timers (preserving permanent entry hazards)
    weather_changed = False
    for condition_name in list(state.weather.keys()):
        state.weather[condition_name] -= 1
        if state.weather[condition_name] <= 0:
            del state.weather[condition_name]
            weather_changed = True

    fields_changed = False
    for condition_name in list(state.fields.keys()):
        state.fields[condition_name] -= 1
        if state.fields[condition_name] <= 0:
            del state.fields[condition_name]
            fields_changed = True

    if weather_changed or fields_changed:
        _check_paradox_abilities(state)

    _PERMANENT_HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
    for condition_name in list(state.side_conditions.keys()):
        if condition_name not in _PERMANENT_HAZARDS:
            state.side_conditions[condition_name] -= 1
            if state.side_conditions[condition_name] <= 0:
                del state.side_conditions[condition_name]

    for condition_name in list(state.opponent_side_conditions.keys()):
        if condition_name not in _PERMANENT_HAZARDS:
            state.opponent_side_conditions[condition_name] -= 1
            if state.opponent_side_conditions[condition_name] <= 0:
                del state.opponent_side_conditions[condition_name]

    _replace_fainted_slots(state, side=1)
    _replace_fainted_slots(state, side=2)


def _replace_fainted_slots(state: SimulatedBattleState, side: int) -> None:
    active_pokemon_list = state.active_pokemon if side == 1 else state.opponent_active_pokemon
    bench_pokemon_list = state.bench if side == 1 else state.opponent_bench

    for slot_index, pokemon in enumerate(active_pokemon_list):
        if pokemon and pokemon.fainted:
            replacement_pokemon = None
            for bench_index, candidate_bench_pokemon in enumerate(bench_pokemon_list):
                if candidate_bench_pokemon and not candidate_bench_pokemon.fainted:
                    replacement_pokemon = candidate_bench_pokemon
                    bench_pokemon_list.pop(bench_index)
                    break
            if replacement_pokemon:
                replacement_pokemon.slot = slot_index
                replacement_pokemon.first_turn = True
                active_pokemon_list[slot_index] = replacement_pokemon
                _sync_neutralizing_gas(state)
                _apply_entry_hazards(replacement_pokemon, state)
                if not replacement_pokemon.fainted:
                    _trigger_entry_abilities(replacement_pokemon, state)
            else:
                active_pokemon_list[slot_index] = pokemon
    _sync_neutralizing_gas(state)


