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

import src.constants as constants
import src.meta_priors as meta_priors

# In Champions, IVs are max
MAXIMUM_INDIVIDUAL_VALUE: int = 31

# ----------------- Static Data Loader -----------------

_MOVES_CACHE: dict[str, dict] = {}
_POKEDEX_CACHE: dict[str, dict] = {}
_TYPECHART_CACHE: dict[str, dict] = {}

_SOUND_MOVES: set[str] = {
    "hypervoice", "snarl", "boomburst", "roar", "sing", "perishsong",
    "bugbuzz", "clangoroussoul", "clangingscales", "sparklingaria",
    "uproar", "screech", "metalsound", "partingshot", "disarmingvoice",
    "echoedvoice", "round", "relicsong", "alluringvoice", "howl", "torchsong"
}

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

# ----------------- Simulated Core Objects -----------------

class SimulatedMove:
    __slots__ = (
        "id", "name", "type", "category", "base_power", "accuracy",
        "priority", "target", "flags", "secondary", "current_pp",
        "max_pp", "is_spread"
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
        self.secondary = secondary
        self.current_pp = int(current_pp)
        self.max_pp = int(max_pp)
        self.is_spread = is_spread if is_spread is not None else (self.target in ("allAdjacentFoes", "allAdjacent"))

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
        if cleaned_move_id in constants.PROTECT_MOVES:
            priority = 4
        elif cleaned_move_id in ("wideguard", "quickguard", "craftyshield"):
            priority = 3
        elif cleaned_move_id == "fakeout":
            priority = 3
        elif cleaned_move_id in ("followme", "ragepowder", "allyswitch"):
            priority = 2
        elif cleaned_move_id == "extremespeed":
            priority = 2
        elif cleaned_move_id in (
            "aquajet", "machpunch", "bulletpunch", "iceshard",
            "shadowsneak", "suckerpunch", "vacuumwave", "grassyglide"
        ):
            priority = 1
        elif cleaned_move_id == "trickroom":
            priority = -7

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
        return cloned_move

class SimulatedPokemon:
    __slots__ = (
        "species", "name", "types", "base_types", "roost_active", "protean_activated",
        "level", "current_hp", "max_hp",
        "current_hp_fraction", "base_stats", "stats", "boosts",
        "status", "toxic_turns", "item", "ability", "moves",
        "protect_counter", "is_protected", "first_turn", "is_mega", "can_mega_evolve",
        "fainted", "side", "slot", "flinch", "weight", "helping_hand",
        "taunt_turns", "encore_turns", "encore_move", "disabled_move",
        "disable_turns", "torment", "destiny_bond", "cursed", "last_move_used", "just_flinched",
        "perish_turns", "is_trapped", "imprison"
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
    ):
        self.species = clean_id(species)
        self.name = name or self.species
        self.types = tuple(clean_id(type_name) for type_name in types)
        self.base_types = self.types
        self.roost_active: bool = False
        self.protean_activated: bool = False
        self.level = level
        self.max_hp = max(1.0, float(max_hp))
        self.current_hp = max(0.0, min(self.max_hp, float(hp)))
        self.current_hp_fraction = self.current_hp / self.max_hp

        self.base_stats = base_stats or {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80}
        self.stats = stats or self._calculate_stats_from_base()
        self.boosts = boosts if boosts is not None else {
            "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0
        }
        self.status = clean_id(status) if status else None
        self.toxic_turns = 0
        self.item = clean_id(item)
        self.ability = clean_id(ability)
        self.moves = moves if moves is not None else {}
        self.protect_counter = protect_counter
        self.is_protected: bool = False
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

    def _calculate_stats_from_base(
        self,
        effort_values_spread: dict[str, int] | None = None,
        nature: str = "hardy"
    ) -> dict[str, int]:
        """Calculates level 50 competitive stats with permanently maxed IVs (31) for Pokemon Champions."""
        effort_values = effort_values_spread or {}
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
            weight = 80.0
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
        )
        simulated_pokemon.stats = simulated_pokemon._calculate_stats_from_base(
            effort_values_spread=spread.get("evs"), nature=spread.get("nature", "hardy")
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
        return cloned_pokemon

    def take_damage(self, damage_amount: float) -> float:
        """Deducts damage and updates faint status. Returns actual damage taken."""
        actual_damage = min(self.current_hp, max(0.0, damage_amount))
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
        """Applies stat stage modification [-6, +6]."""
        if stat_name in self.boosts:
            current_boost = self.boosts[stat_name]
            self.boosts[stat_name] = max(-6, min(6, current_boost + stages))

    def clear_boosts(self) -> None:
        for stat_name in self.boosts:
            self.boosts[stat_name] = 0

    def damage_multiplier(self, attacking_type: str, has_gravity: bool = False) -> float:
        return get_type_effectiveness(attacking_type, self.types, has_gravity=has_gravity)

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

        # Item modifiers
        if self.item == "choicescarf":
            effective_speed *= 1.5
        elif self.item == "ironball":
            effective_speed *= 0.5

        # Weather abilities
        if "raindance" in weather or "primordialsea" in weather:
            if self.ability == "swiftswim":
                effective_speed *= 2.0
        elif "sunnyday" in weather or "desolateland" in weather:
            if self.ability == "chlorophyll":
                effective_speed *= 2.0
        elif "sandstorm" in weather:
            if self.ability == "sandrush":
                effective_speed *= 2.0
        elif "snow" in weather or "hail" in weather:
            if self.ability == "slushrush":
                effective_speed *= 2.0

        # Electric Terrain
        if "electricterrain" in fields and self.ability == "surgesurfer":
            effective_speed *= 2.0

        return effective_speed

    def calculate_effective_attack(self, is_special: bool, weather: dict[str, int], fields: dict[str, int]) -> float:
        stat_name = "spa" if is_special else "atk"
        raw_attack = float(self.stats.get(stat_name, 80))
        attack_boost = self.boosts.get(stat_name, 0)
        attack_multiplier = (2.0 + attack_boost) / 2.0 if attack_boost >= 0 else 2.0 / (2.0 - attack_boost)

        if not is_special:
            if self.status == "brn" and self.ability != "guts":
                attack_multiplier *= 0.5
            if self.item == "choiceband":
                attack_multiplier *= 1.5
            if self.ability in ("hugepower", "purepower"):
                attack_multiplier *= 2.0
        else:
            if self.item == "choicespecs":
                attack_multiplier *= 1.5

        return raw_attack * attack_multiplier

    def calculate_effective_defense(self, is_special: bool, weather: dict[str, int], fields: dict[str, int]) -> float:
        stat_name = "spd" if is_special else "def"
        raw_defense = float(self.stats.get(stat_name, 80))
        defense_boost = self.boosts.get(stat_name, 0)
        defense_multiplier = (2.0 + defense_boost) / 2.0 if defense_boost >= 0 else 2.0 / (2.0 - defense_boost)

        if is_special:
            if self.item == "assaultvest":
                defense_multiplier *= 1.5
            if "sandstorm" in weather and "rock" in self.types:
                defense_multiplier *= 1.5
            if self.ability == "icescales":
                defense_multiplier *= 2.0
        else:
            if ("snow" in weather or "hail" in weather) and "ice" in self.types:
                defense_multiplier *= 1.5
            if self.ability == "furcoat":
                defense_multiplier *= 2.0

        if self.item == "eviolite":
            defense_multiplier *= 1.5

        return raw_defense * defense_multiplier

class SimulatedBattleState:
    """
    Snapshot of battle state for Doubles (2 active per side).
    Supports high-speed deep cloning and forward simulation.
    """
    battle_format: str
    side1_active: list[Optional[SimulatedPokemon]]
    side2_active: list[Optional[SimulatedPokemon]]
    side1_bench: list[SimulatedPokemon]
    side2_bench: list[SimulatedPokemon]
    weather: dict[str, int]
    fields: dict[str, int]
    side1_conditions: dict[str, int]
    side2_conditions: dict[str, int]
    turn: int
    hit_counts: dict[str, int]
    force_switch: list[bool]
    trapped: list[bool]
    _raw_battle_reference: Optional[Any]
    redirect_target: dict[int, Optional[tuple[int, str]]]
    wide_guard: dict[int, bool]
    quick_guard: dict[int, bool]
    crafty_shield: dict[int, bool]

    side1_mega_used: bool
    side2_mega_used: bool

    __slots__ = (
        "battle_format", "side1_active", "side2_active", "side1_bench", "side2_bench",
        "weather", "fields", "side1_conditions", "side2_conditions", "turn",
        "hit_counts", "force_switch", "trapped", "_raw_battle_reference",
        "redirect_target", "wide_guard", "quick_guard", "crafty_shield",
        "side1_mega_used", "side2_mega_used"
    )

    def __init__(
        self,
        battle_format: str = "doubles",
        side1_active: Optional[list[Optional[SimulatedPokemon]]] = None,
        side2_active: Optional[list[Optional[SimulatedPokemon]]] = None,
        side1_bench: Optional[list[SimulatedPokemon]] = None,
        side2_bench: Optional[list[SimulatedPokemon]] = None,
        weather: Optional[dict[str, int]] = None,
        fields: Optional[dict[str, int]] = None,
        side1_conditions: Optional[dict[str, int]] = None,
        side2_conditions: Optional[dict[str, int]] = None,
        turn: int = 1,
        hit_counts: Optional[dict[str, int]] = None,
    ):
        self.battle_format = "doubles"
        default_active: list[Optional[SimulatedPokemon]] = [None, None]
        self.side1_active = list(side1_active) if side1_active is not None else list(default_active)
        self.side2_active = list(side2_active) if side2_active is not None else list(default_active)
        self.side1_bench = list(side1_bench) if side1_bench is not None else []
        self.side2_bench = list(side2_bench) if side2_bench is not None else []
        self.weather = weather if weather is not None else {}
        self.fields = fields if fields is not None else {}
        self.side1_conditions = side1_conditions if side1_conditions is not None else {}
        self.side2_conditions = side2_conditions if side2_conditions is not None else {}
        self.turn = turn
        self.hit_counts = hit_counts if hit_counts is not None else {}
        self.force_switch = [False, False]
        self.trapped = [False, False]
        self._raw_battle_reference = None
        self.redirect_target = {1: None, 2: None}
        self.wide_guard = {1: False, 2: False}
        self.quick_guard = {1: False, 2: False}
        self.crafty_shield = {1: False, 2: False}
        self.side1_mega_used = False
        self.side2_mega_used = False

    def clone(self) -> "SimulatedBattleState":
        """Fast deep clone for game tree rollouts."""
        cloned_state = SimulatedBattleState.__new__(SimulatedBattleState)
        cloned_state.battle_format = self.battle_format
        cloned_state.side1_active = [pokemon.clone() if pokemon else None for pokemon in self.side1_active]
        cloned_state.side2_active = [pokemon.clone() if pokemon else None for pokemon in self.side2_active]
        cloned_state.side1_bench = [pokemon.clone() for pokemon in self.side1_bench]
        cloned_state.side2_bench = [pokemon.clone() for pokemon in self.side2_bench]
        cloned_state.weather = self.weather.copy()
        cloned_state.fields = self.fields.copy()
        cloned_state.side1_conditions = self.side1_conditions.copy()
        cloned_state.side2_conditions = self.side2_conditions.copy()
        cloned_state.turn = self.turn
        cloned_state.hit_counts = self.hit_counts.copy()
        cloned_state.force_switch = self.force_switch.copy()
        cloned_state.trapped = self.trapped.copy()
        cloned_state._raw_battle_reference = self._raw_battle_reference
        cloned_state.redirect_target = self.redirect_target.copy()
        cloned_state.wide_guard = self.wide_guard.copy()
        cloned_state.quick_guard = self.quick_guard.copy()
        cloned_state.crafty_shield = self.crafty_shield.copy()
        cloned_state.side1_mega_used = self.side1_mega_used
        cloned_state.side2_mega_used = self.side2_mega_used
        return cloned_state

    def all_side1_pokemon(self) -> list[SimulatedPokemon]:
        return [pokemon for pokemon in self.side1_active if pokemon] + self.side1_bench

    def all_side2_pokemon(self) -> list[SimulatedPokemon]:
        return [pokemon for pokemon in self.side2_active if pokemon] + self.side2_bench

    def alive_side1_count(self) -> int:
        return sum(1 for pokemon in self.all_side1_pokemon() if not pokemon.fainted)

    def alive_side2_count(self) -> int:
        return sum(1 for pokemon in self.all_side2_pokemon() if not pokemon.fainted)

    def is_finished(self) -> bool:
        return self.alive_side1_count() == 0 or self.alive_side2_count() == 0

    def winner(self) -> Optional[int]:
        side1_alive_count = self.alive_side1_count()
        side2_alive_count = self.alive_side2_count()
        if side1_alive_count == 0 and side2_alive_count == 0:
            return 0  # Tie
        if side1_alive_count == 0:
            return 2  # Side 2 won
        if side2_alive_count == 0:
            return 1  # Side 1 won
        return None  # Battle ongoing

    def is_pokemon_trapped(self, pokemon: Optional[SimulatedPokemon]) -> bool:
        """Determines if a Pokemon is trapped and prohibited from switching."""
        if not pokemon or pokemon.fainted:
            return False
        if "ghost" in pokemon.types:
            return False
        if pokemon.item == "shedshell":
            return False
        if pokemon.is_trapped:
            return True

        opponents = self.side2_active if pokemon.side == 1 else self.side1_active
        for opp in opponents:
            if opp and not opp.fainted:
                if opp.ability == "shadowtag" and pokemon.ability != "shadowtag":
                    return True
                if opp.ability == "arenatrap":
                    is_grounded = ("gravity" in self.fields) or (
                        "flying" not in pokemon.types and pokemon.ability != "levitate" and pokemon.item != "airballoon"
                    )
                    if is_grounded:
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

        # Weather mapping
        if hasattr(battle, "weather"):
            for weather_condition in battle.weather:
                state.weather[clean_id(weather_condition)] = 5

        # Fields mapping
        if hasattr(battle, "fields"):
            for field_condition in battle.fields:
                state.fields[clean_id(field_condition)] = 5

        # Side conditions
        if hasattr(battle, "side_conditions"):
            for side_condition in battle.side_conditions:
                state.side1_conditions[clean_id(side_condition)] = 4
        if hasattr(battle, "opponent_side_conditions"):
            for side_condition in battle.opponent_side_conditions:
                state.side2_conditions[clean_id(side_condition)] = 4

        # Parse Side 1 (Self) - 2 active slots
        raw_side1_active = battle.active_pokemon if isinstance(battle.active_pokemon, (list, tuple)) else [battle.active_pokemon]
        for slot_index in range(2):
            pokemon_object = raw_side1_active[slot_index] if slot_index < len(raw_side1_active) else None
            if pokemon_object and not pokemon_object.fainted:
                simulated_pokemon = cls._convert_poke_env_mon(pokemon_object, side=1, slot=slot_index, fill_priors=False)
                state.side1_active[slot_index] = simulated_pokemon

        # Parse Side 1 Bench
        for pokemon_object in battle.team.values():
            if pokemon_object and not pokemon_object.fainted and pokemon_object not in raw_side1_active:
                if getattr(pokemon_object, "selected_in_teampreview", True):
                    simulated_pokemon = cls._convert_poke_env_mon(pokemon_object, side=1, slot=-1, fill_priors=False)
                    state.side1_bench.append(simulated_pokemon)

        # Parse Side 2 (Opponent) - 2 active slots
        raw_side2_active = list(battle.opponent_active_pokemon) if isinstance(battle.opponent_active_pokemon, (list, tuple)) else [battle.opponent_active_pokemon]
        for slot_index in range(2):
            pokemon_object = raw_side2_active[slot_index] if slot_index < len(raw_side2_active) else None
            if pokemon_object and not getattr(pokemon_object, "fainted", False):
                simulated_pokemon = cls._convert_poke_env_mon(
                    pokemon_object, side=2, slot=slot_index, fill_priors=fill_unrevealed
                )
                state.side2_active[slot_index] = simulated_pokemon

        # Parse Side 2 Bench
        for pokemon_object in battle.opponent_team.values():
            if pokemon_object and not getattr(pokemon_object, "fainted", False) and pokemon_object not in raw_side2_active:
                if getattr(pokemon_object, "revealed", True):
                    simulated_pokemon = cls._convert_poke_env_mon(
                        pokemon_object, side=2, slot=-1, fill_priors=fill_unrevealed
                    )
                    state.side2_bench.append(simulated_pokemon)

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
                if is_tr and slot_idx < len(state.side1_active):
                    active_pokemon = state.side1_active[slot_idx]
                    if active_pokemon is not None:
                        active_pokemon.is_trapped = True

        state.side1_mega_used = any(p and p.is_mega for p in state.side1_active + state.side1_bench)
        state.side2_mega_used = any(p and p.is_mega for p in state.side2_active + state.side2_bench)
        state._raw_battle_reference = battle
        return state

    @staticmethod
    def _convert_poke_env_mon(mon: Any, side: int, slot: int, fill_priors: bool) -> SimulatedPokemon:
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

        # Sync volatile effects from live battle (Taunt, Encore, Disable, Torment, Perish Song, Trapping)
        effects_dict = getattr(mon, "effects", {})
        if not isinstance(effects_dict, dict):
            effects_dict = getattr(mon, "_effects", {})
        if isinstance(effects_dict, dict):
            for eff, count in effects_dict.items():
                eff_name = getattr(eff, "name", str(eff)).upper()
                if "TAUNT" in eff_name:
                    simulated_pokemon.taunt_turns = max(1, 3 - count) if isinstance(count, int) else 3
                elif "ENCORE" in eff_name:
                    simulated_pokemon.encore_turns = max(1, 3 - count) if isinstance(count, int) else 3
                    if getattr(mon, "last_move", None):
                        simulated_pokemon.encore_move = clean_id(mon.last_move.id)
                elif "DISABLE" in eff_name:
                    simulated_pokemon.disable_turns = max(1, 4 - count) if isinstance(count, int) else 4
                    if getattr(mon, "last_move", None):
                        simulated_pokemon.disabled_move = clean_id(mon.last_move.id)
                elif "TORMENT" in eff_name:
                    simulated_pokemon.torment = True
                elif "PERISH" in eff_name:
                    if "PERISH3" in eff_name: simulated_pokemon.perish_turns = 3
                    elif "PERISH2" in eff_name: simulated_pokemon.perish_turns = 2
                    elif "PERISH1" in eff_name: simulated_pokemon.perish_turns = 1
                elif "TRAP" in eff_name:
                    simulated_pokemon.is_trapped = True
                elif "IMPRISON" in eff_name:
                    simulated_pokemon.imprison = True

        if getattr(mon, "last_move", None):
            simulated_pokemon.last_move_used = clean_id(mon.last_move.id)

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
    and dynamic effective speed within brackets (inverted if Trick Room active).
    4. Moves execute with KO checks, Protect blocks, damage calculations, stat modifications.
    5. End of turn status damage, weather damage, passive recovery, field timer countdowns.
    """
    next_state = state.clone()
    next_state.turn += 1
    next_state.redirect_target = {1: None, 2: None}
    next_state.wide_guard = {1: False, 2: False}
    next_state.quick_guard = {1: False, 2: False}
    next_state.crafty_shield = {1: False, 2: False}

    # Reset per-turn defensive shields for all active Pokemon
    for active_pkmn in next_state.side1_active + next_state.side2_active:
        if active_pkmn:
            active_pkmn.is_protected = False
            active_pkmn.helping_hand = False
            active_pkmn.just_flinched = False

    player1_actions = [action_player1] if isinstance(action_player1, SimulatedAction) else list(action_player1)
    player2_actions = [action_player2] if isinstance(action_player2, SimulatedAction) else list(action_player2)

    # Reset protect streak for slots that pass
    for action in player1_actions:
        if action.type == "pass" and action.slot < len(next_state.side1_active):
            pkmn = next_state.side1_active[action.slot]
            if pkmn:
                pkmn.protect_counter = 0
                pkmn.is_protected = False
    for action in player2_actions:
        if action.type == "pass" and action.slot < len(next_state.side2_active):
            pkmn = next_state.side2_active[action.slot]
            if pkmn:
                pkmn.protect_counter = 0
                pkmn.is_protected = False

    # 1. Switch Phase
    _execute_switches(next_state, player1_actions, side=1)
    _execute_switches(next_state, player2_actions, side=2)

    # 2. Mega Evolution Phase
    for action in player1_actions:
        if action.type == "move" and action.mega and action.slot < len(next_state.side1_active):
            _trigger_mega(next_state.side1_active[action.slot], next_state)
    for action in player2_actions:
        if action.type == "move" and action.mega and action.slot < len(next_state.side2_active):
            _trigger_mega(next_state.side2_active[action.slot], next_state)

    # 3. Collect and Sort Move Executions
    move_execution_queue = []
    is_trick_room_active = "trickroom" in next_state.fields

    for action in player1_actions:
        if action.type == "move" and action.move_id and action.slot < len(next_state.side1_active):
            user_pokemon = next_state.side1_active[action.slot]
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
                else:
                    continue
                effective_speed = user_pokemon.calculate_effective_speed(
                    next_state.weather, next_state.fields, next_state.side1_conditions
                )
                move_execution_queue.append((move_object.priority, effective_speed, 1, action.slot, action))

    for action in player2_actions:
        if action.type == "move" and action.move_id and action.slot < len(next_state.side2_active):
            user_pokemon = next_state.side2_active[action.slot]
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
                else:
                    continue
                effective_speed = user_pokemon.calculate_effective_speed(
                    next_state.weather, next_state.fields, next_state.side2_conditions
                )
                move_execution_queue.append((move_object.priority, effective_speed, 2, action.slot, action))

    def move_priority_sort_key(queue_item: tuple) -> tuple:
        priority, speed, side_index, slot_index, _ = queue_item
        adjusted_speed = -speed if is_trick_room_active else speed
        return (priority, adjusted_speed, -side_index)

    move_execution_queue.sort(key=move_priority_sort_key, reverse=True)

    # 4. Move Execution Loop
    for _, _, side_index, slot_index, action in move_execution_queue:
        active_pokemon_list = next_state.side1_active if side_index == 1 else next_state.side2_active
        opponent_pokemon_list = next_state.side2_active if side_index == 1 else next_state.side1_active

        user_pokemon = active_pokemon_list[slot_index]
        if not user_pokemon or user_pokemon.fainted:
            continue

        if user_pokemon.flinch:
            user_pokemon.flinch = False
            user_pokemon.just_flinched = True
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
        if not current_move:
            continue

        # Protect handling
        if current_move.id in constants.PROTECT_MOVES:
            protect_streak = user_pokemon.protect_counter
            protect_probability = 1.0 / (3.0 ** protect_streak)
            if protect_probability > 0.5:
                user_pokemon.is_protected = True
                user_pokemon.protect_counter += 1
            else:
                user_pokemon.is_protected = False
                user_pokemon.protect_counter = 0
            user_pokemon.last_move_used = current_move.id
            continue
        else:
            # Any non-Protect move breaks the consecutive protect streak
            user_pokemon.is_protected = False
            user_pokemon.protect_counter = 0

        # Target Resolution
        targets = []
        partner = active_pokemon_list[1 - slot_index] if len(active_pokemon_list) > 1 else None

        # Check self-targeting moves
        is_self_move = current_move.id in (
            "swordsdance", "nastyplot", "calmmind", "dragondance", "bulkup", "quiverdance",
            "irondefense", "amnesia", "agility", "rockpolish", "autotomize", "shiftgear",
            "shellsmash", "bellydrum", "tidyup", "victorydance", "clangoroussoul", "geomancy",
            "coil", "growth", "workup", "cosmicpower", "defendorder", "cottonguard", "tailglow",
            "acidarmor", "barrier", "tailwind", "trickroom", "reflect", "lightscreen", "auroraveil",
            "sunnyday", "raindance", "sandstorm", "snowscape", "hail", "psychicterrain",
            "electricterrain", "grassyterrain", "mistyterrain", "haze", "recover", "roost",
            "slackoff", "softboiled", "milkdrink", "healorder", "wish", "synthesis", "moonlight",
            "morningsun", "lifedew", "howl", "wideguard", "quickguard", "craftyshield",
            "followme", "ragepowder", "allyswitch", "gravity", "chillyreception", "destinybond",
            "conversion", "camouflage", "perishsong", "imprison"
        ) or (current_move.id == "curse" and "ghost" not in user_pokemon.types)

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
                continue
        elif current_move.is_spread:
            targets = [opp for opp in opponent_pokemon_list if opp and not opp.fainted]
            if current_move.id in ("earthquake", "surf", "discharge", "bulldoze", "explosion", "boomburst") and partner and not partner.fainted:
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

            # Redirection check (Follow Me / Rage Powder)
            target_side = 2 if side_index == 1 else 1
            redir_info = next_state.redirect_target.get(target_side)
            if redir_info is not None and targets:
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

        valid_targets = [target for target in targets if target is not None]
        if not valid_targets:
            continue

        for target_pokemon in valid_targets:
            _execute_single_move(
                next_state,
                user_pokemon,
                target_pokemon,
                current_move,
                is_spread=len(valid_targets) > 1,
                move_execution_queue=move_execution_queue
            )

    # 5. End of Turn Effects
    _execute_end_of_turn(next_state)

    return next_state

def _execute_switches(state: SimulatedBattleState, actions: list[SimulatedAction], side: int) -> None:
    active_pokemon_list = state.side1_active if side == 1 else state.side2_active
    bench_pokemon_list = state.side1_bench if side == 1 else state.side2_bench
    opponent_pokemon_list = state.side2_active if side == 1 else state.side1_active

    for action in actions:
        if action.type == "switch" and action.slot < len(active_pokemon_list) and action.switch_idx < len(bench_pokemon_list):
            switched_out_pokemon = active_pokemon_list[action.slot]
            switched_in_pokemon = bench_pokemon_list[action.switch_idx]
            if switched_out_pokemon and state.is_pokemon_trapped(switched_out_pokemon):
                continue  # Blocked from switching due to being trapped!

            if switched_in_pokemon and not switched_in_pokemon.fainted:
                # Reset boosts and volatile conditions on switch out
                if switched_out_pokemon:
                    switched_out_pokemon.clear_boosts()
                    switched_out_pokemon.first_turn = False
                    switched_out_pokemon.taunt_turns = 0
                    switched_out_pokemon.encore_turns = 0
                    switched_out_pokemon.encore_move = None
                    switched_out_pokemon.disabled_move = None
                    switched_out_pokemon.disable_turns = 0
                    switched_out_pokemon.torment = False
                    switched_out_pokemon.destiny_bond = False
                    switched_out_pokemon.cursed = False
                    switched_out_pokemon.last_move_used = None
                    switched_out_pokemon.protect_counter = 0
                    switched_out_pokemon.is_protected = False
                    switched_out_pokemon.types = switched_out_pokemon.base_types
                    switched_out_pokemon.roost_active = False
                    switched_out_pokemon.protean_activated = False
                    switched_out_pokemon.flinch = False
                    switched_out_pokemon.just_flinched = False
                    switched_out_pokemon.perish_turns = 0
                    switched_out_pokemon.is_trapped = False
                    switched_out_pokemon.imprison = False
                    switched_out_pokemon.slot = -1
                    bench_pokemon_list[action.switch_idx] = switched_out_pokemon

                switched_in_pokemon.first_turn = True
                switched_in_pokemon.protect_counter = 0
                switched_in_pokemon.is_protected = False
                switched_in_pokemon.protean_activated = False
                switched_in_pokemon.flinch = False
                switched_in_pokemon.just_flinched = False
                switched_in_pokemon.slot = action.slot
                active_pokemon_list[action.slot] = switched_in_pokemon

                # Intimidate trigger
                if switched_in_pokemon.ability == "intimidate":
                    for opponent_pokemon in opponent_pokemon_list:
                        if opponent_pokemon and not opponent_pokemon.fainted:
                            if opponent_pokemon.ability not in ("clearbody", "innerfocus", "whitesmoke", "hypercutter"):
                                if opponent_pokemon.ability == "defiant":
                                    opponent_pokemon.apply_boost("atk", 2)
                                elif opponent_pokemon.ability == "competitive":
                                    opponent_pokemon.apply_boost("spa", 2)
                                else:
                                    opponent_pokemon.apply_boost("atk", -1)

def _trigger_mega(pokemon: Optional[SimulatedPokemon], state: Optional[SimulatedBattleState] = None) -> None:
    if not pokemon or pokemon.is_mega or pokemon.fainted:
        return
    if state is not None:
        if pokemon.side == 1 and state.side1_mega_used:
            return
        if pokemon.side == 2 and state.side2_mega_used:
            return
    cleaned_species = pokemon.species
    mega_stones = constants.BASE_TO_MEGA_STONE.get(cleaned_species, [])
    if not mega_stones and cleaned_species != "rayquaza":
        return
    if cleaned_species != "rayquaza" and pokemon.item not in mega_stones:
        return
    mega_species = None
    for candidate_mega_species, stone_name in constants.MEGA_TO_MEGA_STONE.items():
        if stone_name in mega_stones and (candidate_mega_species.startswith(cleaned_species) or cleaned_species in candidate_mega_species):
            mega_species = candidate_mega_species
            break

    if mega_species and mega_species in constants.MEGA_SPECIES_STATS:
        mega_data = constants.MEGA_SPECIES_STATS[mega_species]
        pokemon.species = mega_species
        pokemon.types = tuple(clean_id(type_name) for type_name in mega_data["types"])
        pokemon.base_types = pokemon.types
        pokemon.ability = clean_id(mega_data["ability"])
        pokemon.base_stats = mega_data["base_stats"]
        pokemon.stats = pokemon._calculate_stats_from_base()
        pokemon.is_mega = True
        if state is not None:
            if pokemon.side == 1:
                state.side1_mega_used = True
            else:
                state.side2_mega_used = True

def _execute_single_move(
    state: SimulatedBattleState,
    attacker: SimulatedPokemon,
    target: SimulatedPokemon,
    move: SimulatedMove,
    is_spread: bool = False,
    move_execution_queue: Optional[list[Any]] = None,
) -> None:
    if attacker.fainted or target.fainted:
        return

    if attacker.slot not in (0, 1):
        active_list = state.side1_active if attacker.side == 1 else state.side2_active
        if attacker in active_list:
            attacker.slot = active_list.index(attacker)

    is_ally = (target.side == attacker.side and target != attacker)

    # 1. Attacker volatile & item restrictions (Taunt, Disable, Encore, Torment, Choice lock, Assault Vest, Imprison)
    if move.id != "struggle":
        if attacker.taunt_turns > 0 and move.category == "status":
            return
        if attacker.item == "assaultvest" and move.category == "status":
            return
        if attacker.disable_turns > 0 and move.id == attacker.disabled_move:
            return
        if attacker.encore_turns > 0 and attacker.encore_move and move.id != attacker.encore_move:
            return
        if attacker.torment and attacker.last_move_used and move.id == attacker.last_move_used:
            return
        if attacker.item in ("choiceband", "choicespecs", "choicescarf"):
            if attacker.last_move_used and attacker.last_move_used in attacker.moves and move.id != attacker.last_move_used:
                return

        # Imprison restriction: cannot use any move that an active opposing Imprison user knows
        opponents = state.side2_active if attacker.side == 1 else state.side1_active
        for opp in opponents:
            if opp and not opp.fainted and opp.imprison and move.id in opp.moves:
                return

    # 2. Defensive shield protections (Protect, Wide Guard, Quick Guard, Crafty Shield)
    if target.is_protected and "protect" in move.flags and not is_ally:
        return
    if is_spread and not is_ally and state.wide_guard.get(target.side, False):
        return
    if move.priority > 0 and not is_ally and state.quick_guard.get(target.side, False):
        return
    if move.category == "status" and not is_ally and state.crafty_shield.get(target.side, False):
        return

    # 3. Psychic Terrain priority block (grounded target protected against non-ally priority)
    if "psychicterrain" in state.fields and move.priority > 0 and not is_ally:
        if "flying" not in target.types and target.ability != "levitate":
            return

    # Record last move used
    attacker.last_move_used = move.id

    # 4. Fake Out flinch logic
    if move.id == "fakeout":
        if attacker.first_turn and not is_ally:
            if "ghost" not in target.types and target.ability != "innerfocus":
                target.flinch = True

    # 1. Ally Support & Targeting Moves
    if move.id == "helpinghand":
        target.helping_hand = True
        return
    elif move.id == "coaching":
        target.apply_boost("atk", 1)
        target.apply_boost("def", 1)
        return
    elif move.id == "decorate":
        target.apply_boost("atk", 2)
        target.apply_boost("spa", 2)
        return
    elif move.id == "aromaticmist":
        target.apply_boost("spd", 1)
        return
    elif move.id == "healpulse":
        heal_fraction = 0.75 if attacker.ability == "megalauncher" else 0.50
        target.heal(target.max_hp * heal_fraction)
        return
    elif move.id == "floralhealing":
        heal_fraction = 0.66 if "grassyterrain" in state.fields else 0.50
        target.heal(target.max_hp * heal_fraction)
        return
    elif move.id == "lifedew":
        attacker.heal(attacker.max_hp * 0.25)
        active_list = state.side1_active if attacker.side == 1 else state.side2_active
        partner = next((p for p in active_list if p is not None and p != attacker), None)
        if partner and not partner.fainted:
            partner.heal(partner.max_hp * 0.25)
        return
    elif move.id == "howl":
        attacker.apply_boost("atk", 1)
        active_list = state.side1_active if attacker.side == 1 else state.side2_active
        partner = next((p for p in active_list if p is not None and p != attacker), None)
        if partner and not partner.fainted:
            partner.apply_boost("atk", 1)
        return
    elif move.id == "pollenpuff" and is_ally:
        target.heal(target.max_hp * 0.50)
        return
    elif move.id == "spicyextract":
        target.apply_boost("atk", 2)
        target.apply_boost("def", -2)
        return
    elif move.id == "swagger":
        target.apply_boost("atk", 2)
        return
    elif move.id == "flatter":
        target.apply_boost("spa", 1)
        return

    # 2. Status Moves (Self & Opponent)
    if move.category == "status":
        # Target debuffs
        if move.id == "partingshot":
            target.apply_boost("atk", -1)
            target.apply_boost("spa", -1)
        elif move.id == "screech":
            target.apply_boost("def", -2)
        elif move.id in ("faketears", "metalsound"):
            target.apply_boost("spd", -2)
        elif move.id in ("charm", "featherdance"):
            target.apply_boost("atk", -2)
        elif move.id == "eerieimpulse":
            target.apply_boost("spa", -2)
        elif move.id in ("scaryface", "stringshot"):
            target.apply_boost("spe", -2)
        elif move.id == "haze":
            for active_pkmn in state.side1_active + state.side2_active:
                if active_pkmn and not active_pkmn.fainted:
                    active_pkmn.clear_boosts()
        elif move.id == "clearsmog":
            target.clear_boosts()

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
        elif move.id == "bellydrum":
            attacker.take_damage(attacker.max_hp * 0.50)
            attacker.boosts["atk"] = 6
        elif move.id == "tidyup":
            attacker.apply_boost("atk", 1)
            attacker.apply_boost("spe", 1)
        elif move.id == "victorydance":
            attacker.apply_boost("atk", 1)
            attacker.apply_boost("def", 1)
            attacker.apply_boost("spe", 1)
        elif move.id == "clangoroussoul":
            attacker.take_damage(attacker.max_hp * 0.33)
            for stat_key in ("atk", "def", "spa", "spd", "spe"):
                attacker.apply_boost(stat_key, 1)
        elif move.id == "geomancy":
            attacker.apply_boost("spa", 2)
            attacker.apply_boost("spd", 2)
            attacker.apply_boost("spe", 2)
        elif move.id == "coil":
            attacker.apply_boost("atk", 1)
            attacker.apply_boost("def", 1)
            attacker.apply_boost("accuracy", 1)
        elif move.id == "growth":
            boost_amount = 2 if ("sunnyday" in state.weather or "desolateland" in state.weather) else 1
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

        # Field and Side Conditions
        elif move.id == "tailwind":
            side_conditions = state.side1_conditions if attacker.side == 1 else state.side2_conditions
            if side_conditions.get("tailwind", 0) == 0:
                side_conditions["tailwind"] = 4
        elif move.id == "trickroom":
            if "trickroom" in state.fields:
                del state.fields["trickroom"]
            else:
                state.fields["trickroom"] = 5
        elif move.id == "reflect":
            side_conditions = state.side1_conditions if attacker.side == 1 else state.side2_conditions
            side_conditions["reflect"] = 5
        elif move.id == "lightscreen":
            side_conditions = state.side1_conditions if attacker.side == 1 else state.side2_conditions
            side_conditions["lightscreen"] = 5
        elif move.id == "auroraveil":
            if "snowscape" in state.weather or "hail" in state.weather:
                side_conditions = state.side1_conditions if attacker.side == 1 else state.side2_conditions
                side_conditions["auroraveil"] = 5
        elif move.id == "sunnyday":
            state.weather = {"sunnyday": 5}
        elif move.id == "raindance":
            state.weather = {"raindance": 5}
        elif move.id == "sandstorm":
            state.weather = {"sandstorm": 5}
        elif move.id in ("snowscape", "hail"):
            state.weather = {"snowscape": 5}
        elif move.id == "psychicterrain":
            state.fields["psychicterrain"] = 5
        elif move.id == "electricterrain":
            state.fields["electricterrain"] = 5
        elif move.id == "grassyterrain":
            state.fields["grassyterrain"] = 5
        elif move.id == "mistyterrain":
            state.fields["mistyterrain"] = 5

        # Status afflictions
        elif move.id in ("thunderwave", "glare") and target.status is None:
            if "electric" not in target.types and (move.id == "glare" or "ground" not in target.types) and target.ability not in ("limber", "purifyingsalt"):
                target.status = "par"
        elif move.id in ("willowisp",) and target.status is None:
            if "fire" not in target.types and target.ability not in ("waterveil", "thermalexchange", "purifyingsalt"):
                target.status = "brn"
        elif move.id in ("spore", "hypnosis", "sleeppowder", "darkvoid", "sing", "yawn") and target.status is None:
            if "grass" not in target.types and "electricterrain" not in state.fields and target.ability not in ("vitalspirit", "insomnia", "purifyingsalt"):
                target.status = "slp"
        elif move.id in ("toxic", "poisonpowder") and target.status is None:
            if not any(type_name in target.types for type_name in ("poison", "steel")) and target.ability not in ("immunity", "purifyingsalt"):
                target.status = "tox"

        # Direct Healing
        elif move.id in ("recover", "slackoff", "softboiled", "milkdrink", "healorder", "wish"):
            attacker.heal(attacker.max_hp * 0.50)
        elif move.id == "roost":
            attacker.heal(attacker.max_hp * 0.50)
            if "flying" in attacker.types:
                attacker.types = tuple(t for t in attacker.types if t != "flying")
                attacker.roost_active = True
        elif move.id in ("synthesis", "moonlight", "morningsun"):
            heal_rate = 0.66 if "sunnyday" in state.weather else (0.25 if any(w in state.weather for w in ("raindance", "sandstorm", "snowscape", "hail")) else 0.50)
            attacker.heal(attacker.max_hp * heal_rate)
        elif move.id == "strengthsap":
            if not is_ally:
                target_atk = target.calculate_effective_attack(is_special=False, weather=state.weather, fields=state.fields)
                target.apply_boost("atk", -1)
                attacker.heal(target_atk)

        # Team Mitigation & Redirection
        elif move.id == "wideguard":
            state.wide_guard[attacker.side] = True
        elif move.id == "quickguard":
            state.quick_guard[attacker.side] = True
        elif move.id == "craftyshield":
            state.crafty_shield[attacker.side] = True
        elif move.id in ("followme", "ragepowder"):
            state.redirect_target[attacker.side] = (attacker.slot, move.id)
        elif move.id == "allyswitch":
            active_list = state.side1_active if attacker.side == 1 else state.side2_active
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

        # Disruption & Debuffs
        elif move.id == "taunt":
            if not is_ally:
                if target.item == "mentalherb":
                    target.item = ""
                elif target.ability != "aromaveil":
                    target.taunt_turns = 3
        elif move.id == "encore":
            if not is_ally and target.last_move_used and target.last_move_used in target.moves:
                if target.item == "mentalherb":
                    target.item = ""
                elif target.ability != "aromaveil":
                    target.encore_turns = 3
                    target.encore_move = target.last_move_used
        elif move.id == "disable":
            if not is_ally and target.last_move_used and target.last_move_used in target.moves:
                if target.item == "mentalherb":
                    target.item = ""
                elif target.ability != "aromaveil":
                    target.disabled_move = target.last_move_used
                    target.disable_turns = 4
        elif move.id == "torment":
            if not is_ally:
                if target.item == "mentalherb":
                    target.item = ""
                elif target.ability != "aromaveil":
                    target.torment = True
        elif move.id == "gravity":
            state.fields["gravity"] = 5
        elif move.id == "chillyreception":
            state.weather = {"snowscape": 5}
            bench = state.side1_bench if attacker.side == 1 else state.side2_bench
            for b_idx, b_mon in enumerate(bench):
                if b_mon and not b_mon.fainted:
                    _execute_switches(state, [SimulatedAction("switch", slot=attacker.slot, switch_idx=b_idx)], side=attacker.side)
                    break
        elif move.id == "curse":
            if "ghost" in attacker.types:
                attacker.take_damage(attacker.max_hp * 0.50)
                target.cursed = True
            else:
                attacker.apply_boost("spe", -1)
                attacker.apply_boost("atk", 1)
                attacker.apply_boost("def", 1)
        elif move.id == "sweetscent":
            target.apply_boost("evasion", -2)
        elif move.id == "tearfullook":
            target.apply_boost("atk", -1)
            target.apply_boost("spa", -1)
        elif move.id == "tickle":
            target.apply_boost("atk", -1)
            target.apply_boost("def", -1)
        elif move.id == "soak":
            target.types = ("water",)
        elif move.id == "magicpowder":
            target.types = ("psychic",)
        elif move.id == "forestscurse":
            if "grass" not in target.types:
                target.types = target.types + ("grass",)
        elif move.id == "trickortreat":
            if "ghost" not in target.types:
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
        elif move.id == "perishsong":
            for active_pkmn in state.side1_active + state.side2_active:
                if active_pkmn and not active_pkmn.fainted and active_pkmn.ability != "soundproof":
                    if active_pkmn.perish_turns == 0:
                        active_pkmn.perish_turns = 3
        elif move.id in ("meanlook", "block", "spiderweb"):
            if not is_ally and "ghost" not in target.types and target.item != "shedshell":
                target.is_trapped = True
        elif move.id == "imprison":
            attacker.imprison = True
        return

    # 3. Dynamic Base Power & Move Typing
    base_power = move.base_power
    effective_type = move.type

    if move.id in ("waterspout", "eruption", "dragonenergy"):
        base_power = max(1.0, math.floor(150.0 * attacker.current_hp_fraction))
    elif move.id == "lastrespects":
        fainted_allies = sum(
            1 for allied_pokemon in (state.all_side1_pokemon() if attacker.side == 1 else state.all_side2_pokemon()) if allied_pokemon.fainted
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
    elif move.id == "knockoff":
        if target.item not in ("none", "") and not target.item.endswith("ite"):
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
    elif move.id == "weatherball":
        if "sunnyday" in state.weather: base_power, effective_type = 100.0, "fire"
        elif "raindance" in state.weather: base_power, effective_type = 100.0, "water"
        elif "sandstorm" in state.weather: base_power, effective_type = 100.0, "rock"
        elif "snowscape" in state.weather or "hail" in state.weather: base_power, effective_type = 100.0, "ice"
    elif move.id == "terrainpulse":
        if "electricterrain" in state.fields and "flying" not in attacker.types: base_power, effective_type = 100.0, "electric"
        elif "grassyterrain" in state.fields and "flying" not in attacker.types: base_power, effective_type = 100.0, "grass"
        elif "psychicterrain" in state.fields and "flying" not in attacker.types: base_power, effective_type = 100.0, "psychic"
        elif "mistyterrain" in state.fields and "flying" not in attacker.types: base_power, effective_type = 100.0, "fairy"
    elif move.id == "expandingforce":
        if "psychicterrain" in state.fields and "flying" not in attacker.types:
            base_power = 120.0
    elif move.id == "risingvoltage":
        if "electricterrain" in state.fields and "flying" not in target.types:
            base_power = 140.0
    elif move.id == "psyblade":
        if "electricterrain" in state.fields:
            base_power = 120.0
    elif move.id == "hardpress":
        base_power = max(1.0, math.floor(100.0 * target.current_hp_fraction))
    elif move.id == "revelationdance":
        if attacker.types:
            effective_type = attacker.types[0]
    elif move.id == "aurawheel":
        effective_type = "electric"
    elif move.id == "ragingbull":
        if "fire" in attacker.species:
            effective_type = "fire"
        elif "water" in attacker.species:
            effective_type = "water"
        else:
            effective_type = "fighting"
    elif move.id == "ivyudgel":
        if "hearthflame" in attacker.species or attacker.item == "hearthflamemask":
            effective_type = "fire"
        elif "wellspring" in attacker.species or attacker.item == "wellspringmask":
            effective_type = "water"
        elif "cornerstone" in attacker.species or attacker.item == "cornerstonemask":
            effective_type = "rock"
        else:
            effective_type = "grass"

    if base_power <= 0:
        return

    # Double Shock & Burn Up: fail if user lacks corresponding type; strip type on use
    had_electric_stab = False
    had_fire_stab = False
    if move.id == "doubleshock":
        if "electric" not in attacker.types:
            return  # Move fails if user is not Electric type
        had_electric_stab = True
        attacker.types = tuple(t for t in attacker.types if t != "electric")
    elif move.id == "burnup":
        if "fire" not in attacker.types:
            return  # Move fails if user is not Fire type
        had_fire_stab = True
        attacker.types = tuple(t for t in attacker.types if t != "fire")

    # Normal-type conversion abilities & move-type altering abilities
    base_power_multiplier = 1.0
    if effective_type == "normal":
        if attacker.ability == "pixilate": effective_type, base_power_multiplier = "fairy", 1.2
        elif attacker.ability == "aerilate": effective_type, base_power_multiplier = "flying", 1.2
        elif attacker.ability == "refrigerate": effective_type, base_power_multiplier = "ice", 1.2
        elif attacker.ability == "galvanize": effective_type, base_power_multiplier = "electric", 1.2
    if attacker.ability == "normalize":
        effective_type, base_power_multiplier = "normal", 1.2
    elif attacker.ability == "liquidvoice" and move.id in _SOUND_MOVES:
        effective_type = "water"

    # Protean / Libero (changes user's type to move type before STAB/damage)
    if attacker.ability in ("protean", "libero") and not attacker.protean_activated:
        attacker.types = (effective_type,)
        attacker.protean_activated = True

    has_gravity = ("gravity" in state.fields)
    # Type effectiveness & Scrappy
    if move.id == "struggle":
        type_multiplier = 1.0  # Struggle deals typeless damage, hits Ghost-types neutrally
    else:
        type_multiplier = target.damage_multiplier(effective_type, has_gravity=has_gravity)
    if move.id == "freezedry" and "water" in target.types:
        type_multiplier *= 4.0  # Ice vs Water becomes 2.0x instead of 0.5x
    elif move.id == "thousandarrows" and "flying" in target.types and type_multiplier == 0.0:
        type_multiplier = 1.0
    if type_multiplier == 0.0:
        if attacker.ability == "scrappy" and effective_type in ("normal", "fighting"):
            type_multiplier = 1.0
        else:
            return

    # Defensive ability immunities
    if effective_type == "ground" and target.ability in ("levitate", "eelevate") and not has_gravity: return
    if effective_type == "ground" and target.ability == "eartheater": return
    if effective_type == "fire" and target.ability in ("flashfire", "wellbakedbody"): return
    if effective_type == "water" and target.ability in ("waterabsorb", "stormdrain", "dryskin"): return
    if effective_type == "electric" and target.ability in ("voltabsorb", "lightningrod", "motordrive"): return
    if effective_type == "grass" and target.ability == "sapsipper": return
    if is_ally and target.ability == "telepathy": return

    # STAB calculation
    same_type_attack_bonus = 1.0
    if effective_type in attacker.types or (move.id == "doubleshock" and had_electric_stab) or (move.id == "burnup" and had_fire_stab):
        same_type_attack_bonus = 2.0 if attacker.ability == "adaptability" else 1.5

    # Attack and Defense Stat Resolution
    is_special = (move.category == "special")
    if move.id == "bodypress":
        effective_attack_stat = attacker.calculate_effective_defense(False, state.weather, state.fields)
        effective_defense_stat = target.calculate_effective_defense(False, state.weather, state.fields)
    elif move.id == "foulplay":
        effective_attack_stat = target.calculate_effective_attack(False, state.weather, state.fields)
        effective_defense_stat = target.calculate_effective_defense(False, state.weather, state.fields)
    elif move.id in ("psyshock", "psystrike", "secretsword"):
        effective_attack_stat = attacker.calculate_effective_attack(True, state.weather, state.fields)
        effective_defense_stat = target.calculate_effective_defense(False, state.weather, state.fields)
    else:
        effective_attack_stat = attacker.calculate_effective_attack(is_special, state.weather, state.fields)
        effective_defense_stat = target.calculate_effective_defense(is_special, state.weather, state.fields)

    # Facade ignores burn penalty
    if move.id == "facade" and attacker.status == "brn" and attacker.ability != "guts":
        effective_attack_stat *= 2.0

    total_damage_modifier = base_power_multiplier * same_type_attack_bonus * type_multiplier

    # Super-effective moves bonus (Collision Course, Electro Drift)
    if move.id in ("collisioncourse", "electrodrift") and type_multiplier > 1.0:
        total_damage_modifier *= 1.33

    if is_spread:
        total_damage_modifier *= 0.75

    # Weather multipliers
    if effective_type == "fire":
        if "sunnyday" in state.weather or "desolateland" in state.weather: total_damage_modifier *= 1.5
        elif "raindance" in state.weather: total_damage_modifier *= 0.5
    elif effective_type == "water":
        if move.id == "hydrosteam" and ("sunnyday" in state.weather or "desolateland" in state.weather):
            total_damage_modifier *= 1.5
        elif "raindance" in state.weather or "primordialsea" in state.weather: total_damage_modifier *= 1.5
        elif "sunnyday" in state.weather: total_damage_modifier *= 0.5

    # Terrain multipliers
    if "psychicterrain" in state.fields and effective_type == "psychic" and "flying" not in attacker.types:
        total_damage_modifier *= 1.3
    elif "electricterrain" in state.fields and effective_type == "electric" and "flying" not in attacker.types:
        total_damage_modifier *= 1.3
    elif "grassyterrain" in state.fields and effective_type == "grass" and "flying" not in attacker.types:
        total_damage_modifier *= 1.3
    elif "grassyterrain" in state.fields and move.id in ("earthquake", "bulldoze", "magnitude"):
        total_damage_modifier *= 0.5
    elif "mistyterrain" in state.fields and effective_type == "dragon" and "flying" not in target.types:
        total_damage_modifier *= 0.5

    # Screens reduction
    target_side_conditions = state.side2_conditions if target.side == 2 else state.side1_conditions
    screen_factor = 0.66
    if not is_special and "reflect" in target_side_conditions and move.id not in ("brickbreak", "psychicfangs"):
        total_damage_modifier *= screen_factor
    elif is_special and "lightscreen" in target_side_conditions and move.id not in ("brickbreak", "psychicfangs"):
        total_damage_modifier *= screen_factor
    if "auroraveil" in target_side_conditions and move.id not in ("brickbreak", "psychicfangs"):
        total_damage_modifier *= screen_factor

    # Helping Hand multiplier
    if attacker.helping_hand:
        total_damage_modifier *= 1.5

    # Standard Gen 9 Damage Formula (Level 50)
    calculated_damage = (
        ((22.0 * base_power * (effective_attack_stat / max(1.0, effective_defense_stat))) / 50.0) + 2.0
    ) * total_damage_modifier
    actual_damage_dealt = target.take_damage(calculated_damage)

    # Destiny Bond trigger: if target faints from foe's attack, attacker faints too
    if target.destiny_bond and target.fainted and not is_ally and not attacker.fainted:
        attacker.take_damage(attacker.current_hp)

    # Ally Ability & Item Procs (e.g. Weakness Policy, Justified)
    if is_ally:
        if target.item == "weaknesspolicy" and type_multiplier > 1.0:
            target.apply_boost("atk", 2)
            target.apply_boost("spa", 2)
            target.item = ""
        elif target.ability == "justified" and effective_type == "dark":
            target.apply_boost("atk", 1)

    # Rage Fist hit counter tracking
    state.hit_counts[target.species] = state.hit_counts.get(target.species, 0) + 1

    # Color Change ability on target
    if target.ability == "colorchange" and not target.fainted and effective_type:
        target.types = (effective_type,)

    # Trapping binding moves (Fire Spin, Whirlpool, Sand Tomb, Magma Storm, Infestation, Snap Trap, Anchor Shot, Jaw Lock, Thousand Waves)
    if move.id in ("firespin", "whirlpool", "sandtomb", "magmastorm", "infestation", "snaptrap", "anchorshot", "jawlock", "thousandwaves"):
        if "ghost" not in target.types and target.item != "shedshell" and not target.fainted:
            target.is_trapped = True

    # Struggle recoil (25% of max HP)
    if move.id == "struggle":
        attacker.take_damage(attacker.max_hp * 0.25)

    # Recoil & Life Orb
    if attacker.item == "lifeorb":
        attacker.take_damage(attacker.max_hp * 0.10)
    if getattr(move, "recoil", 0.0) > 0.0 and attacker.ability != "rockhead":
        attacker.take_damage(actual_damage_dealt * 0.33)

    # Drain moves
    if move.id in ("drainpunch", "gigadrain", "hornleech", "bitterblade", "paraboliccharge"):
        attacker.heal(actual_damage_dealt * 0.50)

    # Secondary stat drops on target
    if move.id in ("snarl", "skittersmack", "mysticalfire", "strugglebug", "spiritbreak"):
        target.apply_boost("spa", -1)
    elif move.id in ("icywind", "electroweb", "mudshot", "rocktomb", "bulldoze", "lowsweep", "pounce"):
        target.apply_boost("spe", -1)
    elif move.id in ("lunge", "tropkick", "chillingwater"):
        target.apply_boost("atk", -1)
    elif move.id in ("firelash", "gravapple", "thunderouskick"):
        target.apply_boost("def", -1)
    elif move.id == "applesacid":
        target.apply_boost("spd", -1)

    # Secondary stat boosts on attacker
    if move.id == "torchsong":
        attacker.apply_boost("spa", 1)
    elif move.id == "aquastep":
        attacker.apply_boost("spe", 1)
    elif move.id == "poweruppunch":
        attacker.apply_boost("atk", 1)
    elif move.id in ("trailblaze", "flamecharge"):
        attacker.apply_boost("spe", 1)
    elif move.id in ("fierydance", "chargebeam"):
        attacker.apply_boost("spa", 1)
    elif move.id == "diamondstorm":
        attacker.apply_boost("def", 2)

    # Self stat drops from powerful attacks
    if move.id in ("closecombat", "headlongrush", "armorcannon"):
        attacker.apply_boost("def", -1)
        attacker.apply_boost("spd", -1)
    elif move.id in ("dracometeor", "overheat", "leafstorm", "fleurcannon", "psychoboost"):
        attacker.apply_boost("spa", -2)
    elif move.id == "makeitrain":
        attacker.apply_boost("spa", -1)
    elif move.id == "superpower":
        attacker.apply_boost("atk", -1)
        attacker.apply_boost("def", -1)
    elif move.id == "vcreate":
        attacker.apply_boost("def", -1)
        attacker.apply_boost("spd", -1)
        attacker.apply_boost("spe", -1)
    elif move.id in ("hammerarm", "icehammer"):
        attacker.apply_boost("spe", -1)
    elif move.id == "spinout":
        attacker.apply_boost("spe", -2)
    elif move.id == "clangingscales":
        attacker.apply_boost("def", -1)

def _execute_end_of_turn(state: SimulatedBattleState) -> None:
    all_active_pokemon = [pokemon for pokemon in state.side1_active + state.side2_active if pokemon and not pokemon.fainted]

    for pokemon in all_active_pokemon:
        if pokemon.item == "leftovers":
            pokemon.heal(pokemon.max_hp / 16.0)

        if "grassyterrain" in state.fields and "flying" not in pokemon.types and pokemon.ability != "levitate":
            pokemon.heal(pokemon.max_hp / 16.0)

        if pokemon.status == "brn":
            pokemon.take_damage(pokemon.max_hp / 16.0)
        elif pokemon.status == "psn":
            pokemon.take_damage(pokemon.max_hp / 8.0)
        elif pokemon.status == "tox":
            pokemon.toxic_turns += 1
            pokemon.take_damage(pokemon.max_hp * (pokemon.toxic_turns / 16.0))

        if "sandstorm" in state.weather:
            if not any(type_name in pokemon.types for type_name in ("rock", "ground", "steel")):
                pokemon.take_damage(pokemon.max_hp / 16.0)

        # Volatile status decrements & cursed damage
        if pokemon.taunt_turns > 0:
            pokemon.taunt_turns -= 1
        if pokemon.encore_turns > 0:
            pokemon.encore_turns -= 1
            if pokemon.encore_turns == 0:
                pokemon.encore_move = None
        if pokemon.disable_turns > 0:
            pokemon.disable_turns -= 1
            if pokemon.disable_turns == 0:
                pokemon.disabled_move = None
        if pokemon.cursed:
            pokemon.take_damage(pokemon.max_hp * 0.25)
        if pokemon.perish_turns > 0:
            pokemon.perish_turns -= 1
            if pokemon.perish_turns == 0:
                pokemon.take_damage(pokemon.max_hp)

        pokemon.first_turn = False
        pokemon.flinch = False
        pokemon.helping_hand = False
        pokemon.destiny_bond = False
        if pokemon.roost_active:
            pokemon.types = pokemon.base_types
            pokemon.roost_active = False

    # Decrement Field and Condition Timers
    for condition_name in list(state.weather.keys()):
        state.weather[condition_name] -= 1
        if state.weather[condition_name] <= 0:
            del state.weather[condition_name]

    for condition_name in list(state.fields.keys()):
        state.fields[condition_name] -= 1
        if state.fields[condition_name] <= 0:
            del state.fields[condition_name]

    for condition_name in list(state.side1_conditions.keys()):
        state.side1_conditions[condition_name] -= 1
        if state.side1_conditions[condition_name] <= 0:
            del state.side1_conditions[condition_name]

    for condition_name in list(state.side2_conditions.keys()):
        state.side2_conditions[condition_name] -= 1
        if state.side2_conditions[condition_name] <= 0:
            del state.side2_conditions[condition_name]

    _replace_fainted_slots(state, side=1)
    _replace_fainted_slots(state, side=2)

def _replace_fainted_slots(state: SimulatedBattleState, side: int) -> None:
    active_pokemon_list = state.side1_active if side == 1 else state.side2_active
    bench_pokemon_list = state.side1_bench if side == 1 else state.side2_bench

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
            else:
                active_pokemon_list[slot_index] = pokemon
