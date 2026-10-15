"""
Module for loading, classifying, and sampling competitive Pokémon VGC teams
for multi-team neural network and lookahead training.
"""
import glob
import os
import random
import re
from dataclasses import dataclass, field
from typing import Optional

ALL_ARCHETYPES = [
    "trick_room",
    "sun",
    "rain",
    "sand",
    "snow",
    "psyspam",
    "perish_trap",
    "other",
]

@dataclass
class TeamInfo:
    """Represents a competitive Pokémon team and its detected archetypes."""
    team_id: str
    file_path: str
    paste: str
    pokemon_names: list[str] = field(default_factory=list)
    archetypes: set[str] = field(default_factory=set)

    def has_archetype(self, archetype: str) -> bool:
        """Checks if this team contains the specified archetype."""
        return archetype in self.archetypes

    def __repr__(self) -> str:
        arch_str = ", ".join(sorted(self.archetypes))
        mons_str = ", ".join(self.pokemon_names[:4])
        if len(self.pokemon_names) > 4:
            mons_str += f" (+{len(self.pokemon_names) - 4} more)"
        return f"<TeamInfo {self.team_id} [{arch_str}]: {mons_str}>"

def extract_pokemon_names(paste: str) -> list[str]:
    """Extracts species names from a Pokémon Showdown export paste."""
    names: list[str] = []
    # Each Pokémon block is typically separated by blank lines
    blocks = re.split(r"\n\s*\n", paste.strip())
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        first_line = lines[0]
        # Ignore lines that are comments or metadata
        if first_line.startswith("===") or first_line.startswith("#"):
            continue
        # Strip item after '@'
        if "@" in first_line:
            first_line = first_line.split("@")[0].strip()
        # Strip gender markers (M) or (F)
        first_line = re.sub(r"\s*\([MF]\)", "", first_line, flags=re.IGNORECASE).strip()
        # Check for nickname format: Nickname (Species)
        match = re.search(r"\((.+?)\)", first_line)
        if match:
            species = match.group(1).strip()
        else:
            species = first_line
        if species and not species.lower().startswith("ability"):
            names.append(species)
    return names

def classify_paste(text: str) -> set[str]:
    """
    Classifies a team paste into archetypes based on abilities, items, moves, and key species.
    Teams can belong to multiple categories (e.g. dual-mode Sun + Trick Room).
    """
    lower = text.lower()
    archetypes: set[str] = set()

    # Perish Trap: Perish Song + Trapper (Gothitelle, Mega Gengar, or Shadow Tag ability)
    has_perish_song = bool(re.search(r"-\s*perish song", lower))
    has_trapper = bool(re.search(r"(ability:\s*shadow tag|gothitelle|gengarite|gengar-mega)", lower))
    if has_perish_song and has_trapper:
        archetypes.add("perish_trap")

    # Psyspam: Psychic Surge ability, or Expanding Force with Psychic Terrain enabler
    has_psychic_surge = "ability: psychic surge" in lower
    has_expanding_force = bool(re.search(r"-\s*expanding force", lower))
    if has_psychic_surge or (
        has_expanding_force and ("indeedee" in lower or "armarouge" in lower or "hatterene" in lower or "lele" in lower)
    ):
        archetypes.add("psyspam")

    # Sun: Drought, Orichalcum Pulse, Sunny Day, Chlorophyll, Mega Charizard Y, Torkoal, Ninetales
    has_sun_setter = bool(
        re.search(
            r"(ability:\s*(drought|orichalcum pulse)|-\s*sunny day|charizardite y|charizard-mega-y|torkoal|ninetales\b)",
            lower,
        )
    )
    has_sun_abuser = bool(re.search(r"ability:\s*chlorophyll", lower))
    if has_sun_setter or has_sun_abuser:
        archetypes.add("sun")

    # Rain: Drizzle, Rain Dance, Swift Swim, Mega Swampert, Pelipper, Politoed, Kyogre
    has_rain_setter = bool(re.search(r"(ability:\s*drizzle|-\s*rain dance|pelipper|politoed|kyogre)", lower))
    has_rain_abuser = bool(re.search(r"(ability:\s*swift swim|swampertite|swampert-mega)", lower))
    if has_rain_setter or has_rain_abuser:
        archetypes.add("rain")

    # Sand: Sand Stream, Sandstorm, Tyranitar, Hippowdon, Sand Rush, Sand Force, Excadrill
    has_sand_setter = bool(re.search(r"(ability:\s*sand stream|-\s*sandstorm|tyranitar|hippowdon|gigalith)", lower))
    has_sand_abuser = bool(re.search(r"ability:\s*(sand rush|sand force|sand veil)", lower))
    if has_sand_setter or has_sand_abuser:
        archetypes.add("sand")

    # Snow: Snow Warning, Snowscape, Chilly Reception, Aurora Veil, Slush Rush, Alolan Ninetales, Abomasnow
    has_snow_setter = bool(
        re.search(
            r"(ability:\s*snow warning|-\s*snowscape|-\s*chilly reception|ninetales-alola|abomasnow)",
            lower,
        )
    )
    has_snow_abuser = bool(re.search(r"(ability:\s*slush rush|-\s*aurora veil)", lower))
    if has_snow_setter or has_snow_abuser:
        archetypes.add("snow")

    # Trick Room: Has move Trick Room
    has_trick_room = bool(re.search(r"-\s*trick room", lower))
    if has_trick_room:
        archetypes.add("trick_room")

    # If no specialized archetype matched, mark as 'other' (standard balance / goodstuffs)
    if not archetypes:
        archetypes.add("other")

    return archetypes

class TeamPool:
    """
    Manages a pool of competitive Pokémon teams categorized by metagame archetypes.
    Supports uniform and stratified archetype sampling for RL and lookahead search training.
    """
    def __init__(self, teams_directory: str = "teams"):
        self.teams_directory = teams_directory
        self.teams: list[TeamInfo] = []
        self.by_id: dict[str, TeamInfo] = {}
        self.by_archetype: dict[str, list[TeamInfo]] = {arch: [] for arch in ALL_ARCHETYPES}
        self.load_directory(teams_directory)

    def load_directory(self, directory_path: str) -> None:
        """Loads and classifies all .txt team files from the specified directory."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Teams directory not found: {directory_path}")

        pattern = os.path.join(directory_path, "*.txt")
        file_paths = sorted(glob.glob(pattern))

        self.teams.clear()
        self.by_id.clear()
        for arch in self.by_archetype:
            self.by_archetype[arch].clear()

        for path in file_paths:
            team_id = os.path.splitext(os.path.basename(path))[0]
            with open(path, "r", encoding="utf-8", errors="replace") as file_handle:
                paste = file_handle.read().strip()
            # Sanitize mojibake and curly quotes so poke-env and showdown never fail on them
            paste = paste.replace("â€™", "'").replace("â", "'").replace("’", "'").replace("‘", "'")

            if not paste:
                continue

            pokemon_names = extract_pokemon_names(paste)
            archetypes = classify_paste(paste)

            team_info = TeamInfo(
                team_id=team_id,
                file_path=path,
                paste=paste,
                pokemon_names=pokemon_names,
                archetypes=archetypes,
            )

            self.teams.append(team_info)
            self.by_id[team_id] = team_info
            for arch in archetypes:
                if arch in self.by_archetype:
                    self.by_archetype[arch].append(team_info)

    def sample(self, archetype: Optional[str] = None) -> TeamInfo:
        """
        Samples a team uniformly at random from the entire pool, or from a specific archetype.
        """
        if not self.teams:
            raise ValueError("Team pool is empty.")

        if archetype is not None:
            if archetype not in self.by_archetype or not self.by_archetype[archetype]:
                raise ValueError(f"No teams available for archetype '{archetype}'")
            return random.choice(self.by_archetype[archetype])

        return random.choice(self.teams)

    def sample_stratified(self, weights: Optional[dict[str, float]] = None) -> TeamInfo:
        """
        Samples a team via stratified sampling: first selects an archetype (with optional weights),
        then samples uniformly within that archetype. This guarantees balanced exposure across
        both common and less common archetypes (e.g. Rain vs Sand).
        """
        active_archetypes = [arch for arch, pool in self.by_archetype.items() if pool]
        if not active_archetypes:
            raise ValueError("No active archetypes in team pool.")

        if weights:
            active_weights = [weights.get(arch, 1.0) for arch in active_archetypes]
            chosen_archetype = random.choices(active_archetypes, weights=active_weights, k=1)[0]
        else:
            chosen_archetype = random.choice(active_archetypes)

        return random.choice(self.by_archetype[chosen_archetype])

    def sample_pair(self, stratified: bool = True) -> tuple[TeamInfo, TeamInfo]:
        """
        Samples a pair of teams: (agent_team, opponent_team).
        Ensures they are distinct teams whenever pool size >= 2.
        """
        if len(self.teams) < 2:
            single_team = self.sample()
            return single_team, single_team

        sampler = self.sample_stratified if stratified else self.sample
        agent_team = sampler()
        opponent_team = sampler()
        
        return agent_team, opponent_team

    def summary(self) -> dict[str, int]:
        """Returns the number of teams belonging to each archetype."""
        return {arch: len(self.by_archetype[arch]) for arch in ALL_ARCHETYPES}

    def print_summary(self) -> None:
        """Prints a human-readable summary of the team pool."""
        print(f"=== TeamPool Summary: {len(self.teams)} Total Teams ===")
        for arch in ALL_ARCHETYPES:
            count = len(self.by_archetype[arch])
            pct = (count / len(self.teams) * 100) if self.teams else 0.0
            print(f"  {arch:14s}: {count:3d} teams ({pct:4.1f}%)")
        print("===============================================")
