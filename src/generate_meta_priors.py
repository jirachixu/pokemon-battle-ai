import os
import glob
import re
import json
from collections import defaultdict

TEAMS_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "teams")
OUTPUT_FILE_PATH = os.path.join(os.path.dirname(__file__), "meta_priors.py")


def to_alphanumeric(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


def clean_species_name(species_name_raw: str) -> str:
    """Extracts base species identifier, removing gender flags, nicknames, and special quotes."""
    species_name_stripped = re.sub(r"\s*\([MF]\)$", "", species_name_raw.strip())
    species_match = re.search(r"\(([^)]+)\)$", species_name_stripped)
    raw_species = species_match.group(1).strip() if species_match else species_name_stripped.strip()
    return to_alphanumeric(raw_species)


def generate_champions_priors():
    team_files = sorted(glob.glob(os.path.join(TEAMS_DIRECTORY, "*.txt")))
    print(f"Processing {len(team_files)} Champions Regulation M-C team files...")

    counts_moves = defaultdict(lambda: defaultdict(int))
    counts_items = defaultdict(lambda: defaultdict(int))
    counts_abilities = defaultdict(lambda: defaultdict(int))
    counts_spreads = defaultdict(list)
    species_appearances = defaultdict(int)

    for file_path in team_files:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file_handle:
            team_content = file_handle.read().strip()
        pokemon_blocks = [block.strip() for block in team_content.split("\n\n") if block.strip()]
        for pokemon_block in pokemon_blocks:
            lines = pokemon_block.split("\n")
            first_line = lines[0].strip()
            item_name = "none"
            if "@" in first_line:
                name_portion, item_portion = first_line.split("@", 1)
                item_name = to_alphanumeric(item_portion)
            else:
                name_portion = first_line

            species_identifier = clean_species_name(name_portion)
            species_appearances[species_identifier] += 1
            counts_items[species_identifier][item_name] += 1

            pokemon_nature = "hardy"
            effort_values = {}
            for line in lines[1:]:
                stripped_line = line.strip()
                if stripped_line.startswith("Ability:"):
                    ability_name = to_alphanumeric(stripped_line.split(":", 1)[1])
                    counts_abilities[species_identifier][ability_name] += 1
                elif stripped_line.startswith("- "):
                    move_name = to_alphanumeric(stripped_line[2:])
                    counts_moves[species_identifier][move_name] += 1
                elif stripped_line.endswith(" Nature"):
                    pokemon_nature = stripped_line.replace(" Nature", "").strip().lower()
                elif stripped_line.startswith("EVs:"):
                    effort_values_string = stripped_line[4:].strip()
                    for effort_value_part in effort_values_string.split("/"):
                        stat_split = effort_value_part.strip().split()
                        if len(stat_split) == 2:
                            try:
                                stat_name = stat_split[1].lower()
                                effort_value_amount = int(stat_split[0])
                                effort_values[stat_name] = effort_value_amount
                            except ValueError:
                                pass
            counts_spreads[species_identifier].append((pokemon_nature, effort_values))

    move_priors = {}
    for species_identifier, moves_dict in sorted(counts_moves.items()):
        total_pokemon_count = species_appearances[species_identifier]
        move_priors[species_identifier] = {
            move_name: round(move_count / total_pokemon_count, 4)
            for move_name, move_count in sorted(moves_dict.items(), key=lambda item_pair: item_pair[1], reverse=True)
        }

    ability_priors = {}
    for species_identifier, abilities_dict in sorted(counts_abilities.items()):
        total_abilities_count = sum(abilities_dict.values())
        ability_priors[species_identifier] = {
            ability_name: round(ability_count / total_abilities_count, 4)
            for ability_name, ability_count in sorted(abilities_dict.items(), key=lambda item_pair: item_pair[1], reverse=True)
        }

    item_priors = {}
    for species_identifier, items_dict in sorted(counts_items.items()):
        total_items_count = sum(items_dict.values())
        item_priors[species_identifier] = {
            item_name: round(item_count / total_items_count, 4)
            for item_name, item_count in sorted(items_dict.items(), key=lambda item_pair: item_pair[1], reverse=True)
        }

    common_spreads = {}
    for species_identifier, spreads_list in sorted(counts_spreads.items()):
        nature_counts = defaultdict(int)
        for spread_nature, _ in spreads_list:
            nature_counts[spread_nature] += 1
        most_common_nature = sorted(nature_counts.items(), key=lambda item_pair: item_pair[1], reverse=True)[0][0]
        effort_value_sums = defaultdict(int)
        effort_value_counts = defaultdict(int)
        for spread_nature, spread_effort_values in spreads_list:
            if spread_nature == most_common_nature:
                for stat_name, stat_value in spread_effort_values.items():
                    effort_value_sums[stat_name] += stat_value
                    effort_value_counts[stat_name] += 1
        average_effort_values = {
            stat_name: int(round(effort_value_sums[stat_name] / max(1, effort_value_counts[stat_name])))
            for stat_name in effort_value_sums
        }
        common_spreads[species_identifier] = {
            "nature": most_common_nature,
            "evs": average_effort_values,
        }

    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as output_file_handle:
        output_file_handle.write("# Automatically extracted empirical priors for Pokemon Champions Regulation M-C VGC\n")
        output_file_handle.write("import re\n\n")

        output_file_handle.write("CHAMPIONS_MOVE_PRIORS = {\n")
        for species_identifier, moves_dict in move_priors.items():
            output_file_handle.write(f'    "{species_identifier}": {{\n')
            for move_name, probability in moves_dict.items():
                output_file_handle.write(f'        "{move_name}": {probability},\n')
            output_file_handle.write("    },\n")
        output_file_handle.write("}\n\n")

        output_file_handle.write("CHAMPIONS_ABILITY_PRIORS = {\n")
        for species_identifier, abilities_dict in ability_priors.items():
            output_file_handle.write(f'    "{species_identifier}": {{\n')
            for ability_name, probability in abilities_dict.items():
                output_file_handle.write(f'        "{ability_name}": {probability},\n')
            output_file_handle.write("    },\n")
        output_file_handle.write("}\n\n")

        output_file_handle.write("CHAMPIONS_ITEM_PRIORS = {\n")
        for species_identifier, items_dict in item_priors.items():
            output_file_handle.write(f'    "{species_identifier}": {{\n')
            for item_name, probability in items_dict.items():
                output_file_handle.write(f'        "{item_name}": {probability},\n')
            output_file_handle.write("    },\n")
        output_file_handle.write("}\n\n")

        output_file_handle.write("CHAMPIONS_COMMON_SPREADS = {\n")
        for species_identifier, spread_dictionary in common_spreads.items():
            output_file_handle.write(f'    "{species_identifier}": {json.dumps(spread_dictionary)},\n')
        output_file_handle.write("}\n\n")

        output_file_handle.write("""
def get_champions_move_priors(species: str) -> dict[str, float]:
    \"\"\"Return move priors for a given species in Champions Regulation M-C.\"\"\"
    species_clean = re.sub(r"[^a-zA-Z0-9]", "", species).lower()
    return CHAMPIONS_MOVE_PRIORS.get(species_clean, {})

def get_champions_ability_priors(species: str) -> dict[str, float]:
    \"\"\"Return ability priors for a given species in Champions Regulation M-C.\"\"\"
    species_clean = re.sub(r"[^a-zA-Z0-9]", "", species).lower()
    return CHAMPIONS_ABILITY_PRIORS.get(species_clean, {})

def get_champions_item_priors(species: str) -> dict[str, float]:
    \"\"\"Return item priors for a given species in Champions Regulation M-C.\"\"\"
    species_clean = re.sub(r"[^a-zA-Z0-9]", "", species).lower()
    return CHAMPIONS_ITEM_PRIORS.get(species_clean, {})

def get_champions_common_spread(species: str) -> dict:
    \"\"\"Return typical nature and EV spread for a given species in Champions Regulation M-C.\"\"\"
    species_clean = re.sub(r"[^a-zA-Z0-9]", "", species).lower()
    return CHAMPIONS_COMMON_SPREADS.get(species_clean, {"nature": "hardy", "evs": {}})
""")

    print(f"Successfully generated {OUTPUT_FILE_PATH} for {len(move_priors)} species.")


if __name__ == "__main__":
    generate_champions_priors()
