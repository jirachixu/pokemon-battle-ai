# Patch poke-env's to_id_str to properly strip mojibake, accents, and non-ASCII characters
import unicodedata
import re
import poke_env.data.normalize as _pk_norm
import poke_env.battle.pokemon as _pk_mon
import poke_env.teambuilder.teambuilder as _pk_tb

def _robust_to_id_str(name: str) -> str:
    """Normalizes names to Showdown-compliant alphanumeric lowercase IDs."""
    if not name:
        return ""
    if hasattr(name, "name"):
        name = name.name
    s = str(name)
    s = s.replace("â€™", "").replace("â", "").replace("’", "").replace("'", "")
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()

_pk_norm.to_id_str = _robust_to_id_str
_pk_mon.to_id_str = _robust_to_id_str
_pk_tb.to_id_str = _robust_to_id_str

"""
Pokemon Battle AI core simulation and multi-turn lookahead search package.
"""

from src.simulation import (
    SimulatedBattleState,
    SimulatedPokemon,
    SimulatedMove,
    SimulatedAction,
    step_simulation,
    clean_id,
)
from src.evaluator import (
    CompetitiveHeuristicEvaluator,
    NeuralValueEvaluator,
    HybridEvaluator,
)
from src.search import (
    LookaheadSearch,
    SearchResult,
    solve_zero_sum_matrix_game,
)
from src.players import (
    LookaheadPlayer,
    RLPlayer,
)
from src.team_pool import (
    TeamPool,
    TeamInfo,
    ALL_ARCHETYPES,
)

__all__ = [
    "SimulatedBattleState",
    "SimulatedPokemon",
    "SimulatedMove",
    "SimulatedAction",
    "step_simulation",
    "clean_id",
    "CompetitiveHeuristicEvaluator",
    "NeuralValueEvaluator",
    "HybridEvaluator",
    "LookaheadSearch",
    "SearchResult",
    "LookaheadPlayer",
    "RLPlayer",
    "TeamPool",
    "TeamInfo",
    "ALL_ARCHETYPES",
    "solve_zero_sum_matrix_game",
]