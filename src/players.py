from poke_env.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.player.battle_order import BattleOrder
from typing import Any, Optional
import uuid
import queue
import poke_env.battle as pkmn_b
import asyncio
import torch
from itertools import combinations

from src.search import LookaheadSearch
from src.evaluator import BattleStateEvaluator, CompetitiveHeuristicEvaluator
from src.state_encoder import TeamPreviewEncoder
from src.models import TeamPreviewModel

def _select_teampreview(
    team_preview_encoder: TeamPreviewEncoder,
    team_preview_model: TeamPreviewModel,
    battle: pkmn_b.AbstractBattle,
    should_track_grad: bool = False,
) -> tuple[str, dict]:
    """
    Uses the TeamPreviewEncoder and the TeamPreviewModel to decide which Pokémon to send out in the first \
    turn of a double battle.
    Args:
        battle (pkmn_b.AbstractBattle): The battle currently in team preview.
    """
    assert isinstance(battle, pkmn_b.DoubleBattle), "Expected a DoubleBattle instance"
    observation = team_preview_encoder.encode(battle)
    device = next(team_preview_model.parameters()).device
    observation = {key: value.to(device) for key, value in observation.items()}
    
    # Generate all possible combinations of 4 Pokémon from the 6 brought to battle, 
    # and then all combinations of 2 leads from those 4. The remaining 2 are the backline Pokemon. 
    team_preview_order = []
    for brought_mons in combinations(range(6), 4):
        for leads in combinations(brought_mons, 2):
            back = [mon for mon in brought_mons if mon not in leads]
            team_preview_order.append((leads[0], leads[1], back[0], back[1]))
    
    with torch.set_grad_enabled(team_preview_model.training and should_track_grad):
        logits, value = team_preview_model(observation)
        distribution = torch.distributions.Categorical(logits=logits)
        if not team_preview_model.training:
            action_index = torch.argmax(logits)
        else:
            action_index = distribution.sample()
        # Policy gradient loss uses log probabilities
        log_probability = distribution.log_prob(action_index)
        
        # Used in training to calculate loss and update weights
        preview_decision = {
            "log_probability": log_probability,
            "value": value,
            "entropy": distribution.entropy(),
        }
    
    lead_0, lead_1, back_0, back_1 = team_preview_order[int(action_index.item())]
    # poke-env expects _selected_in_teampreview to be set for selected Pokemon
    team_mons = list(battle.team.values())
    for idx in (lead_0, lead_1, back_0, back_1):
        if idx < len(team_mons):
            team_mons[idx]._selected_in_teampreview = True
    
    return f"/team {lead_0 + 1}{lead_1 + 1}{back_0 + 1}{back_1 + 1}", preview_decision

class LookaheadPlayer(Player):
    """
    Autonomous poke-env battle agent for Doubles / VGC powered by 2-3 turn simultaneous
    lookahead search with Counterfactual Regret Matching (CFR) Nash equilibrium.
    """

    searcher: LookaheadSearch

    def __init__(
        self,
        depth: int = 2,
        max_candidates: int = 10,
        evaluator: Optional[BattleStateEvaluator] = None,
        team_preview_encoder: Optional[TeamPreviewEncoder] = None,
        team_preview_model: Optional[TeamPreviewModel] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("accept_open_team_sheet", True)
        if "account_configuration" not in kwargs:
            kwargs["account_configuration"] = AccountConfiguration(f"Lookahead_{uuid.uuid4().hex[:6]}", None)
        kwargs.pop("method", None)
        super().__init__(*args, **kwargs)
        self.searcher = LookaheadSearch(
            evaluator=evaluator or CompetitiveHeuristicEvaluator(),
            depth=depth,
            max_candidates=max_candidates,
        )
        self.team_preview_encoder = team_preview_encoder or (TeamPreviewEncoder() if team_preview_model is not None else None)
        self.team_preview_model = team_preview_model

    def choose_move(self, battle: pkmn_b.AbstractBattle) -> BattleOrder:
        assert isinstance(battle, pkmn_b.DoubleBattle), "Expected a DoubleBattle instance"
        if self._trying_again.is_set():
            self._trying_again.clear()
            return self.choose_default_move()
        try:
            return self.searcher.select_order(battle, player=self)
        except Exception:
            return self.choose_default_move()
        
    def teampreview(self, battle: pkmn_b.AbstractBattle) -> str:
        if self.team_preview_model is not None and self.team_preview_encoder is not None:
            return _select_teampreview(
                self.team_preview_encoder,
                self.team_preview_model,
                battle,
                False,
            )[0]
        return str(super().teampreview(battle))

class RLPlayer(Player):
    """
    Custom Player class for reinforcement learning.
    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("accept_open_team_sheet", True)
        if "account_configuration" not in kwargs:
            kwargs["account_configuration"] = AccountConfiguration(f"RLPlayer_{uuid.uuid4().hex[:6]}", None)
        super().__init__(*args, **kwargs)
        self.battle_queue = queue.Queue()
        self.current_order_future: asyncio.Future | None = None
        self.team_preview_encoder: TeamPreviewEncoder = TeamPreviewEncoder()
        self.team_preview_model: TeamPreviewModel = TeamPreviewModel()
        self.last_preview_decision = {}
        
    async def choose_move(self, battle: pkmn_b.AbstractBattle) -> BattleOrder:
        """
        Puts the current battle state into the queue and waits for the RL agent to provide an order. Because the battle is \
        put into the queue before the order is retrieved, this ensures that the RL agent always has the most up-to-date \
        battle state when making a decision.
        Args:
            battle (pkmn_b.AbstractBattle): The current battle state.
        Returns:
            DoubleBattleOrder: The Order object chosen by the RL agent.
        """
        assert isinstance(battle, pkmn_b.DoubleBattle), "Expected a DoubleBattle instance"
        if self._trying_again.is_set():
            self._trying_again.clear()
            return self.choose_default_move()
        self.current_order_future = self.ps_client.loop.create_future()
        self.battle_queue.put(battle)
        return await self.current_order_future
    
    def receive_order(self, order: BattleOrder) -> None:
        """
        Receives the order from the RL agent and sets the result of the current_order_future. This unblocks the \
        choose_move coroutine, allowing it to return the order to Showdown.
        Args:
            order (BattleOrder): The Order object chosen by the RL agent.
        """
        if self.current_order_future and not self.current_order_future.done():
            self.ps_client.loop.call_soon_threadsafe(self.current_order_future.set_result, order)
    
    def _battle_finished_callback(self, battle: pkmn_b.AbstractBattle) -> None:
        """
        Callback function called when a battle is finished. Puts the finished battle into the queue. This ensures that \
        the thread isn't blocked waiting for the RL agent to provide an order after the battle has ended.
        Args:
            battle (pkmn_b.AbstractBattle): The finished battle.
        """
        assert isinstance(battle, pkmn_b.DoubleBattle), "Expected a DoubleBattle instance"
        super()._battle_finished_callback(battle)
        self.battle_queue.put(battle)
        
    def teampreview(self, battle: pkmn_b.AbstractBattle) -> str:
        """
        Selects Pokemon for the battle in team preview.
        Args:
            battle (pkmn_b.AbstractBattle): The current battle.
        Returns:
            str: The selected Pokemon by index in the format expected by Showdown \
            (e.g., "/team 1234" for selecting Pokemon 1, 2, 3, and 4).
        """
        assert isinstance(battle, pkmn_b.DoubleBattle), "Expected a DoubleBattle instance"
        selected_team, decision_info = _select_teampreview(
            self.team_preview_encoder,
            self.team_preview_model,
            battle,
            self.team_preview_model.training,
        )
        self.last_preview_decision = decision_info
        return selected_team
