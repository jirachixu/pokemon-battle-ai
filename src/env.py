import poke_env.battle as pkmn_b
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import DoubleBattleOrder, PassBattleOrder, ForfeitBattleOrder, BattleOrder

import gymnasium.spaces as spaces
import gymnasium as gym
import numpy as np
from typing import Any
import queue
import asyncio

from src.state_encoder import StateEncoder
from src import constants

# ----------------- Action and Order Helper Functions -----------------

def get_target_position(target_index: int, is_slot_a: bool) -> int:
    """
    Get the target position based on the target index and whether the active Pokémon is in slot A or B.
    Args:
        target_index (int): The index of the target (0, 1, or 2).
        is_slot_a (bool): True if the active Pokémon is in slot A, False if in slot B (left, right slot).
    Returns:
        int: The position of the target as defined by poke-env. 1 or 2 for left or right opponent respectively, \
        -1 or -2 for left or right teammate respectively.
    """
    return target_index + 1 if target_index < 2 else (-2 if is_slot_a else -1)

def get_slot_action_mask(battle: pkmn_b.DoubleBattle, slot_idx: int) -> list[bool]:
    """
    Gets the action mask for a specific slot in a double battle, indicating which actions are valid.
    Args:
        battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        slot_idx (int): The slot (which Pokemon) for which to get the action mask (0 for left slot, 1 for right slot).
    Returns:
        list[bool]: The action mask for the specified slot, where True indicates a valid action and False indicates \
        an invalid action.
    """
    mask = [False] * 26
    
    if any(battle.force_switch):
        if not battle.force_switch[slot_idx]:
            # Pokemon still alive, must pass
            mask[0] = True
            return mask
        else:
            bench = [
                pokemon for pokemon in battle.team.values() 
                if pokemon is not None 
                and not pokemon.fainted 
                and pokemon not in battle.active_pokemon 
                and pokemon.selected_in_teampreview
            ]
            available_switches = battle.available_switches[slot_idx]
            
            for action in range(24, 26):
                switch_index = action - 24
                if switch_index < len(bench) and bench[switch_index] in available_switches:
                    mask[action] = True
            
            if all(battle.force_switch) and len(available_switches) == 1:
                mask[0] = True
            
            if not any(mask):
                mask[0] = True
            
            return mask
    
    pokemon = battle.active_pokemon[slot_idx]
    
    if pokemon is None or pokemon.fainted:
        mask[0] = True # Dummy action to avoid fully NaN mask
        return mask
    
    opp1 = battle.opponent_active_pokemon[0]
    opp2 = battle.opponent_active_pokemon[1]
    partner = battle.active_pokemon[1 - slot_idx]
    valid_targets = {
        0: opp1 is not None and not opp1.fainted,
        1: opp2 is not None and not opp2.fainted,
        2: partner is not None and not partner.fainted
    }
    
    moves = list(pokemon.moves.values())
    available_moves = battle.available_moves[slot_idx]
    
    for action in range(12):
        move_index = action // 3
        target_index = action % 3
        if move_index < len(moves) and moves[move_index] in available_moves and valid_targets[target_index]:
            mask[action] = True
            if battle.can_mega_evolve[slot_idx]:
                mask[action + 12] = True
        if move_index < len(moves) and moves[move_index].id == "fakeout" and not pokemon.first_turn:
            mask[action] = False
            mask[action + 12] = False
    
    bench = [
        pokemon for pokemon in battle.team.values() 
        if pokemon is not None 
        and not pokemon.fainted 
        and pokemon not in battle.active_pokemon 
        and pokemon.selected_in_teampreview
    ]
    available_switches = battle.available_switches[slot_idx]
    
    for action in range(24, 26):
        switch_index = action - 24
        if switch_index < len(bench) and bench[switch_index] in available_switches and not battle.trapped[slot_idx]:
            mask[action] = True
            
    if not any(mask):
        mask[0] = True # Dummy action to avoid fully NaN mask
    
    return mask

def action_to_double_order(player: Player, battle: pkmn_b.DoubleBattle, actions: tuple[int, int]) -> DoubleBattleOrder:
    """
    Generate a double battle order based on the actions for each slot.
    Args:
        player (Player): The player object making the decision.
        battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        actions (tuple[int, int]): The actions to convert to orders (each 0-25). First element is for slot 0, second for slot 1.
    Returns:
        DoubleBattleOrder: The double battle order for the given slot index.
    """
    orders = []
    bench = [
        pokemon for pokemon in battle.team.values() 
        if pokemon is not None 
        and not pokemon.fainted 
        and pokemon not in battle.active_pokemon 
        and pokemon.selected_in_teampreview
    ]
    
    for slot_index, action in enumerate(actions):
        if any(battle.force_switch):
            # or is because both slots may be forced to switch with 1 remaining bench Pokemon, in which case one slot must pass
            if not battle.force_switch[slot_index] or action == 0:
                orders.append(PassBattleOrder())
            else:
                switch_index = action - 24
                if 0 <= switch_index < len(bench) and bench[switch_index] in battle.available_switches[slot_index]:
                    orders.append(player.create_order(bench[switch_index]))
                elif battle.available_switches[slot_index]:
                    orders.append(player.create_order(battle.available_switches[slot_index][0]))
                else:
                    orders.append(PassBattleOrder())
            continue

        pokemon = battle.active_pokemon[slot_index]
        
        if pokemon is None or pokemon.fainted:
            orders.append(PassBattleOrder())  # No order for fainted Pokémon
            continue
        
        if action < 24:
            moves = list(pokemon.moves.values())
            is_mega = action >= 12
            move_action = action % 12
            move_index = move_action // 3
            target_index = move_action % 3
            # target_pos = get_target_position(target_index, slot_index == 0)
            if move_index < len(moves):
                move = moves[move_index]
                # Check for spread move, field move, etc. since they want move_target=0
                if move.target in [
                    pkmn_b.Target.NORMAL, 
                    pkmn_b.Target.ANY, 
                    pkmn_b.Target.ADJACENT_FOE, 
                    pkmn_b.Target.ADJACENT_ALLY, 
                    pkmn_b.Target.ADJACENT_ALLY_OR_SELF
                ]:
                    target_pos = get_target_position(target_index, slot_index == 0)
                else:
                    target_pos = 0
                orders.append(player.create_order(order=move, move_target=target_pos, mega=is_mega))
            elif battle.available_moves[slot_index]:
                orders.append(player.create_order(order=battle.available_moves[slot_index][0]))
            else:
                orders.append(PassBattleOrder())
        else:
            switch_index = action - 24
            if 0 <= switch_index < len(bench) and bench[switch_index] in battle.available_switches[slot_index]:
                orders.append(player.create_order(order=bench[switch_index]))
            elif battle.available_switches[slot_index]:
                orders.append(player.create_order(order=battle.available_switches[slot_index][0]))
            else:
                orders.append(PassBattleOrder())
            
    return DoubleBattleOrder(*orders)

# ----------------- Custom Gymnasium Environment -----------------

class RLPlayer(Player):
    """
    Custom Player class for reinforcement learning.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.battle_queue = queue.Queue(maxsize=1)
        self.current_order_future: asyncio.Future | None = None
        
    async def choose_move(self, battle: pkmn_b.AbstractBattle) -> DoubleBattleOrder:
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

class VGCEnv(gym.Env):
    """
    Custom Gymnasium environment for Pokémon VGC battles.
    """
    def __init__(
        self, 
        agent: RLPlayer | None = None, 
        opponent: Player | None = None, 
        battle_format: str = "gen9championsvgc2026regmc"
    ):
        super().__init__()
        self.battle_format = battle_format
        self.agent = agent if agent is not None else RLPlayer(battle_format=self.battle_format)
        self.opponent = opponent if opponent is not None else RandomPlayer(battle_format=self.battle_format)
        self.state_encoder = StateEncoder()
        self.observation_space = spaces.Dict({
            "numeric": spaces.Box(low=-1.0, high=1.0, shape=(1149,), dtype=np.float32),
            "abilities": spaces.Box(low=0.0, high=1.0, shape=(8, 215), dtype=np.float32),
            "items": spaces.Box(low=0.0, high=1.0, shape=(8, 167), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(676, dtype=np.int64)
        
    def action_masks(self) -> np.ndarray:
        """
        Get the action masks for both slots in a double battle.
        Args:
            battle (pkmn_b.DoubleBattle): The battle object containing the current state.
        Returns:
            np.ndarray: A numpy array containing the combined action mask.
        """
        mask = []
        slot_a_mask = get_slot_action_mask(self.current_battle, 0)
        slot_b_mask = get_slot_action_mask(self.current_battle, 1)
        bench = [
            pokemon for pokemon in self.current_battle.team.values() 
            if pokemon is not None 
            and not pokemon.fainted 
            and pokemon not in self.current_battle.active_pokemon
            and pokemon.selected_in_teampreview
        ]
        
        for i in range(676):
            action_a = i // 26
            action_b = i % 26
            is_legal = slot_a_mask[action_a] and slot_b_mask[action_b]
            # Prevent both slots from switching to the same Pokemon at the same time
            if action_a >= 24 and action_b >= 24 and action_a == action_b:
                is_legal = False
            # If forced to switch and there is a bench mon, don't allow both slots to pass (force one to switch)
            if all(self.current_battle.force_switch) and len(bench) > 0 and action_a == 0 and action_b == 0:
                is_legal = False
            if (12 <= action_a < 24) and (12 <= action_b < 24):
                is_legal = False
            mask.append(is_legal)
        
        return np.array(mask, dtype=bool)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment for a new battle.
        Args:
            seed (int | None): Optional seed for reproducibility.
            options (dict[str, Any] | None): Optional dictionary of additional options.
        Returns:
            dict[str, np.ndarray]: The initial observation after resetting the environment.
        """
        super().reset(seed=seed)
        # Check if we can reuse the current battle (if it's still ongoing and it's the first turn)
        can_reuse_battle = (
            hasattr(self, "current_battle")
            and self.current_battle is not None
            and not self.current_battle.finished
            and self.current_battle.turn <= 1
            and self.agent.current_order_future is not None
            and not self.agent.current_order_future.done()
        )
        
        if not can_reuse_battle:
            # If a battle is unfinished mid-game, forfeit AND wait for Showdown to confirm it's closed
            if hasattr(self, "current_battle") and self.current_battle and not self.current_battle.finished:
                self.agent.receive_order(ForfeitBattleOrder())
                try:
                    while not self.current_battle.finished:
                        b = self.agent.battle_queue.get(timeout=2.0)
                        if b.battle_tag == self.current_battle.battle_tag:
                            self.current_battle = b
                except queue.Empty:
                    pass
            
            # Flush any stale finished battles from the queue
            while not self.agent.battle_queue.empty():
                try:
                    self.agent.battle_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Safely runs the async battle in the background thread of poke-env. This ensures that we don't need an async 
            # signature on the reset method, which is not supported by Gymnasium.
            asyncio.run_coroutine_threadsafe(
                self.agent.battle_against(self.opponent, n_battles=1), 
                # poke-env spawns a dedicated OS background thread, which is the ps_client.loop
                # When Showdown sends a new battle state, it is processed in the ps_client.loop thread, 
                # which then calls choose_move.
                self.agent.ps_client.loop
            )
            # Gets the initial state as soon as agent calls choose_move, which then blocks until the agent provides an order.
            self.current_battle: pkmn_b.DoubleBattle = self.agent.battle_queue.get()

        opp_mons = [
            p for p in self.current_battle.opponent_team.values() if p is not None 
            and not p.fainted 
        ]
        self_mons = [
            p for p in self.current_battle.team.values() if p is not None 
            and not p.fainted 
        ]
        self.num_opponent_alive = len(opp_mons)
        self.num_self_alive = len(self_mons)
        state = self.state_encoder.encode(self.current_battle)
        self.opp_hp = sum(p.current_hp_fraction for p in opp_mons) if opp_mons else 0.0
        self.self_hp = sum(p.current_hp_fraction for p in self_mons) if self_mons else 0.0
        return state, {}
    
    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        An action is taken by the agent, which is converted into a DoubleBattleOrder and sent to the agent's order queue. \
        The environment then waits for the next state of the battle after the action is processed.
        Args:
            action (int): The action to take (0-675).
        Returns:
            tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]: A tuple containing the next observation, reward, \
            done flag (did the battle end?), truncated flag, and additional info.
        """
        if self.current_battle.turn >= 30:
            self.agent.receive_order(ForfeitBattleOrder())
            self.current_battle = self.agent.battle_queue.get()
            state = self.state_encoder.encode(self.current_battle)
            return state, -1.0, True, True, {}
        
        action_a = action // 26
        action_b = action % 26

        order = action_to_double_order(
            self.agent, 
            self.current_battle, 
            (action_a, action_b)
        )
        self.agent.receive_order(order)
        # Wait for the next state of the battle after the action is processed.
        self.current_battle = self.agent.battle_queue.get()
        reward = self.calculate_reward(self.current_battle)
        done = self.current_battle.finished
        state = self.state_encoder.encode(self.current_battle)
        return state, reward, done, False, {}
    
    def calculate_reward(self, battle: pkmn_b.DoubleBattle) -> float:
        """
        Calculate the reward for the current battle state.
        Args:
            battle (pkmn_b.DoubleBattle): The current battle state.
        Returns:
            float: The calculated reward.
        """
        opp_mons = [
            p for p in battle.opponent_team.values() if p is not None 
            and not p.fainted 
        ]
        self_mons = [
            p for p in battle.team.values() if p is not None 
            and not p.fainted 
        ]
        
        curr_opponent_alive = len(opp_mons)
        curr_self_alive = len(self_mons)
        curr_opp_hp = sum(p.current_hp_fraction for p in opp_mons) if opp_mons else 0.0
        curr_self_hp = sum(p.current_hp_fraction for p in self_mons) if self_mons else 0.0
        
        reward = 0.0
        reward += (self.num_self_alive - curr_self_alive) * -0.1
        reward += (self.num_opponent_alive - curr_opponent_alive) * 0.1
        reward += (curr_self_hp - self.self_hp) * 0.02 # if negative, punish
        reward += (curr_opp_hp - self.opp_hp) * -0.02
        
        self.num_self_alive = curr_self_alive
        self.num_opponent_alive = curr_opponent_alive
        self.self_hp = curr_self_hp
        self.opp_hp = curr_opp_hp
        
        return reward + 2.0 if battle.won else reward - 2.0 if battle.lost else reward
    
    def close(self) -> None:
        """
        Close the environment and clean up resources.
        """
        # Do nothing as Python and the OS kernel will clean up the background thread and resources when the process exits.
        pass
