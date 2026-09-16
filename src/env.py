import poke_env.battle as pkmn_b
from poke_env.player import Player, RandomPlayer
from poke_env.player.battle_order import DoubleBattleOrder

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
    
    bench = [
        pokemon for pokemon in battle.team.values() 
        if pokemon is not None and pokemon not in battle.active_pokemon and pokemon.selected_in_teampreview
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
    
    for slot_index, action in enumerate(actions):
        pokemon = battle.active_pokemon[slot_index]
        
        if pokemon is None or pokemon.fainted:
            orders.append(None)  # No order for fainted Pokémon
            continue
        
        if action < 24:
            moves = list(pokemon.moves.values())
            is_mega = action >= 12
            move_action = action % 12
            move_index = move_action // 3
            target_index = move_action % 3
            target_position = get_target_position(target_index, slot_index == 0)
            orders.append(player.create_order(order=moves[move_index], move_target=target_position, mega=is_mega))
        else:
            switch_index = action - 24
            bench = [
                pokemon for pokemon in battle.team.values() 
                if pokemon is not None and pokemon not in battle.active_pokemon and pokemon.selected_in_teampreview
            ]
            orders.append(player.create_order(order=bench[switch_index]))
            
    return DoubleBattleOrder(*orders)

# ----------------- Custom Gymnasium Environment -----------------

class RLPlayer(Player):
    """
    Custom Player class for reinforcement learning.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.battle_queue = queue.Queue(maxsize=1)
        self.order_queue = queue.Queue(maxsize=1)
        
    def choose_move(self, battle: pkmn_b.AbstractBattle) -> DoubleBattleOrder:
        """
        Puts the current battle state into the queue and waits for the RL agent to provide an order. Because the battle is \
        put into the queue before the order is retrieved, this ensures that the RL agent always has the most up-to-date \
        battle state when making a decision. Once the order is returned, Showdown will process the order and call \
        choose_move again with the next battle state, etc.
        Args:
            battle (pkmn_b.AbstractBattle): The current battle state.
        Returns:
            DoubleBattleOrder: The Order object chosen by the RL agent.
        """
        assert isinstance(battle, pkmn_b.DoubleBattle), "Expected a DoubleBattle instance"
        self.battle_queue.put(battle)
        return self.order_queue.get()
    
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
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(565,), dtype=np.float32
        )
        # action_a = action // 26, action_b = action % 26
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
        
        for i in range(676):
            action_a = i // 26
            action_b = i % 26
            is_legal = slot_a_mask[action_a] and slot_b_mask[action_b]
            # Prevent both slots from switching to the same Pokemon at the same time
            if action_a >= 24 and action_b >= 24 and action_a == action_b:
                is_legal = False
            mask.append(is_legal)
            
        return np.array(mask, dtype=bool)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment for a new battle.
        Args:
            seed (int | None): Optional seed for reproducibility.
            options (dict[str, Any] | None): Optional dictionary of additional options.
        Returns:
            np.ndarray: The initial observation after resetting the environment.
        """
        super().reset(seed=seed)
        # Safely runs the async battle in the background thread of poke-env. This ensures that we don't need an async signature 
        # on the reset method, which is not supported by Gymnasium.
        asyncio.run_coroutine_threadsafe(
            self.agent.battle_against(self.opponent, n_battles=1), 
            # poke-env spawns a dedicated OS background thread, which is the ps_client.loop
            # When Showdown sends a new battle state, it is processed in the ps_client.loop thread, which then calls choose_move.
            self.agent.ps_client.loop
        )
        # Gets the initial state as soon as agent calls choose_move, which then blocks until the agent provides an order.
        self.current_battle = self.agent.battle_queue.get()
        state = self.state_encoder.encode(self.current_battle).numpy()
        return state, {}
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        An action is taken by the agent, which is converted into a DoubleBattleOrder and sent to the agent's order queue. \
        The environment then waits for the next state of the battle after the action is processed.
        Args:
            action (int): The action to take (0-675).
        Returns:
            tuple[np.ndarray, float, bool, bool, dict[str, Any]]: A tuple containing the next observation, reward, \
            done flag (did the battle end?), truncated flag, and additional info.
        """
        action_a = action // 26
        action_b = action % 26
        order = action_to_double_order(self.agent, self.current_battle, (action_a, action_b))
        self.agent.order_queue.put(order)
        
        # Wait for the next state of the battle after the action is processed.
        self.current_battle = self.agent.battle_queue.get()
        state = self.state_encoder.encode(self.current_battle).numpy()
        reward = self.calculate_reward(self.current_battle)
        done = self.current_battle.finished
        
        return state, reward, done, False, {}
    
    def calculate_reward(self, battle: pkmn_b.DoubleBattle) -> float:
        """
        Calculate the reward for the current battle state.
        Args:
            battle (pkmn_b.DoubleBattle): The current battle state.
        Returns:
            float: The calculated reward.
        """
        # Placeholder for reward calculation logic
        return 0.0
    
    def close(self) -> None:
        """
        Close the environment and clean up resources.
        """
        # Do nothing as Python and the OS kernel will clean up the background thread and resources when the process exits.
        pass
