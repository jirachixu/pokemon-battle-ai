# test_encoder.py
import asyncio
from poke_env.player import RandomPlayer
from src.state_encoder import StateEncoder
import poke_env.battle as pkmn_b

class TestingPlayer(RandomPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = StateEncoder()

    def choose_move(self, battle):
        assert isinstance(battle, pkmn_b.DoubleBattle), "Expected a DoubleBattle instance"
        # Test the encoder on every turn!
        state_tensor = self.encoder.encode(battle)
        print(f"Turn {battle.turn} - Encoded state shape: {state_tensor.shape}")
        return super().choose_move(battle)

async def main():
    player = TestingPlayer(battle_format="gen9championsrandomdoublesbattle")
    opponent = RandomPlayer(battle_format="gen9championsrandomdoublesbattle")
    await player.battle_against(opponent, n_battles=1)

asyncio.run(main())