import asyncio

from poke_env.player import RandomPlayer

async def main():
    n_battles = 0
    player_1 = RandomPlayer(max_concurrent_battles=24, battle_format="gen9championsrandomdoublesbattle")
    player_2 = RandomPlayer(max_concurrent_battles=24, battle_format="gen9championsrandomdoublesbattle")
    
    while n_battles < 24:
        await player_1.battle_against(player_2, n_battles=24)

        print(f"Finished battles: {player_1.n_finished_battles}")
        print(f"Player 1 wins: {player_1.n_won_battles}")
        n_battles += player_1.n_finished_battles
        
    for battle_id, battle in player_1.battles.items():
        print(f"Battle {battle_id}")
        print(f"Turns: {battle.turn}")
        print(f"Did Player 1 Win: {battle.won}")
        print("--- PLAYER 1'S TEAM ---")
        for identifier, pokemon in battle.team.items():
            moves = ", ".join(pokemon.moves.keys())
            print(f"- {pokemon.species} (Level {pokemon.level}) | Moves: {moves}")
        print("\n--- PLAYER 2'S REVEALED TEAM ---")
        for identifier, pokemon in battle.opponent_team.items():
            # If they haven't used any moves yet, this will be empty!
            revealed_moves = ", ".join(pokemon.moves.keys()) if pokemon.moves else "None revealed"
            print(f"- {pokemon.species} (Level {pokemon.level}) | Revealed Moves: {revealed_moves}")

if __name__ == "__main__":
    asyncio.run(main())