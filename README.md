# Installation

```bash
git clone --recursive https://github.com/jirachixu/pokemon-battle-ai.git
cd showdown-server
npm install
node pokemon-showdown start --no-security
```

In another terminal, in the pokemon-battle-ai directory:

```bash
python3 -m scripts.play_against_bot
```

## Note: 

`scripts/play_against_bot.py` is not currently implemented. You can run the `src/scrape_teams.py` script and train the model yourself using the `scripts/train_expert_iteration.py` script.
