# RL Concepts

## Encoding The State

We need to encode the state into a form that the neural network can understand. For this project, the observation space is all the information of the board state on a given turn. We represent each "aspect" (e.g. field effects like weather/terrain, the Pokemon themselves and their moves/stats, etc.). We flatten the tensor at the end so it can be fed into the network. Read `state_encoder.py` for more concrete examples.

## Action Space

We also have to represent each action for the neural network. In this project, the action space is basically all the moves and switches that could possibly be made in a given turn, regardless of legality in the rules of Pokemon. The network assigns probabilities (logits) for each action. For that, we create a mask that tells the neural network to essentially assign 0 probability to any illegal actions.

### The All-False Mask

We must be careful not to apply a mask with only `False` values. Because the mask changes all `False`-valued actions (illegal actions in this case) to have logits of $-\infty$, when we softmax the logits, we would get $\frac{e^{z_i}}{\sum e^{z_j}} = \mathrm{NaN}$ for each action $i$, where $z$ is the tensor of logits. PyTorch will crash when trying to sample from distributions containing `NaN`, and worse, during the backward pass (gradient update), the gradient for each weight will be calculated as $\nabla_{\theta}L = \mathrm{NaN}$, so when the weights are updated, each weight will become $\theta_{new} = \theta_{old} - \alpha \cdot \mathrm{NaN} = \mathrm{NaN}$, destroying the model. To fix this, we can just set one dummy action in the mask to `True` and ignore it since we know it's illegal.

## "Translating" Between 2 "Worlds"

RL algorithms deal in pure floats and run in a clean loop (world 1). However, Pokemon Showdown (poke-env) runs asynchronously on the server (world 2). Additionally, Showdown does not return data as the pure floats that the RL algorithm needs! We need to create an environment that bridges the gap between these two. Gymnasium environments can do this. It runs Showdown in the background and translates Showdown's battle events into synchronous data for the RL algorithm to use.

When the RL algorithm steps and passes in 2 action indices, the environment translates them into a `DoubleBattleOrder` and sends it to the Showdown server. It also handles action masking, letting the RL algorithm know which actions are legal. When Showdown returns the new battle state after the actions are taken, the environment translates this into something the RL algorithm can understand. This is why we have to specify the observation space and action space to be a certain size; the RL algorithm has to know the size of the tensor it accepts and outputs!

Additionally, Gymnasium environments are ideal because they can be used with prewritten, high-performance RL libraries with 0 modification, they can run RL agents in parallel, and they can catch incorrect input/output sizes (for example, index of 26 when you need [0..25]) before they destroy the neural network's weights.
