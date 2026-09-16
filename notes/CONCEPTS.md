# RL Concepts

## Encoding The State

We need to encode the state into a form that the neural network can understand. For this project, the observation space is all the information of the board state on a given turn. We represent each "aspect" (e.g. field effects like weather/terrain, the Pokemon themselves and their moves/stats, etc.). We flatten the tensor at the end so it can be fed into the network. Read `state_encoder.py` for more concrete examples.

## Action Space

We also have to represent each action for the neural network. In this project, the action space is basically all the moves and switches that could possibly be made in a given turn, regardless of legality in the rules of Pokemon. The network assigns probabilities (logits) for each action. For that, we create a mask that tells the neural network to essentially assign 0 probability to any illegal actions.

### The All-False Mask

We must be careful not to apply a mask with only `False` values. Because the mask changes all `False`-valued actions (illegal actions in this case) to have logits of $-\infty$, when we softmax the logits, we would get $\frac{e^{\tilde{z}_i}}{\sum e^{\tilde{z}_j}} = \mathrm{NaN}$ for each action $i$, where $\tilde{z}$ is the tensor of logits. PyTorch will crash when trying to sample from distributions containing `NaN`, and worse, during the backward pass (gradient update), the gradient for each weight will be calculated as $\nabla_{\theta}L = \mathrm{NaN}$, so when the weights are updated, each weight will become $\theta_{new} = \theta_{old} - \alpha \cdot \mathrm{NaN} = \mathrm{NaN}$, destroying the model. To fix this, we can just set one dummy action in the mask to `True` and ignore it since we know it's illegal.

## "Translating" Between 2 "Worlds"

RL algorithms deal in pure floats and run in a clean loop (world 1). However, Pokemon Showdown (poke-env) runs asynchronously on the server (world 2). Additionally, Showdown does not return data as the pure floats that the RL algorithm needs! We need to create an environment that bridges the gap between these two. Gymnasium environments can do this. It runs Showdown in the background and translates Showdown's battle events into synchronous data for the RL algorithm to use.

When the RL algorithm steps and passes in 2 action indices, the environment translates them into a `DoubleBattleOrder` and sends it to the Showdown server. It also handles action masking, letting the RL algorithm know which actions are legal. When Showdown returns the new battle state after the actions are taken, the environment translates this into something the RL algorithm can understand. This is why we have to specify the observation space and action space to be a certain size; the RL algorithm has to know the size of the tensor it accepts and outputs!

Additionally, Gymnasium environments are ideal because they can be used with prewritten, high-performance RL libraries with 0 modification, they can run RL agents in parallel, and they can catch incorrect input/output sizes (for example, index of 26 when you need [0..25]) before they destroy the neural network's weights.

## PPO

PPO is an actor-critic reinforcement learning method. The actor $\pi_{\theta}(a | s)$ is the part that reads in the state and outputs probabilities assigned to each action (probability of actions given state, the policy), while the critic $V_{\phi}(s)$ (the value function) takes in the state and outputs an expected future reward of this *state* assuming the agent follows the current policy ("win equity"). Both of these components are learned simultaneously.

### Probability Ratio

One problem of early policy gradient algorithms (PPO is a policy gradient algorithm, meaning it learns the best policy to maximize reward), a big issue was models taking too big of a gradient step, leading to model collapse, collecting only garbage data, as the data comes from its own actions. PPO limits the probability ratio (ratio between the new policy and an old policy) $$r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$ to be within some range $[1 - \epsilon, 1 + \epsilon]$, where $\epsilon$ is provided as a hyperparameter, typically $0.2$. This prevents the model from changing the policy too quickly.

### Advantage Function

Another consideration of PPO is in how actions are rewarded. Specifically, actions shouldn't be rewarded simply because the outcome was good. In the Pokemon example, what if the agent blunders on turn 4 but wins the game on turn 8 because it was bailed out by a lucky crit, a flinch, etc.? This is where the advantage function $\hat{A}_t = Q(s_t, a_t) - V(s_t)$ comes into play. $Q(s_t, a_t)$ is the function that the expected reward of this *action*. The $Q$ function and the value function are related: $V^{\pi}(s) = \sum_{a \in A} \pi(a | s)Q^{\pi}(s, a)$. When the advantage function is greater than 0, the action was better than expected and should be made more likely, and vice versa for when it is less than 0. However, in the actor-critic method, $Q$ is unknown, and so the advantage function is calculated with an approximation using *generalized advantage estimation* (GAE). I won't dive into the mathematical details because it's probably not too useful for me currently; what's important is that there are two discount factors $\gamma$ and $\lambda$ (reward discount and bias-variance tradeoff) that are usually $0.99$ and $0.95$ respectively, and that GAE makes PPO resilient against noise (such as critical hits, flinches, accuracy, damage rolls, etc. in the case of Pokemon).

### Loss Function

The PPO loss function actually has 3 parts, first of which is the following: $$\textit{L}^{CLIP}(\theta) = \hat{\mathbb{E}}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t)]$$ which can generally be understood based on the variables introduced earlier ($\hat{\mathbb{E}}_t$ means the empirical expectation over the timesteps). This $\min$ part is simply the mathematical way to express, "get the lower between the probability ratio and the clipped ratio", or "don't let the ratio become bigger than or smaller than some value". This works because the advantage function is positive for "good" actions and negative for "bad" ones, meaning the gradient will point uphill for "good" actions and vice versa. The clipping means that once the ratio reaches a certain number, the gradient will become 0 and thus prevents the model from continuously increasing the probability for that action. This is the part that refines the actor (the policy).

The second part is the part that refines the critic (value function): $$\textit{L}^{VF}(\phi) = \hat{\mathbb{E}}_t[(V_{\phi}(s_t) - V_t^{\text{target}})^2]$$ where $V_t^{\text{target}} = \hat{A}_t + V_{\phi_{old}}(s_t)$. This part ensures that the value function can learn the nuances of the state, as it shares the same neural network trunk as the policy until the last layer. It is multiplied by a weighting coefficient in the loss function $c_1$, usually $0.5$.

The final part is the part that incentivizes trying different actions, $$S[\pi_{\theta}](s_t) = -\sum_{a \in A} \pi_{\theta}(a | s_t)\log{\pi_{\theta}(a | s_t)}$$ which introduces entropy to the loss, meaning that no action can end up with 100% probability. This is multiplied with a weighting coefficient $c_2$ that controls how much entropy there should be.

The total loss looks like this: $$\text{Loss}_{\text{total}}(\theta, \phi) = -\textit{L}^{CLIP}(\theta) + c_1\textit{L}^{VF}(\phi) - c_2S[\pi_{\theta}](s_t)$$

### Steps in Training

1. Rollout (Data Collection): The current policy $\pi_{\theta_{\text{old}}}$ takes `n_steps` number of steps (e.g. 2048). At each turn $t$ the network outputs the action distribution $\pi_{\text{old}}(a | s_t)$ and the value estimate $V_{\text{old}}(s_t)$. The agent samples the action distribution and gets $a_t$ and saves everything into memory.
2. GAE: Once all steps are calculated, PPO runs GAE backwards from step 2048 down to 1 and standardizes advantages calculated for all 2048 steps.
3. Optimization: PPO then takes the 2048 steps and slices random batches of `batch_size`, then trains for `n_epochs` epochs. This is where new weights, probability ratio, and loss are calculated and optimized.
4. Discard, Repeat: PPO discards the 2048 steps as they are now outdated (generated by an old policy). The new weights $\theta$ are now stored as the new $\theta_{\text{old}}$ and the loop repeats from phase 1.

### Hyperparameters

| Hyperparameter | Typical Value | What It Controls | What Happens If Too High | What Happens If Too Low |
| :--- | :--- | :--- | :--- | :--- |
| `learning_rate` ($\alpha$) | $3 \times 10^{-4}$ | Step size for the Adam optimizer | Model destabilizes, win rate can crash to 0% | Learning crawls at a snail's pace |
| `n_steps` | $2,048$ | Turns collected before each gradient update | Policy updates become slow; rollout data gets stale | Gradients become noisy; reduced long-term vision |
| `batch_size` | $64$ or $128$ | Mini-batch size for SGD updates | Too smooth; requires more GPU memory | Noisy gradient updates; training becomes erratic |
| `n_epochs` | $10$ | Passes over the rollout buffer per update | Overfits to the batch | Wastes simulation data; agent learns too slowly |
| `gamma` ($\gamma$) | $0.99$ to $0.995$ | Discount factor for future rewards | Prioritizes long-term strategy (turns 15–20) | Bot becomes short-sighted / greedy for chip damage |
| `gae_lambda` ($\lambda$) | $0.95$ | Variance vs. bias in Advantage estimation | Smoothes out crit and damage roll noise | High variance; lucky crits skew learning |
| `clip_range` ($\epsilon$) | $0.2$ | The policy ratio clipping threshold | Risk of destructive policy collapse | Updates are too timid; policy learning stalls |
| `ent_coef` ($c_2$) | $0.01$ | Weight of the curiosity / entropy bonus | Bot plays randomly and won't commit to lethal KOs | **Entropy Collapse:** Bot locks onto 1 move and stops exploring |
| `vf_coef` ($c_1$) | $0.5$ | Weight of Critic loss in the total loss | Critic overpowers Actor; policy ignores move selection | Critic learns poorly; advantage estimates become inaccurate |
| `max_grad_norm` | $0.5$ | Gradient clipping threshold | Exploding gradients on rare/unusual turn states | Gradients are truncated too heavily |
