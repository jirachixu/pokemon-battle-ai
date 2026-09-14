# Work Diary

I will be documenting my thought process and the general changes I have made, along with future plans and whatnot here.

## 2026/9/13

I started this project for a couple of reasons. First, I of course wanted a nice project for my resume. Second, I'm an AI major, so I figured that I should maybe have some more ML/AI related projects anyways. And third, I am a big Pokemon fan, and have wanted to get better at battling for a long time, but playing on the ladder is kind of anxiety-inducing. So, I wanted to solve that.

I've set up a random double battles bot simply to test, and I found that it can do around 135 turns per second, which seems quite good as VGC battles take like 10 turns or so on average assuming that players know to some extent what they're doing. This means around 13 games per second.

I have realized that this is probably a pretty difficult problem to try to tackle. There are simply SO many variables in competitive Pokemon, especially VGC doubles. Of course, this means two things: if I'm able to actually do it, it's all the more impressive, but at the same time, it will be difficult for me to adjust to the learning curve.

## 2026/9/14

I'm gonna start with training a very rudimentary agent. This agent will only really be able to make judgments based on Pokemon's moves, types, stat changes, and field effects. The expectation isn't for it to be *good* at battling, but I do expect that after training, it should be able to at least battle much better than just randomly selecting moves.

I have created the rudimentary state encoder to encode the board state of the current turn, and my next steps will be to actually get an RL agent up and train it.
