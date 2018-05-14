# Policy Gradient

This respiratory is some simple realizations of policy gradient method to solve some simple reinforcement learning problems.

Short corridor is a simple environment solved with tabular policy gradient.

The program PGagent.py use neural networks to parameterize policy gradient method.

For more detail please see the file directly.

# Performance on Cartople

![image](https://github.com/antaressgzz/policy-gradient/blob/master/picture/rewards.PNG)

This is the exponential smoothed rewards of the PGagent on cartople problem. Default hyper-parameters is used. The reward increases after a while, indicating that the agent has learned something. But the performance is unstable. The performance may improve if hyper-parameters are better tuned or if the agent is trained for a longer period.
