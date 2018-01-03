# Torcs-with-DDPG
### Implementation of Deep Deterministic Policy Gradient with Keras in TORCS racing car video-game

We used Deep Deterministic Policy Gradient [1][2] to build an agent that plays
TORCS: a racing car video-game.

The principal component of our car controller consists in a feed forward neural
network which uses 29 inputs (track angle, track position, speeds along 3 axis,
RPM, 4 wheel spin velocities and 19 proximity sensors) to predict 2 values from
-1 to 1 that represent the steering (-1=full right and 1=full left) and the 
acceleration/brace (-1=full brake and 1=full accelerate) respectively. We choose to
merge both acceleration and brake into a single output in order to reduce the
number of tunable parameters along with the action space size. Making acceleration
and brake mutually exclusive experimentally led to comparable results.

Furthermore, the necessity of flexibility resulted in the creation of multiple
smaller models, each one trained on a different kind of track with a specific
reward function, rather than a single one. These trained model were merged
into a better performing controller by combining the outputs during the race [3].

Our work shows how the use reinforcement learning on continuous domains can
lead to excellent results even with a relatively small feed-forward network and
limited computational resources. We show how curriculum learning technique
boosted the convergence of the training process and we highlight how averaging
multiple networks can have a positive effect reducing the variance of the resulting
model, increasing its precision and allowing parallel training. Heuristics and
in-domain knowledge can help increasing stability and performances while the
emerging complexity can be successfully handled by genetic algorithms.

See ![report](https://github.com/nicola-decao/Torcs-with-DDPG/blob/master/ensembling-deep-deterministic.pdf)
for further details.

### References

[1] D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, and M. Riedmiller.
Deterministic policy gradient algorithms. 2014.

[2] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra. Continuous control with deep reinforcement learning.
CoRR, abs/1509.02971, 2015.

[3] Y. Bengio, J. Louradour, R. Collobert, and J. Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine
learning, pages 41–48. ACM, 2009.
