# Mgr - normalizing flows


## What is it about?

In this repo explore normalizing flows and try to asses their usefulness in monte carlo estimations - i try Crude Monte Carlo and Stratified Sampling.

## Structure of repo

Currently `utils.py` contains all written functions and modules. I conducted several experiments, they can be accessed opening TensorBoard using
```
tensorboard --logdir <directory_to_experiments folder>
```
providing one of the directories in the subfolder of `Experiments in TensorBoard`.

## Results

In a clean version, results of my code are shown in a notebook called `Overview of methods` https://github.com/SirPentyna/mgr_normalizing_flows/blob/main/Overview%20of%20methods.ipynb

## Sources

My code is heavily related to implementation: https://github.com/VincentStimper/normalizing-flows
In fact, what I tried to understand the code written above. What I ended up doing, I tried to implement Normalizing Flows on my own, and when I got stuck I looked through this repository. 

## References

It is based on classic papers:
- NICE: https://arxiv.org/abs/1410.8516
- RealNVP: https://arxiv.org/abs/1605.08803

and other.

