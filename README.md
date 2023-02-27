# Equivariant Augmented Normalizing Flows


# Overview of the repo
- For blackjax we need `pip install blackjax-nightly` (latest version) otherwise we get errors.
- `flow`: Contains the base distributions, bijectors (e.g. projected flow transform), and EGNNs.
- `examples/train.py`: Generic training script for all problems.
- `utils/train_and_eval.py`: Various utils used for training, such as the loss function and evaluation functions.
- `examples/dw4.py` script for running the dw4 problem. Set `local_config = True` inside the py file to run locally.
- **logging**: I am using wandb: https://wandb.ai/flow-ais-bootstrap?shareProfileType=copy. I also have a minimal 
ListLogger that can may used if you don't want to use wandb.
- `qmp.py` the whole folder is a copy and paste (with some small adjustments). The key file is `dataset.py` which 
**MUST** be run before running `examples/qm9.py`.


# TODO:

## Burning
    - Add code that allows visualisation of what's happening in the flow.
    - Rewrite for e3nn
        1. Write EGNN with e3nn, retain MACE performance
        2. Let networks take in positional and feature info.
        3. Alanine dipeptide
    - Code diffusion/CNF
    - Rewrite with jgraph, generalise to varying number of nodes
    - Add more augmented variables.
    - Think of directly parameterizing the basis (3 free params) for the proj flow. 

## Less burning
    - Spline flows
    - Think of initialisation for Egnn that encourages random vectors of reasonable magnitude and as non-collinear as possible.
    - e3nn outputs are typically shift **invariant** rather than equivariant. 
    The flow transforms should use this, which also allows for simplification. 
        - For example, we no longer need to calculate vectors in such a hacky way, as this is the typical output. 
        - For the projected flow, we can set the origin to the "current point" and then do (x + shift)*scale as 
          our transform. This completely the pesky calculation of origin, and basis vectors - origin etc.  
    - Cut down number of forward passes during evaluation.
    - Move shift to come before scale?
    - Seems like we could try enforce q(x, a) \propto p(a) = N(\mu=x, \sigma) in an additional loss?
      This would help decrease the variance in the estimate of the marginal q(x) = E_{p(a)}[q(x, a)/p(a)]. 