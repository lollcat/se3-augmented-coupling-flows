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
    - Big rewrite: with jgraph and e3nn
    - e3nn outputs are typically shift **invariant** rather than equivariant. 
    The flow transforms should use this, which also allows for simplification. 
        - For example, we no longer need to calculate vectors in such a hacky way, as this is the typical output. 
        - For the projected flow, we can set the origin to the "current point" and then do (x + shift)*scale as 
          our transform. This completely the pesky calculation of origin, and basis vectors - origin etc.  