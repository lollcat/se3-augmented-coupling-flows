# Equivariant Augmented Normalizing Flows

Equivariance looking broken for multiplicity. 

# Overview of the repo
- `flow`: Contains the base distributions, bijectors (e.g. projected flow transform), and EGNNs.
- `examples/train.py`: Generic training script for all problems.
- `utils/train_and_eval.py`: Various utils used for training, such as the loss function and evaluation functions.
- `examples/dw4.py` script for running the dw4 problem. Set `local_config = True` inside the py file to run locally.
- **logging**: I am using wandb: https://wandb.ai/flow-ais-bootstrap?shareProfileType=copy. I also have a minimal 
ListLogger that can may used if you don't want to use wandb.
- `qmp.py` the whole folder is a copy and paste (with some small adjustments). The key file is `dataset.py` which 
**MUST** be run before running `examples/qm9.py`.
- We have to be very careful with haiku params, as the flow forward and reverse call the params in different orders. 
Hence naming of modules is crucial. 


# Install
For the alanine dipeptide problem we need to install openmmtools with conda:
`conda install -c conda-forge openmm openmmtools`

## Experiments
```shell
python examples.dw4.py
python examples/lj13.py
python examples/aldp.py
python examples/qm9_download_data.py
```

## Instalation

1. Install packages in requirements.txt
2. Run 

```
pip install -e .
```


# TODO:

## Burning
    - Spline flows
    - Clean proj realnvp config args (has unused args that are no longer an option). 
    - Centralise net config naming to be more consistent. 
    - Add whether sample has aux to sample info?
    - Think of directly parameterizing the basis (3 free params) for the proj flow. 
    - Make sure non-equivariant flow is working well.

## Less burning

    - Rewrite with jgraph, generalise to varying number of nodes. 
    - Also have some augmented graph nodes?
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
    - Lots of *_test.py files are no longer working, as the code has changed since they were written. These should be rewritten. 