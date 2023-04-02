# Equivariant Augmented Normalizing Flows

Equivariance looking broken for multiplicity. 

# Overview of the repo
- `examples/dw4.py` script for running the dw4 problem. Set `local_config = True` inside the py file to run locally.
- **logging**: I am using wandb: https://wandb.ai/flow-ais-bootstrap?shareProfileType=copy. I also have a minimal 
ListLogger that can may used if you don't want to use wandb.
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
    - Spherical Flow
        - Global Frame Layer.
    - Base distribution.
        - Sampld d from a distribution
        - Sample direction randomly. 
    - Act norm options: 
        - Scaling towards centre of mass. 
        - Scaling towards augmented pair. 
        - Condition on graph (use transfomrer). 
    - Make sure non-equivariant flow is working well.

## Less burning

    - Rewrite with jgraph, generalise to varying number of nodes.
    - Cut down number of forward passes during evaluation.
    - Seems like we could try enforce q(x, a) \propto p(a) = N(\mu=x, \sigma) in an additional loss?
      This would help decrease the variance in the estimate of the marginal q(x) = E_{p(a)}[q(x, a)/p(a)]. 
    - Lots of *_test.py files are no longer working, as the code has changed since they were written. These should be rewritten. 