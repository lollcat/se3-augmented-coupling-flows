# SE(3) Equivariant Augmented Coupling Flows
Code for the paper # TODO add arxiv link. 
Results can be obtained by running the commands in the [Experiments](#experiments) section.


**Note**: This code is being currently being cleaned and documented further. 
Additionally, we will add a Quickstart jupyter notebook. 

# Install
At time of publishing was using jax 0.4.13 with python 3.10. 
```
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```
Has dependency on pytorch (NB: use CPU version so it doesn't clash with JAX).
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
For the alanine dipeptide problem we need to install openmmtools with conda:
```
conda install -c conda-forge openmm openmmtools
```
Finally then,
```
pip install -e .
```

## Experiments
Experiments may be run with the following commands. The flow type may be set as shown in the first line.
```shell
python examples/dw4.py flow.type=spherical # Flow types: spherical,proj,along_vector,non_equivariant
python examples/lj13.py
python examples/qm9.py
python examples/aldp.py
python examples/dw4_fab.py
python examples/lj13_fab.py
```

## Citation

If you use this code in your research, please cite it as:

```
@article{
midgley2023se3couplingflow,
title={SE(3) Equivariant Augmented Coupling Flows},
author={Laurence Illing Midgley and Vincent Stimper and Javier Antor{\'a}n and Emile Mathieu and Bernhard Sch{\"o}lkopf and Jos{\'e} Miguel Hern{\'a}ndez-Lobato},,
journal={arXiv preprint arXiv:TODO}
year={2023},
}
```
