# Equivariant Augmented Normalizing Flows


# Install
Uses jax 0.4.8
```
conda install jaxlib=*=*cuda* jax=0.4.8 cuda-nvcc -c conda-forge -c nvidia
```
Has dependency on pytorch (NB: use CPU version).
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
```shell
python examples/dw4.py
python examples/lj13.py
python examples/aldp.py
python examples/qm9.py
python examples/dw4_fab.py
python examples/lj13_fab.py
```