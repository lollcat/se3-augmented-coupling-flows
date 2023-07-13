# Equivariant Augmented Normalizing Flows


# Install
At time of publishing was using jax 0.4.13 with python 3.10. 
```
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
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