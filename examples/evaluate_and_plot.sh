echo $PWD
export PYTHONPATH=$PWD

# QM9. Download checkpoints, plot and evaluate.
mkdir examples/qm9_results/models
mkdir examples/plots
python examples/qm9_results/download_models.py
python examples/qm9_results/plot.py