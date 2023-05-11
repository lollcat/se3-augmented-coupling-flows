import re
import os

from examples.get_wandb_runs import download_checkpoint


if __name__ == '__main__':
    flow_types = ['spherical', 'along_vector', 'proj', 'non_equivariant']
    seeds = [0]
    for flow_type in flow_types:
        for seed in seeds:
            download_checkpoint(flow_type=flow_type, tags=["lj13", "post_kigali_1"], seed=seed, max_iter=256,
                                base_path='./examples/lj13_results/models')