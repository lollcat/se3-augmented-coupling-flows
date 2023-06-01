from examples.analyse_results.get_wandb_runs import download_checkpoint


def download_qm9_models():
    flow_types = ['spherical', 'along_vector', 'proj', 'non_equivariant']
    seeds = [0]
    for flow_type in flow_types:
        for seed in seeds:
            download_checkpoint(flow_type=flow_type, tags=["qm9pos", "post_kigali_1"], seed=seed, max_iter=128,
                                base_path='./examples/qm9_results/models')


if __name__ == '__main__':
    download_qm9_models()
