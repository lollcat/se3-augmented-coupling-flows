from examples.get_wandb_runs import download_checkpoint


def download_dw4_models():
    flow_types = ['spherical', 'along_vector', 'proj', 'non_equivariant']
    seeds = [0]
    for flow_type in flow_types:
        for seed in seeds:
            download_checkpoint(flow_type=flow_type, tags=["dw4", "post_kigali_2"], seed=seed, max_iter=100,
                                base_path='./examples/dw4_results/models')


if __name__ == '__main__':
    download_dw4_models()
