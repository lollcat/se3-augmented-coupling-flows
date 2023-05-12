from examples.get_wandb_runs import get_wandb_run


def download_eval_metrics():
    flow_types = ['spherical', 'along_vector', 'proj', 'non_equivariant']
    seeds = [0]
    tags = ['post_kigali_4', 'dw4']
    for flow_type in flow_types:
        for seed in seeds:
            run = get_wandb_run(flow_type, tags, seed)
