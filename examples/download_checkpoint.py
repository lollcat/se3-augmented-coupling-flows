import re
import wandb
import os

api = wandb.Api()


def get_wandb_run(
        flow_type,
        tags,
        seed,
        donwload_latest_completed: bool = True):
    filter_list = [{"tags": tag} for tag in tags]
    filter_list.extend([
         {"config.flow": {"$regex": f"'type': '{flow_type}',"}},
         {"config.training": {"$regex": f"'seed': {seed},"}}
    ])
    filters = {"$and": filter_list}
    if 'fab' not in tags:
        filters.update({"$not": {"tags": 'fab'}})
    runs = api.runs(path='flow-ais-bootstrap/fab',
                    filters=filters)
    if len(runs) > 1:
        if not donwload_latest_completed:
            print(f"found {len(runs)} multiple runs, getting first")
            return runs[-1]
        print(f"Found {len(runs)} runs for flow_type {flow_type}, tags {tags}, seed {seed}. "
              f"Taking the most recent.")
    elif len(runs) == 0:
        raise Exception(f"No runs for for flow_type {flow_type}, tags {tags}, seed {seed} found!")
    j = 0
    while not "finished" in str(runs[j]):
        j += 1
    run = runs[j]  # Get latest finished run.
    # if 'fab' not in tags:
    #     assert '"fab": ' not in run.json_config
    # else:
    #     assert '"fab": ' in run.json_config
    return run


def download_checkpoint(flow_type, tags, seed, max_iter, base_path):
    assert os.path.exists(base_path)
    run = get_wandb_run(flow_type, tags, seed)
    for file in run.files():
        if re.search(fr'.*{max_iter-1}.pkl', str(file)):
            file.download(exist_ok=True)
            path = re.search(r"([^\s]*model_checkpoints[^\s]*)", str(file)).group()
            os.replace(path, f"{base_path}/{flow_type}_seed{seed}.pkl")
            print("saved" + path)


def get_run_history(flow_type, tags, seed,
                    fields=['marginal_log_lik', 'lower_bound_marginal_gap', '_timestamp', "_step"]):
    run = get_wandb_run(flow_type, tags, seed)
    history = run.history(keys=fields)
    return history



if __name__ == '__main__':
    # get_run_history(flow_type='spherical', tags=["dw4",  "florence"], seed=0)
    download_checkpoint(
        flow_type='spherical',
        tags=["dw4", "fab", "Leucadendron1"],
        seed=0,
        max_iter=20000,
        base_path='./examples/dw4_results/models'
    )

