import re
import wandb
import os

api = wandb.Api()


def get_wandb_run(flow_type, tags, seed):
    filter_list = [{"tags": tag} for tag in tags]
    filter_list.extend([
         {"config.flow": {"$regex": f"'type': '{flow_type}',"}},
         {"config.training": {"$regex": f"'seed': {seed},"}}
    ])
    filters = {"$and": filter_list}
    runs = api.runs(path='flow-ais-bootstrap/fab',
                    filters=filters)
    # assert len(runs) == 1  # Should only have 1 match.

    return runs[0]  # Get latest run.
    # runs_list = []
    # i = 0
    # key = 'marginal_log_lik'
    # while not len(runs_list) == 1:
    #     if i >= (len(runs)):
    #         print(f"not enough seeds:")
    #         print(f"only {len(runs_list)} seeds \n")
    #         break
    #     run = runs[i]
    #     history = run.history(keys=[key])
    #     if "finished" not in str(run) or key not in history.keys():
    #         i += 1
    #         continue
    #     runs_list.append(run)
    #     i += 1
    return runs


def download_checkpoint(flow_type, tags, seed, max_iter, base_path):
    run = get_wandb_run(flow_type, tags, seed)
    for file in run.files():
        if re.search(fr'.*{max_iter-1}.pkl', str(file)):
            file.download(exist_ok=True)
            path = re.search(r"([^\s]*results[^\s]*)", str(file)).group()
            os.replace(path, f"{base_path}/{flow_type}_seed{seed}.pkl")
            print("saved" + path)


if __name__ == '__main__':
    download_checkpoint(flow_type='spherical', tags=["lj13", "post_kigali_1"], seed=0, max_iter=256,
                        base_path='./examples/lj13_results/models')


