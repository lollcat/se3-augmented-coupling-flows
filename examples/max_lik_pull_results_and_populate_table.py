import pandas as pd

from examples.get_wandb_runs import get_run_history


_TAGS = ['post_kigali_4']

def download_eval_metrics(problem="dw4"):
    flow_types = ['spherical', 'along_vector', 'proj', 'non_equivariant']
    seeds = [0, 1, 2, 3, 4]
    tags = _TAGS.copy()
    tags.append(problem)
    data = pd.DataFrame()

    i = 0
    for flow_type in flow_types:
        for seed in seeds:
            hist = get_run_history(flow_type, tags, seed, fields=['marginal_log_lik', 'lower_bound_marginal_gap'])
            info = dict(hist.iloc[-1])
            if info["_step"] == 0:
                print(f"skipping {flow_type} seed={seed} as it only has 1 step")
                continue
            info.update(flow_type=flow_type, seed=seed)
            data = data.join(pd.Series(info, name=i), how="outer")

            i += 1
    return data.T


def create_latex_table():
    flow_types = ['non_equivariant', 'along_vector', 'proj', 'spherical']
    row_names = ["ANF + data augmentation", "Vector-proj E-ANF", "Cartesian-proj E-ANF", "Spherical-proj E-ANF"]
    keys = ['marginal_log_lik', 'lower_bound_marginal_gap']

    data_dw4 = download_eval_metrics("dw4")
    data_lj13 = download_eval_metrics("lj13")
    data_qm9 = download_eval_metrics("qm9pos")


    means_dw4 = data_dw4.groupby("flow_type")[keys].mean()
    sem_dw4 = data_dw4.groupby("flow_type")[keys].sem(ddof=0)

    means_lj13 = data_lj13.groupby("flow_type")[keys].mean()
    sem_lj13 = data_lj13.groupby("flow_type")[keys].sem(ddof=0)

    means_qm9 = data_qm9.groupby("flow_type")[keys].mean()
    sem_qm9 = data_qm9.groupby("flow_type")[keys].sem(ddof=0)

    table_values_string = ""
    for i, flow_type in enumerate(flow_types):
        table_values_string += \
            f"{row_names[i]} & " \
            f"{means_dw4.loc[flow_type]['marginal_log_lik']:.2f},{sem_dw4.loc[flow_type]['marginal_log_lik']:.2f} & " \
            f"{means_lj13.loc[flow_type]['marginal_log_lik']:.2f},{sem_lj13.loc[flow_type]['marginal_log_lik']:.2f} & " \
            f"{means_qm9.loc[flow_type]['marginal_log_lik']:.2f},{sem_qm9.loc[flow_type]['marginal_log_lik']:.2f} \ \n "

    print(table_values_string)


if __name__ == '__main__':
    # data = download_eval_metrics_dw4()
    create_latex_table()
