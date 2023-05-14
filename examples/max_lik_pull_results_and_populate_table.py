import pandas as pd

from examples.get_wandb_runs import get_run_history


_TAGS = ['final_run']

def download_eval_metrics(problem="dw4", n_runs=3):
    flow_types = ['spherical', 'along_vector', 'proj', 'non_equivariant']
    seeds = [0, 1, 2, 3, 4]
    tags = _TAGS.copy()
    tags.append(problem)
    data = pd.DataFrame()

    i = 0
    for flow_type in flow_types:
        n_runs_found = 0
        for seed in seeds:
            try:
                hist = get_run_history(flow_type, tags, seed, fields=['marginal_log_lik', 'lower_bound_marginal_gap',
                                                                      'ess'] if problem in ["dw4", "lj13"] else
                ['marginal_log_lik', 'lower_bound_marginal_gap'])

                info = dict(hist.iloc[-1])
                if info["_step"] == 0:
                    print(f"skipping {flow_type} seed={seed} as it only has 1 step")
                    continue
                info.update(flow_type=flow_type, seed=seed)
                data = data.join(pd.Series(info, name=i), how="outer")
                i += 1
                n_runs_found += 1
                if n_runs_found == n_runs:
                    break
            except:
                pass
                # print(f"No runs for for flow_type {flow_type}, tags {tags} seed {seed} found!")

        if n_runs_found != n_runs:
            print(f"Less than {n_runs} runs found for flow {flow_type}")
    return data.T


def create_latex_table():
    flow_types = ['non_equivariant', 'along_vector', 'proj', 'spherical']
    row_names = ["ANF + data augmentation", "Vector-proj E-ANF", "Cartesian-proj E-ANF", "Spherical-proj E-ANF"]
    keys = ['marginal_log_lik', 'lower_bound_marginal_gap', 'ess']

    data_qm9 = download_eval_metrics("qm9pos")
    data_dw4 = download_eval_metrics("dw4")
    data_lj13 = download_eval_metrics("lj13")


    means_dw4 = data_dw4.groupby("flow_type").mean()
    sem_dw4 = data_dw4.groupby("flow_type").sem(ddof=0)

    means_lj13 = data_lj13.groupby("flow_type").mean()
    sem_lj13 = data_lj13.groupby("flow_type").sem(ddof=0)

    means_qm9 = data_qm9.groupby("flow_type").mean()
    sem_qm9 = data_qm9.groupby("flow_type").sem(ddof=0)

    table_values_string = ""
    table_ess = ""
    table_lower_bound_gap = ""
    for i, flow_type in enumerate(flow_types):
        table_values_string += \
            f"{row_names[i]} & " \
            f"{-means_dw4.loc[flow_type]['marginal_log_lik']:.2f},{sem_dw4.loc[flow_type]['marginal_log_lik']:.2f} & " \
            f"{-means_lj13.loc[flow_type]['marginal_log_lik']:.2f},{sem_lj13.loc[flow_type]['marginal_log_lik']:.2f} & " \
            f"{-means_qm9.loc[flow_type]['marginal_log_lik']:.2f},{sem_qm9.loc[flow_type]['marginal_log_lik']:.2f} \\\ \n "


        table_lower_bound_gap += \
            f"{row_names[i]} & " \
            f"{means_dw4.loc[flow_type]['lower_bound_marginal_gap']:.2f},{sem_dw4.loc[flow_type]['lower_bound_marginal_gap']:.2f} & " \
            f"{means_lj13.loc[flow_type]['lower_bound_marginal_gap']:.2f},{sem_lj13.loc[flow_type]['lower_bound_marginal_gap']:.2f} & " \
            f"{means_qm9.loc[flow_type]['lower_bound_marginal_gap']:.2f},{sem_qm9.loc[flow_type]['lower_bound_marginal_gap']:.2f} \\\ \n "


        table_ess += \
            f"{row_names[i]} & " \
            f"{means_dw4.loc[flow_type]['ess']*100:.2f},{sem_dw4.loc[flow_type]['ess']*100:.2f} & " \
            f"{means_lj13.loc[flow_type]['ess']*100:.2f},{sem_lj13.loc[flow_type]['ess']*100:.2f} \\\ \n "

    print(table_values_string)
    print("\n\n")
    print(table_lower_bound_gap)
    print("\n\n")
    print(table_ess)


if __name__ == '__main__':
    # data = download_eval_metrics_dw4()
    create_latex_table()
