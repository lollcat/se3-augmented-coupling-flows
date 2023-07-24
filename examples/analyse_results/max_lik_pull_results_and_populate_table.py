import pandas as pd

from examples.analyse_results.get_wandb_runs import get_run_history

def download_eval_metrics(problem, tags, n_runs=3):
    flow_types = ['spherical', 'along_vector', 'proj', 'non_equivariant']
    seeds = [0, 1, 2]
    data = pd.DataFrame()

    i = 0
    for flow_type in flow_types:
        n_runs_found = 0
        for seed in seeds:
            try:
                fields = ['marginal_log_lik', 'lower_bound_marginal_gap', 'ess', '_runtime', "_step"] \
                    if problem in ["dw4", "lj13"] else \
                    ['marginal_log_lik', 'lower_bound_marginal_gap', '_runtime', "_step"]
                hist = get_run_history(flow_type, tags, seed, fields=fields)

                info = dict(hist.iloc[-1])
                if info["_step"] == 0:
                    print(f"skipping {flow_type} seed={seed} as it only has 1 step")
                    continue
                info.update(flow_type=flow_type, seed=seed)
                data = data.join(pd.Series(info, name=i), how="outer")
                data.loc[fields] = data.loc[fields].astype("float32")
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
    row_names = ['\\' + "noneanf", "\\vecproj \ \eanf", "\\cartproj \ \eanf", "\\sphproj \ \eanf"]


    data_dw4 = download_eval_metrics("dw4", ["ml", "post_sub1", "cblgpu", "dw4"])
    data_lj13 = download_eval_metrics("lj13", ["ml", "post_sub1", "cblgpu", "lj13"])
    data_qm9 = download_eval_metrics("qm9pos", ["ml", "post_sub1", "cblgpu10", "layer_norm"])


    means_dw4 = data_dw4.groupby("flow_type").mean()
    sem_dw4 = data_dw4.groupby("flow_type").sem(ddof=0)

    means_lj13 = data_lj13.groupby("flow_type").mean()
    sem_lj13 = data_lj13.groupby("flow_type").sem(ddof=0)

    means_qm9 = data_qm9.groupby("flow_type").mean()
    sem_qm9 = data_qm9.groupby("flow_type").sem(ddof=0)

    table_values_string = ""
    table_ess = ""
    table_lower_bound_gap = ""
    table_runtimes = ""

    for i, flow_type in enumerate(flow_types):
        table_runtimes += \
            f"{row_names[i]} & " \
            f"{means_dw4.loc[flow_type]['_runtime']/3600:.2f},{sem_dw4.loc[flow_type]['_runtime']/3600:.2f} & " \
            f"{means_lj13.loc[flow_type]['_runtime']/3600:.2f},{sem_lj13.loc[flow_type]['_runtime']/3600:.2f} & " \
            f"{means_qm9.loc[flow_type]['_runtime']/3600:.2f},{sem_qm9.loc[flow_type]['_runtime']/3600:.2f} \\\ \n "


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

    print("************** main table ************** \n")
    print(table_values_string)
    print("\n\n")
    print("************** table lower bound gap ************** \n")
    print(table_lower_bound_gap)
    print("\n\n")
    print("************** ess ************** \n")
    print(table_ess)
    print("\n\n")
    print("************** runtimes ************** \n")
    print(table_runtimes)


if __name__ == '__main__':
    # data = download_eval_metrics_dw4()
    create_latex_table()
