import pandas as pd

from examples.analyse_results.get_wandb_runs import get_run_history


_TAGS = ['final_run', 'fab']

def download_eval_metrics(problem="dw4",
                          n_runs=3,
                          flow_types=('spherical', 'along_vector', 'proj', 'non_equivariant'),
                          step_number=-1):
    seeds = [0, 1, 2, 3, 4]
    tags = _TAGS.copy()
    tags.append(problem)
    data = pd.DataFrame()

    i = 0
    for j, flow_type in enumerate(flow_types):
        n_runs_found = 0
        if isinstance(step_number, list):
            iter_n = step_number[j]
        else:
            iter_n = step_number
        for seed in seeds:
            try:
                hist = get_run_history(flow_type, tags, seed, fields=['marginal_log_lik', 'lower_bound_marginal_gap',
                                                                      'eval_ess_flow', 'eval_ess_ais', '_runtime',
                                                                      "_step"])
                info = dict(hist.iloc[iter_n])
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
    step_numbers_dw4 = [2,3,3,3]
    step_numbers_lj13 = [10,6,6,6]
    flow_types = ['non_equivariant', 'along_vector', 'proj', 'spherical'] #
    row_names = ['\\' + "noneanf", "\\vecproj \ \eanf", "\\cartproj \ \eanf",
                 "\\sphproj \ \eanf"]
    keys = ['eval_ess_flow', 'eval_ess_ais', 'marginal_log_lik', 'lower_bound_marginal_gap', 'runtime']

    data_dw4 = download_eval_metrics("dw4", flow_types=flow_types, step_number=step_numbers_dw4)
    data_lj13 = download_eval_metrics("lj13", flow_types=flow_types, step_number=step_numbers_lj13)




    means_dw4 = data_dw4.groupby("flow_type")[keys].mean()
    sem_dw4 = data_dw4.groupby("flow_type")[keys].sem(ddof=0)

    means_lj13 = data_lj13.groupby("flow_type")[keys].mean()
    sem_lj13 = data_lj13.groupby("flow_type")[keys].sem(ddof=0)


    table_values_string = ""
    table_v2_string = ""
    table_lower_bound_gap = ""
    ess_table_string = ""
    runtime_table_string = ""

    for i, flow_type in enumerate(flow_types):
        # if i == 0:
        #     table_values_string += f"{row_names[i]} & " \
        #     f"0,0 & 0,0 & 0,0 &" \
        #     f"0,0 & 0,0 & 0,0  \\\ \n"
        #     continue
        table_values_string += \
            f"{row_names[i]} & " \
            f"{means_dw4.loc[flow_type]['eval_ess_flow']*100:.2f},{sem_dw4.loc[flow_type]['eval_ess_flow']*100:.2f} & " \
            f"{means_dw4.loc[flow_type]['eval_ess_ais'] * 100:.2f},{sem_dw4.loc[flow_type]['eval_ess_ais'] * 100:.2f} & " \
            f"{-means_dw4.loc[flow_type]['marginal_log_lik']:.2f},{sem_dw4.loc[flow_type]['marginal_log_lik']:.2f} & " \
            f"{means_lj13.loc[flow_type]['eval_ess_flow'] * 100:.2f},{sem_lj13.loc[flow_type]['eval_ess_flow'] * 100:.2f} & " \
            f"{means_lj13.loc[flow_type]['eval_ess_ais'] * 100:.2f},{sem_lj13.loc[flow_type]['eval_ess_ais'] * 100:.2f} & " \
            f"{-means_lj13.loc[flow_type]['marginal_log_lik']:.2f},{sem_lj13.loc[flow_type]['marginal_log_lik']:.2f} \\\ \n"
            # f"0,0 & 0,0 & 0,0  \\\ \n"

        table_v2_string += \
            f"{row_names[i]} & " \
            f"{means_dw4.loc[flow_type]['eval_ess_flow']*100:.2f},{sem_dw4.loc[flow_type]['eval_ess_flow']*100:.2f} & " \
            f"{-means_dw4.loc[flow_type]['marginal_log_lik']:.2f},{sem_dw4.loc[flow_type]['marginal_log_lik']:.2f} & " \
            f"{means_lj13.loc[flow_type]['eval_ess_flow'] * 100:.2f},{sem_lj13.loc[flow_type]['eval_ess_flow'] * 100:.2f} & " \
            f"{-means_lj13.loc[flow_type]['marginal_log_lik']:.2f},{sem_lj13.loc[flow_type]['marginal_log_lik']:.2f} \\\ \n"
            # f"0,0 & 0,0 & 0,0  \\\ \n"


        runtime_table_string += \
            f"{row_names[i]} & " \
            f"{means_dw4.loc[flow_type]['_runtime']/3600:.1f},{sem_dw4.loc[flow_type]['_runtime']/3600:.1f} & " \
            f"{means_lj13.loc[flow_type]['_runtime']/3600:.1f},{sem_lj13.loc[flow_type]['_runtime']/3600:.1f} \\\ \n "


        table_lower_bound_gap += \
            f"{row_names[i]} & " \
            f"{means_dw4.loc[flow_type]['lower_bound_marginal_gap']:.2f},{sem_dw4.loc[flow_type]['lower_bound_marginal_gap']:.2f} & " \
            f"{-means_lj13.loc[flow_type]['lower_bound_marginal_gap']:.2f},{sem_lj13.loc[flow_type]['lower_bound_marginal_gap']:.2f} \\\ \n "
    #                 f"0,0 \\\ \n"

    # print(table_values_string)
    # print("\n\n")
    # print(table_lower_bound_gap)
    # print("\n\n")
    print(runtime_table_string)


if __name__ == '__main__':
    # data = download_eval_metrics_dw4()
    create_latex_table()
