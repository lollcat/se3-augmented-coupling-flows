import pandas as pd

from examples.analyse_results.get_wandb_runs import get_run_history


_TAGS_DW4_run = ['post_sub1','cblgpu', 'fab', "dw4"]
_TAGS_DW4_eval = ["dw4_fab", "evaluation", "eval_3", "with_fwd_ess"]
_TAGS_Lj13_eval = ["lj13_fab", "evaluation", "eval_3", "with_fwd_ess"]
_TAGS_LJ13_run = ['post_sub', 'cblgpu', 'fab', "lj13"]

def download_eval_metrics(tags,
                          n_runs=3,
                          flow_types=('spherical', 'along_vector', 'proj', 'non_equivariant'),
                          step_number=-1,
                          allow_single_step: bool = True,
                          fields=('marginal_log_lik', 'lower_bound_marginal_gap',
                                                                      'eval_ess_flow', 'eval_ess_ais',
                                                                      '_runtime',
                                                                      "_step")):
    fields = list(fields)
    seeds = [0, 1, 2]
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
                hist = get_run_history(flow_type, tags, seed, fields=fields)
                if iter_n == -1:
                    info = dict(hist.iloc[iter_n])
                else:
                    bounds = (hist["_step"] <= iter_n + 200) & (hist["_step"] >= iter_n - 200)
                    info = dict(hist[bounds].iloc[-1])
                if info["_step"] == 0 and not allow_single_step:
                    print(f"skipping {flow_type} seed={seed} as it only has 1 step")
                    continue
                info.update(flow_type=flow_type, seed=seed)
                data = data.join(pd.Series(info, name=i), how="outer")
                i += 1
                n_runs_found += 1
                if n_runs_found == n_runs:
                    break
            except:
                print(f"No runs for for flow_type {flow_type}, tags {tags} seed {seed} found!")

        if n_runs_found != n_runs:
            print(f"Less than {n_runs} runs found for flow {flow_type} tags {tags}")
    return data.T


def create_latex_table():
    flow_types = ['non_equivariant', 'along_vector', 'proj', 'spherical'] #
    row_names = ['\\' + "noneanf", "\\vecproj \ \eanf", "\\cartproj \ \eanf",
                 "\\sphproj \ \eanf"]
    keys = ['forward_ess', 'eval_ess_flow', 'eval_ess_ais', 'marginal_log_lik', 'lower_bound_marginal_gap', '_runtime']

    data_dw4 = download_eval_metrics(_TAGS_DW4_eval, flow_types=flow_types, step_number=-1,
                fields = ('marginal_log_lik', 'lower_bound_marginal_gap',
                          'eval_ess_flow', 'eval_ess_ais', 'forward_ess',
                          "_step"))
    dw4_runtime = download_eval_metrics(_TAGS_DW4_run, flow_types=flow_types,
                                        step_number=[64037, -1, -1, -1],
                                         fields=('_runtime', "_step"))
    data_dw4["_runtime"] = dw4_runtime["_runtime"]

    data_lj13 = download_eval_metrics(_TAGS_Lj13_eval, flow_types=flow_types, step_number=-1,
                fields = ('marginal_log_lik', 'lower_bound_marginal_gap',
                          'eval_ess_flow', 'eval_ess_ais', 'forward_ess',
                          "_step"))
    lj13_runtime = download_eval_metrics(_TAGS_LJ13_run, flow_types=flow_types, step_number=-1,
                                         fields=('_runtime',"_step"))
    data_lj13["_runtime"] = lj13_runtime["_runtime"]


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
            f"{means_dw4.loc[flow_type]['forward_ess'] * 100:.2f},{sem_dw4.loc[flow_type]['forward_ess'] * 100:.2f} & " \
            f"{-means_dw4.loc[flow_type]['marginal_log_lik']:.2f},{sem_dw4.loc[flow_type]['marginal_log_lik']:.2f} & " \
            f"{means_lj13.loc[flow_type]['eval_ess_flow'] * 100:.2f},{sem_lj13.loc[flow_type]['eval_ess_flow'] * 100:.2f} & " \
            f"{means_lj13.loc[flow_type]['forward_ess'] * 100:.2f},{sem_lj13.loc[flow_type]['forward_ess'] * 100:.2f} & " \
            f"{-means_lj13.loc[flow_type]['marginal_log_lik']:.2f},{sem_lj13.loc[flow_type]['marginal_log_lik']:.2f} \\\ \n"
            # f"0,0 & 0,0 & 0,0  \\\ \n"

        ess_table_string += f"{row_names[i]} & " \
            f"{means_dw4.loc[flow_type]['eval_ess_ais'] * 100:.2f},{sem_dw4.loc[flow_type]['eval_ess_ais'] * 100:.2f}  \\\ \n"


        runtime_table_string += \
            f"{row_names[i]} & " \
            f"{means_dw4.loc[flow_type]['_runtime']/3600:.1f},{sem_dw4.loc[flow_type]['_runtime']/3600:.1f} & " \
            f"{means_lj13.loc[flow_type]['_runtime']/3600:.1f},{sem_lj13.loc[flow_type]['_runtime']/3600:.1f} \\\ \n "


        # table_lower_bound_gap += \
        #     f"{row_names[i]} & " \
        #     f"{means_dw4.loc[flow_type]['lower_bound_marginal_gap']:.2f},{sem_dw4.loc[flow_type]['lower_bound_marginal_gap']:.2f} & " \
        #     f"{-means_lj13.loc[flow_type]['lower_bound_marginal_gap']:.2f},{sem_lj13.loc[flow_type]['lower_bound_marginal_gap']:.2f} \\\ \n "
    #                 f"0,0 \\\ \n"

    print("main table results")
    print(table_v2_string)
    print("\n\n")
    print("appendix ess table results")
    print(ess_table_string)
    # print(table_lower_bound_gap)
    print("\n\n")
    print("runtime table")
    print(runtime_table_string)


if __name__ == '__main__':
    # data = download_eval_metrics_dw4()
    create_latex_table()
