from pathlib import Path

################################### Configs #####################################

EXPERIMENT_NAME = 'fab'  # no underscores.
JOBS_FOLDER = f'trial-{EXPERIMENT_NAME}'
DELETE_PREV_FOLDER = True
SCRIPT_NAMES = ['lj13']  #  ['dw4' , 'lj13']
FLOW_TYPES=['spherical'] # ['spherical', 'along_vector', 'proj','non_equivariant']
SCRIPT_FOLDER = 'examples/' #  f'home/lim24/rds/hpc-work/lim24/augmented-equivariant-flows/examples/'
RANDOM_SEEDS = [0,]
N_UPDATES_PER_SMC_FORWARD_PASS = [8, 16, 32]
LR_s = [4e-4, 2e-4, 1e-4]
N_LAYERS=[12, 16]
BATCH_SIZES=[64,128]
################################################################################

times = {
    'dw4': '12:00:00',
    'lj13': '32:00:00',
}

jobsfolder = Path(f'./{JOBS_FOLDER}')
if jobsfolder.exists() and DELETE_PREV_FOLDER:
    for jobfile in jobsfolder.glob('*'):
        jobfile.unlink()
    jobsfolder.rmdir()
jobsfolder.mkdir(exist_ok=True)


for script in SCRIPT_NAMES:
    jobsfile = jobsfolder / f'{EXPERIMENT_NAME}-{script}_{times[script]}.txt'
    if jobsfile.exists():
        jobsfile.unlink()
    jobsfile.touch()

    with open(jobsfile, 'w') as f:
        for flow_type in FLOW_TYPES:
            for smc_updates_per_smc_forward_pass in N_UPDATES_PER_SMC_FORWARD_PASS:
                for lr in LR_s:
                    for n_layers in N_LAYERS:
                        for batch_size in BATCH_SIZES:
                            for random_seed in RANDOM_SEEDS:
                                line = (f'{SCRIPT_FOLDER + script}_fab.py '
                                        f'flow.type={flow_type} '
                                        f'flow.n_layers={n_layers} '
                                        f'fab.n_updates_per_smc_forward_pass={smc_updates_per_smc_forward_pass} '
                                        f'training.seed={random_seed} '
                                        f'training.optimizer.init_lr={lr} '
                                        f'training.batch_size={batch_size} '
                                        )
                                f.write(line + '\n')
