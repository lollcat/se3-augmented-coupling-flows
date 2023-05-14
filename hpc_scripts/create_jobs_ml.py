
from pathlib import Path
from hashlib import md5

################################### Configs #####################################

EXPERIMENT_NAME = 'ml'  # no underscores.
JOBS_FOLDER = f'trial-{EXPERIMENT_NAME}'
DELETE_PREV_FOLDER = True
SCRIPT_NAMES = ['dw4', 'lj13'] # , 'qm9']
FLOW_TYPES=['spherical', 'along_vector', 'proj','non_equivariant'] # ['spherical', 'along_vector', 'proj','non_equivariant']
SCRIPT_FOLDER = 'examples/' #  f'home/lim24/rds/hpc-work/lim24/augmented-equivariant-flows/examples/'
RANDOM_SEEDS = [0,1,2,3,4]
DS_SIZES = {'dw4': [100, 1000, 10000, 100000],
            'lj13': [10, 100, 1000, 10000],
            'qm9': [None]
            }

# NOTE: if you add configs you probably want to specify the results_folder more down below

################################################################################

times = {
    'dw4': '32:00:00',
    'lj13': '32:00:00',
    'qm9': '32:00:00',
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
            problem_ds_sizes = DS_SIZES[script]
            for random_seed in RANDOM_SEEDS:
                for n_samples in problem_ds_sizes:
                    line = (f'{SCRIPT_FOLDER + script}.py '
                                f'flow.type={flow_type} '
                            f'training.seed={random_seed} '
                            f'training.train_set_size={n_samples} '
                            )
                    f.write(line + '\n')

