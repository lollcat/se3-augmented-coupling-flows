from pathlib import Path
from hashlib import md5

################################### Configs #####################################

EXPERIMENT_NAME = 'fab'  # no underscores.
JOBS_FOLDER = f'trial-{EXPERIMENT_NAME}'
DELETE_PREV_FOLDER = True
SCRIPT_NAMES = ['dw4','lj13'] # , 'lj13', 'qm9']
FLOW_TYPES=['non_equivariant'] # ['spherical', 'along_vector', 'proj','non_equivariant']
SCRIPT_FOLDER = 'examples/' #  f'home/lim24/rds/hpc-work/lim24/augmented-equivariant-flows/examples/'
RANDOM_SEEDS = [0,]

################################################################################

times = {
    'dw4': '12:00:00',
    'lj13': '23:00:00',
    'qm9': '24:00:00',
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
            for random_seed in RANDOM_SEEDS:
                line = (f'{SCRIPT_FOLDER + script}_fab.py '
                        f'flow.type={flow_type} '
                        f'training.seed={random_seed} '
                        )
                f.write(line + '\n')