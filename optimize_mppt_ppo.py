import logging
import sys
import optuna
from controllers.PPO.ppo3 import train
from envs.mppt_grid_rl.mppt_grid_rl_train_v0 import MpptEnvGrid_V0
import numpy as np
import matplotlib.pyplot as plt

# work_dir = '/content/gdrive/My Drive/MPPT/'
work_dir = ''

# log_file_name = 'optimize_mppt_ppo_states_V1.csv'
log_file_name = 'optimize_mppt_ppo_hyper_V0.csv'
text_file = open(work_dir + log_file_name, "a")
text_file.close()


class SortedDisplayDict(dict):
    def __str__(self):
        return "{" + ", ".join("%r: %r" % (key, self[key]) for key in sorted(self)) + "}"


def evaluation_trial(trial):
    #  'V', 'P', 'dP', 'dV', 'dPdV', 'd(dPdV)', 'T', 'P1', 'P2', 'P3'
    # states_mask = [
    #     1,
    #     trial.suggest_categorical('P', [0, 1]),
    #     trial.suggest_categorical('dP', [0, 1]),
    #     trial.suggest_categorical('dV', [0, 1]),
    #     trial.suggest_categorical('dPdV', [0, 1]),
    #     trial.suggest_categorical('d(dPdV)', [0, 1]),
    #     1,
    #     trial.suggest_categorical('P1', [0, 1]),
    #     trial.suggest_categorical('P2', [0, 1]),
    #     trial.suggest_categorical('P3', [0, 1])
    # ]

    # all_panels = trial.suggest_categorical('all_panels', [0, 1])
    # if all_panels == 1:
    #   states_mask[7] = 1
    #   states_mask[8] = 1
    #   states_mask[9] = 1
    # else:
    #   states_mask[7] = trial.suggest_categorical('P1', [0, 1])
    #   states_mask[8] = trial.suggest_categorical('P2', [0, 1])
    #   states_mask[9] = trial.suggest_categorical('P3', [0, 1])

    # if np.sum(states_mask) == 0:
    #     return 0

    states_mask = [
        trial.suggest_categorical('V', [0, 1]),  # V
        trial.suggest_categorical('I', [0, 1]),  # I
        trial.suggest_categorical('P', [0, 1]),  # P
        trial.suggest_categorical('dV', [0, 1]),  # dV
        trial.suggest_categorical('dI', [0, 1]),  # dI
        trial.suggest_categorical('dP', [0, 1]),  # dP
        trial.suggest_categorical('dP/dV', [0, 1]),  # dP/dV
        trial.suggest_categorical('d(dP/dV)', [0, 1]),  # d(dP/dV)
        trial.suggest_categorical('T', [0, 1])  # T
    ]

    if np.sum(states_mask) == 0:
        return 0

    env = MpptEnvGrid_V0(alg_name='PPO', states_mask=states_mask, discrete_bins=1000,
                         panels_series=6,
                         panels_parallel=1,
                         discrete_actions=None,
                         reward_function=trial.suggest_categorical('reward', ['power', 'distance']), plot_trace=False)

    print(env.observation_space)
    print(env.action_space)

    hyperparams = dict(batch_size=trial.suggest_categorical('batch_size', [64, 128, 256]),
                       clip_range=trial.suggest_categorical('clip_range', [0.1, 0.2, 0.4]),
                       gamma=trial.suggest_categorical('gamma', [0.5, 0.9, 0.99]),
                       hidden_neurons=trial.suggest_categorical('hidden_neurons', [16, 32, 64, 128]),
                       learning_rate=trial.suggest_categorical('learning_rate', [1e-3, 1e-4, 1e-5]),
                       n_epochs=trial.suggest_categorical('n_epochs', [1, 5, 10]),
                       n_steps=trial.suggest_categorical('n_steps', [1024, 2048, 4096, 8192]),
                       num_layers=trial.suggest_categorical('num_layers', [1, 2, 3])
                       )

    train(hyperparams, env, total_episodes=10000, default_policy=False)

    # hyperparams = dict()
    # evaluate(hyperparams, env=env, total_episodes=5000, default_policy=True)

    avg_P = np.average(env.trace_P_avg[-3000:])

    text_file = open(work_dir + log_file_name, "a")
    text_file.write(str(avg_P)
                    + ',' + str(states_mask)
                    + ',' + str(SortedDisplayDict(hyperparams))
                    + '\n')
    text_file.close()

    return avg_P


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = log_file_name
storage_name = "sqlite:///{}.db".format(work_dir + study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize')

print('Trials:', len(study.trials))
if len(study.trials) > 1:
    print("Best Trial so far: ", study.best_trial)

study.optimize(evaluation_trial, n_trials=1000)

print("Best params: ", study.best_params)
print("Best value: ", study.best_value)
print("Best Trial: ", study.best_trial)
print("Trials: ", study.trials)
