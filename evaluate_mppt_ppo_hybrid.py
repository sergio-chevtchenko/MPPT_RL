from controllers.PPO.ppo3 import train as train_ppo
from envs.mppt_grid_rl.mppt_grid_rl_train_v0 import MpptEnvGrid_V0
from envs.mppt_grid_rl.mppt_grid_rl_tests_v0 import MpptEnvGridTest_V0
from stable_baselines3 import PPO
import numpy as np


def test(model_name, params, discrete_actions=None, discrete_bins=1000, panels_series=6, panels_parallel=1):

    states_mask = [
        params['V'],  # V
        params['I'],  # I
        params['P'],  # P
        params['dV'],  # dV
        params['dI'],  # dI
        params['dP'],  # dP
        params['dP/dV'],  # dP/dV
        params['d(dP/dV)'],  # d(dP/dV)
        params['T']  # T
    ]

    '''
        Test 1  
                Initial random irradiance and temperature; 
                Duration: 10000 steps; 
                Random change of irradiance every 1000 steps;
                
        Test 2
                Initial random irradiance; 
                Duration: 10000 steps;
                Random change of irradiance every 1000 steps;
                Periodic change of temperature following a bell curve, centered at 500 steps;
        
        Test 3
                Initial random irradiance; 
                Duration: 1000 steps;                
                Continuous change of temperature following a bell curve, centered at 500 steps
                Continuous change of irradiance following a bell curve with random center
    '''

    min_d = 0.01

    delta_v_1 = 1
    delta_v_2 = delta_v_1 / 3

    test_results = {}
    for test_name in ['Test 1', 'Test 2']:
    #for test_name in ['Test 3']:
        print(test_name)
        test_results[test_name] = {}

        norm_pw_trace_avg = []
        pw_stability_trace_avg = []

        for run in range(30):
            print(run)
            env = MpptEnvGridTest_V0(alg_name='PPO hybrid', test_name=test_name, states_mask=states_mask,
                                     discrete_bins=discrete_bins,
                                     panels_series=panels_series, panels_parallel=panels_parallel,
                                     discrete_actions=discrete_actions)

            model = PPO.load(model_name)

            norm_pw_trace = []
            pw_stability_trace = []

            for trial in range(1):
                obs = env.reset()

                Vk_ = 0
                Ik_ = 0
                Pk_ = 0

                action = 1

                use_ppo = True

                steps_ppo = 0
                steps_ic = 0

                last_pw = 0
                step = 0
                done = False

                max_P = 0
                max_V = 0

                while not done:
                    if use_ppo:
                        steps_ppo += 1

                        if steps_ppo == 20:
                            steps_ppo = 0
                            use_ppo = False

                        action_ppo, _states = model.predict(obs)

                        obs, rewards, done, info = env.step(action_ppo, render=False)
                    else:
                        steps_ic += 1

                        if steps_ic == 20:
                            steps_ic = 0
                            use_ppo = True

                        obs, reward, done, info = env.step(action, render=False, analog_action=True)

                        Vk = info['V']
                        Ik = info['I']
                        Pk = info['P']

                        dV = Vk - Vk_
                        dI = Ik - Ik_
                        dP = Pk - Pk_

                        Gk = (Ik / (Vk + 0.0001)) + (dI / (dV + 0.0001))
                        Sk = np.abs(dP / (dV + 0.0001)) / (Ik + 0.0001)

                        if np.abs(dV) < min_d:
                            # dV == 0
                            if np.abs(dI) < min_d:
                                # dI == 0
                                action = 0
                            else:
                                if dI > 0:
                                    action = delta_v_1
                                else:
                                    action = -delta_v_2
                        else:
                            if np.abs(Gk) < min_d:
                                # dG == 0
                                action = 0
                            else:
                                if Gk > 0:
                                    action = Sk * delta_v_1
                                else:
                                    if Sk < 1:
                                        action = - Sk * delta_v_2
                                    else:
                                        action = - delta_v_2

                        Vk_ = Vk
                        Ik_ = Ik
                        Pk_ = Pk
                        action_ = action

                    current_pw = info['P']
                    norm_pw = current_pw / info['MaxP']
                    delta_p = np.abs(current_pw - last_pw)
                    last_pw = current_pw

                    if step > 0:
                        norm_pw_trace.append(norm_pw)
                        pw_stability_trace.append(delta_p)

                    step += 1

                # env.render()

            norm_pw_trace_avg.append(np.average(norm_pw_trace))
            pw_stability_trace_avg.append(np.average(pw_stability_trace))

        test_results[test_name]['norm_pw_avg'] = np.average(norm_pw_trace_avg)
        test_results[test_name]['norm_pw_std'] = np.std(norm_pw_trace_avg)
        test_results[test_name]['stability_avg'] = np.average(pw_stability_trace_avg)
        test_results[test_name]['stability_std'] = np.std(pw_stability_trace_avg)

        print(test_results)

    return test_results


if __name__ == '__main__':
    panels_series = 3
    panels_parallel = 1

    # # 94.84854343402151 0.20117585967241447 94.5241821032075 95.26602819464611 38
    # params = {'I': 1, 'P': 0, 'T': 0, 'V': 1, 'batch_size': 64, 'clip_range': 0.1, 'd(dP/dV)': 0,
    #           'dI': 1, 'dP': 0, 'dP/dV': 0, 'dV': 1, 'gamma': 0.5, 'hidden_neurons': 16,
    #           'learning_rate': 0.001, 'n_epochs': 5, 'n_steps': 4096, 'num_layers': 2, 'reward': 'distance'}

    # 94.86332848196362 0.2664216347420803 94.31537604293855 95.30364706189461 9
    params = {'I': 1, 'P': 0, 'T': 1, 'V': 1, 'batch_size': 128, 'clip_range': 0.1, 'd(dP/dV)': 0, 'dI': 1, 'dP': 1,
              'dP/dV': 0, 'dV': 1, 'gamma': 0.5, 'hidden_neurons': 64, 'learning_rate': 0.001, 'n_epochs': 10,
              'n_steps': 4096, 'num_layers': 2, 'reward': 'distance'}

    # discrete_actions = None
    discrete_actions = [-5, -1, 0, 1, 5]

    discrete_bins = 10

    random_id = np.random.randint(1000)

    #model_name = 'trained models/model_ppo_mppt_6_panels_None_1024_bins_923'
    model_name = 'model_ppo_mppt_3_panels_[-5, -1, 0, 1, 5]_10_bins_442'


    print('Testing model')
    test_results = test(model_name, params=params, discrete_actions=discrete_actions,
                        discrete_bins=discrete_bins, panels_series=panels_series, panels_parallel=panels_parallel)

    print('Test complete, results:')

    print(test_results)
