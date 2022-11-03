from controllers.PPO.ppo3 import train as train_ppo
from envs.mppt_grid_rl.mppt_grid_rl_train_v0 import MpptEnvGrid_V0
from envs.mppt_grid_rl.mppt_grid_rl_tests_v0 import MpptEnvGridTest_V0
import numpy as np

'''
Li, Chendi, et al. "A high-performance adaptive incremental conductance MPPT algorithm for photovoltaic systems." Energies 9.4 (2016): 288.
'''

def test(discrete_actions=None, discrete_bins=1000, panels_series=6, panels_parallel=1):

    states_mask = [
        1,  # V
        1,  # I
        1,  # P
        1,  # dV
        1,  # dI
        1,  # dP
        1,  # dP/dV
        1,  # d(dP/dV)
        1  # T
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
    delta_v_2 = delta_v_1/3

    test_results = {}
    for test_name in ['Test 1', 'Test 2', 'Test 3']:
    #for test_name in ['Test 3']:
        print(test_name)
        test_results[test_name] = {}

        norm_pw_trace_avg = []
        pw_stability_trace_avg = []

        for run in range(30):
            print(run)
            env = MpptEnvGridTest_V0(alg_name='Incremental conductance', test_name=test_name, states_mask=states_mask,
                                     discrete_bins=discrete_bins,
                                     panels_series=panels_series, panels_parallel=panels_parallel,
                                     discrete_actions=discrete_actions)

            norm_pw_trace = []
            pw_stability_trace = []

            for trial in range(1):
                obs = env.reset()

                Vk_ = 0
                Ik_ = 0
                Pk_ = 0

                action_ = 1
                action = 1

                last_pw = 0
                step = 0
                done = False
                while not done:
                    n_state, reward, done, info = env.step(action, render=False)

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

                #env.render()

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

    discrete_actions = None

    discrete_bins = 1024

    print('Testing incremental conductance')
    test_results = test(discrete_actions=discrete_actions,
                        discrete_bins=discrete_bins, panels_series=panels_series, panels_parallel=panels_parallel)

    print('Test complete, results:')

    print(test_results)
