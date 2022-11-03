from controllers.PPO.ppo3 import train as train_ppo
from envs.mppt_grid_rl.mppt_grid_rl_train_v0 import MpptEnvGrid_V0
from envs.mppt_grid_rl.mppt_grid_rl_tests_v0 import MpptEnvGridTest_V0
import numpy as np

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

    min_dV = 0.01
    min_dI = 0.01

    max_step = 1

    test_results = {}
    for test_name in ['Test 1', 'Test 2', 'Test 3']:
    #for test_name in ['Test 3']:
        print(test_name)
        test_results[test_name] = {}

        norm_pw_trace_avg = []
        pw_stability_trace_avg = []

        for run in range(30):
            print(run)
            env = MpptEnvGridTest_V0(alg_name='Hill climbing', test_name=test_name, states_mask=states_mask,
                                     discrete_bins=discrete_bins,
                                     panels_series=panels_series, panels_parallel=panels_parallel,
                                     discrete_actions=discrete_actions)

            norm_pw_trace = []
            pw_stability_trace = []

            for trial in range(1):
                obs = env.reset()

                Pk_ = 0

                action_ = 1
                action = 1

                last_pw = 0
                step = 0
                done = False
                while not done:
                    n_state, reward, done, info = env.step(action, render=False)

                    Pk = info['P']

                    dP = Pk - Pk_

                    if dP < 0:
                        action = -action_

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
    panels_series = 12
    panels_parallel = 1

    discrete_actions = None

    discrete_bins = 1024

    print('Testing hill climbing')
    test_results = test(discrete_actions=discrete_actions,
                        discrete_bins=discrete_bins, panels_series=panels_series, panels_parallel=panels_parallel)

    print('Test complete, results:')

    print(test_results)
