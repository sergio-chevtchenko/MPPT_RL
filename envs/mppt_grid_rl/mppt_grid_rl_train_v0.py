import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import random

import matplotlib.pyplot as plt


class Panel(object):

    def __init__(self):
        self.TK = 273  # Kelvin temperature
        self.Tr1 = 40  # Reference temperature in degree fahrenheit
        # self.S = 100  # Solar radiation in mW / sq.cm
        self.ki = 0.00023  # in A / K
        self.Iscr = 3.75  # SC Current at ref.temp. in A
        # self.Irr = 0.000021  # in A
        self.k = 1.38065e-23  # Boltzmann constant
        self.q = 1.6022e-19  # charge of an electron
        self.A = 2.15  # ideality factor
        self.Eg0 = 1.166  # band gap energy
        self.alpha = 0.473
        self.beta = 636
        # open-circuit voltage
        self.Voc = 21
        # panel composed of Np parallel modules each one including Ns photovoltaic cells connected
        self.Np = 1
        self.Ns = 36
        self.Pmax = self.Iscr * self.Voc # W

    def calc_Iph(self, G, T):
        # cell temperature in Kelvin
        Tcell = T + self.TK

        # cell reference temperature in kelvin
        Tr = self.Tr1 + self.TK

        # generated photocurrent
        Iph = (self.Iscr + self.ki * (Tcell - Tr)) * (G / 1000)

        return Iph

    def calc_pv(self, G, T, vx):
        if vx > self.Voc * 2:
            vx = self.Voc * 2

        # cell temperature in Kelvin
        Tcell = T + self.TK

        # cell reference temperature in kelvin
        Tr = self.Tr1 + self.TK

        # generated photocurrent
        Iph = self.calc_Iph(G, T)

        # cell reverse saturation current
        Irs = self.Iscr / (np.exp(self.q * self.Voc / (self.Ns * self.k * self.A * Tcell)) - 1)

        I01 = Irs * ((Tcell / Tr) ** 3) * np.exp((self.q * self.Eg0 / (self.k * self.A)) * ((1 / Tr) - (1 / Tcell)))

        Rs = 0.00001
        Rp = 100000
        Ipv0 = 0.1
        for _ in range(100):
            I = self.Np * (Iph - I01 * (np.exp(self.q * (vx + Ipv0 * Rs) / (self.Ns * self.A * self.k * Tcell)) - 1) -
                           (vx + Ipv0 * Rs * self.Ns) / (Rp * self.Ns))
            delta = np.abs(I - Ipv0)

            if delta < 0.0001:
                break

            Ipv0 = I

        if I < 0:
            I = 0

        # panel output voltage
        V = vx  # este es el Vg?
        # panel power
        P = vx * I

        return I, V, P


class ShadedArray(object):
    def __init__(self, G, T, V_min, V_max):
        self.v_curve, self.i_curve, self.p_curve = self.update_curve(G, T, V_min, V_max)
        self.pv_max = [np.amax(self.p_curve), self.v_curve[np.argmax(self.p_curve)]]

    def data(self, Vpa):
        abs_val = np.abs(self.v_curve - Vpa)
        min_index = abs_val.argmin()

        return self.i_curve[min_index], Vpa, self.p_curve[min_index]

    def update_curve(self, G, T, V_min, V_max):
        n_points = 1000

        i_curve = np.zeros(n_points)
        p_curve = np.zeros(n_points)
        v_curve = np.zeros(n_points)

        for G_series in G:
            # sort arrays by irradiance from highest to lowest
            order = np.argsort(-G_series)

            pv = Panel()

            # generated photocurrents
            Iph = []
            for g in G_series:
                Iph.append(pv.calc_Iph(g, T))

            i = 0

            Ipva = Iph[order[0]]

            for k, pv_voltage in enumerate(np.linspace(start=V_min, stop=V_max, num=n_points)):
                I, V, P = pv.calc_pv(G_series[order[i]], T, pv_voltage / (i + 1))

                if i < (len(G_series) - 1) and (I < Iph[order[min(i + 1, len(G_series) - 1)]]):
                    i += 1
                else:
                    Ipva = I

                i_curve[k] += Ipva
                v_curve[k] = pv_voltage

        for k, Ip in enumerate(i_curve):
            p_curve[k] = Ip * v_curve[k]

        return np.array(v_curve), np.array(i_curve), np.array(p_curve)


class MpptEnvGrid_V0(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, alg_name, states_mask, panels_series, panels_parallel,
                 discrete_bins=50, reward_function='distance',
                 plot_trace=False, discrete_actions=None):

        self.reward_function = reward_function

        assert self.reward_function in ['distance', 'power']

        self.plot_trace = plot_trace
        self.alg_name = alg_name
        self.reward_range = (-float('inf'), float('inf'))

        self.n_panels = panels_series*panels_parallel

        self.G = np.zeros((panels_parallel, panels_series)) # Solar radiation array in mW / sq.cm
        self.V_min = 1
        self.V_max = 21 * panels_series

        self.I_min = 0
        self.I_max = Panel().Iscr * panels_parallel

        self.P_min = 0
        self.P_max = Panel().Pmax * self.n_panels

        self.DeltaV_min = -5
        self.DeltaV_max = 5

        self.DeltaP_min = -self.P_max * 0.5
        self.DeltaP_max = self.P_max * 0.5

        self.DeltaI_min = -self.I_max
        self.DeltaI_max = self.I_max

        self.T_min = 0.
        self.T_max = 75.

        self.G_min = 100.
        self.G_max = 1000.

        # this is a normalized action
        # amplitude is defined above by self.DeltaV_max and self.DeltaV_min
        self.min_actionValue = -1.
        self.max_actionValue = 1.

        # normalized stated values
        self.max_stateValue = 1.
        self.min_stateValue = -1.

        self.discrete_bins = discrete_bins
        print('Using', discrete_bins, 'discrete state bins')
        print()

        self.state_names = [
            'V', 'I', 'P', 'dV', 'dI', 'dP', 'dP/dV', 'd(dP/dV)', 'T'
        ]

        assert len(states_mask) == len(self.state_names)

        print('Using the following states:')
        for i, s in enumerate(states_mask):
            if s == 1:
                print(self.state_names[i])
        print()

        self.full_st_min = [self.V_min, self.I_min, self.P_min, self.DeltaV_min, self.DeltaI_min, self.DeltaP_min,
                            self.DeltaP_min, self.DeltaP_min, self.T_min]
        self.full_st_max = [self.V_max, self.I_max, self.P_max, self.DeltaV_max, self.DeltaI_max, self.DeltaP_max,
                            self.DeltaP_max, self.DeltaP_max, self.T_max]

        self.states_mask = states_mask

        self.state_dim = np.sum(self.states_mask)
        self.action_dim = 1

        self.discrete_actions = discrete_actions

        if self.discrete_actions is None:
            print('Continuous control')
            print()
            self.action_space = spaces.Box(low=self.min_actionValue, high=self.max_actionValue,
                                           shape=(self.action_dim,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(self.discrete_actions))
            print('Discrete control:', self.discrete_actions)
            print()

        self.observation_space = spaces.Box(low=self.min_stateValue, high=self.max_stateValue,
                                            shape=(self.state_dim,), dtype=np.float32)

        self.Temp = 25.     # will be randomly changed on reset

        self.steps = 0
        self.MaxSteps = 100

        self.trace_rw = []
        self.trace_rw_episode = []
        self.trace_P_episode = []
        self.trace_dP_episode = []
        self.trace_V_episode = []
        self.trace_I_episode = []
        self.trace_dPdV_episode = []

        self.trace_V_max_dist = []
        self.trace_P_avg = []
        self.trace_P_max_std = []
        self.trace_P_max_avg = []
        self.trace_P_max_steps = []
        self.trace_P_max_steps_avg = []

        self.trace_analog_state = []
        self.trace_norm_state = []

        self.current_max_pv_point = [0, 0]
        self.current_p_curve = []
        self.current_i_curve = []

        self.sum_rw = 0

        self.G = np.random.randint(100, 1000, (panels_parallel, panels_series))
        self.pv_array = ShadedArray(self.G, self.Temp, self.V_min, self.V_max)

        self.episode_count = 0

        # while True:
        # #for self.Temp in range(0, 75, 5):
        #     #self.Temp = np.random.randint(0, 75)
        #
        #     #self.G = np.random.randint(100, 1000, n_panels)
        #     #self.G = np.random.choice([100, 250, 500, 750, 1000], self.n_panels)
        #     #self.G = np.random.randint(100, 1000, (panels_parallel, panels_series))
        #
        #     self.G = np.random.randint(900, 1000, (panels_parallel, panels_series))
        #
        #     self.Temp = 25
        #     #self.G = np.ones(n_panels) * 1000
        #     #self.G = np.array([500, 600, 700, 800, 900, 1000])
        #     #self.G = np.array([100, 101, 102, 103, 104, 105])
        #     #self.G = np.array([900, 910, 920, 930, 940, 950])
        #
        #     # possible_G = [[500, 600, 700, 800, 900, 1000],
        #     #     [100, 101, 102, 103, 104, 105],
        #     #     [900, 910, 920, 930, 940, 950],
        #     #     [920, 940, 960, 980, 1000, 1020],
        #     #     [900, 901, 902, 903, 904, 905]]
        #
        #     # possible_G = [
        #     #         [500, 600, 700, 800, 900, 1000],
        #     #         [900, 910, 920, 930, 940, 950],
        #     #         [920, 940, 960, 980, 1000, 1020],
        #     #         [900, 901, 902, 903, 904, 905]]
        #     #
        #     # self.G = np.array(possible_G[np.random.randint(len(possible_G))])
        #
        #     self.pv_array = ShadedArray(self.G, self.Temp, self.V_min, self.V_max)
        #
        #     self.current_max_pv_point, self.current_p_curve, self.current_i_curve = self.pv_array.pv_max, self.pv_array.p_curve, self.pv_array.i_curve
        #
        #     #plt.figure(200).clear()
        #     plt.figure(200)
        #
        #     plt.subplot(211)
        #     plt.plot(np.linspace(self.V_min, self.V_max, len(self.current_p_curve)), self.current_p_curve)
        #     plt.ylabel('Power')
        #     plt.plot(self.current_max_pv_point[1], self.current_max_pv_point[0], 'go')
        #
        #     plt.ylim([self.P_min, self.P_max])
        #
        #     plt.subplot(212)
        #     plt.plot(np.linspace(self.V_min, self.V_max, len(self.current_i_curve)), self.current_i_curve)
        #     plt.ylabel('Current')
        #     plt.xlabel('Voltage')
        #
        #     plt.pause(0.00001)

    def reset(self):
        if len(self.trace_P_episode) > 0:
            avg_p_episode = np.average(self.trace_P_episode[:])
            max_v_dist = np.abs(self.current_max_pv_point[1]
                                - np.average(self.trace_V_episode[-10:]))

            max_steps = np.argmax(self.trace_P_episode)

            if len(self.trace_rw) % 100 == 0 and self.plot_trace:
                print('Episode', len(self.trace_rw))
                print('Last 100 episodes % Pmax average (std): {:>.1f} ({:>.1f}), min-max: [{:>.1f}-{:>.1f}]'.format(
                    self.trace_P_max_avg[-1], self.trace_P_max_std[-1], np.amin(self.trace_P_avg[-100:]),
                    np.amax(self.trace_P_avg[-100:])))

                print('Last 100 episodes % Vmax average: {:>.1f}, min-max: [{:>.1f}-{:>.1f}]'.format(
                    np.average(self.trace_V_max_dist[-100:]), np.amin(self.trace_V_max_dist[-100:]),
                    np.amax(self.trace_V_max_dist[-100:])))

                np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
                print('Last 100 episodes analog state stats:')
                print('avg:', np.average(self.trace_analog_state[-10000:], axis=0))
                print('std:', np.std(self.trace_analog_state[-10000:], axis=0))
                print('min:', np.amin(self.trace_analog_state[-10000:], axis=0))
                print('max:', np.amax(self.trace_analog_state[-10000:], axis=0))
                print('set size (analog):', np.unique(self.trace_analog_state[-10000:], axis=0).shape)
                print('set size (normalized):', np.unique(self.trace_norm_state[-10000:], axis=0).shape)

                print()

            # print(np.average(self.trace_dP))
            # print(np.std(self.trace_dP))
            # print(np.amin(self.trace_dP))
            # print(np.amax(self.trace_dP))
            # print()

        else:
            avg_p_episode = 0
            max_steps = 0
            max_v_dist = 0

        self.trace_P_max_steps.append(max_steps)
        self.trace_P_max_steps_avg.append(np.average(self.trace_P_max_steps[-100:]))

        while isinstance(max_v_dist, (list, tuple, np.ndarray)):
            max_v_dist = max_v_dist[0]

        self.trace_V_max_dist.append(max_v_dist)

        new_p = 100 * avg_p_episode / (self.current_max_pv_point[0] + 0.00001)
        while isinstance(new_p, (list, tuple, np.ndarray)):
            new_p = new_p[0]
        self.trace_P_avg.append(new_p)
        self.trace_P_max_avg.append(np.average(self.trace_P_avg[-100:]))
        self.trace_P_max_std.append(np.std(self.trace_P_avg[-100:]))

        self.trace_rw.append(self.sum_rw)

        if (self.episode_count % 10 == 0) and self.plot_trace:
            # plt.clf()
            plt.figure(self.alg_name + ' trace').clear()

            # plt.subplot(121)
            #
            # plt.plot(self.trace_rw)
            # plt.ylabel('Reward')
            #
            # plt.subplot(122)
            #
            # plt.plot(self.trace_P_max_avg)
            # x1 = np.array(list(range(0, len(self.trace_P_max_avg))))
            # y1 = np.array(self.trace_P_max_avg)
            # error1 = np.array(self.trace_P_max_std)
            #
            # plt.fill_between(x1, y1 - error1, y1 + error1, alpha=0.2)
            # plt.ylabel('Avg % of P_max')
            #
            # plt.ylim([50, 100])


            plt.subplot(111)

            plt.plot(self.trace_P_max_avg)
            x1 = np.array(list(range(0, len(self.trace_P_max_avg))))
            y1 = np.array(self.trace_P_max_avg)
            error1 = np.array(self.trace_P_max_std)

            plt.fill_between(x1, y1 - error1, y1 + error1, alpha=0.2)
            plt.ylabel('Average % of $P_{max}$')
            plt.xlabel('Episode #')

            # plt.subplot(313)
            #
            # plt.plot(self.trace_P_max_steps_avg)
            # plt.ylabel('Steps to reach P_max')
            # plt.xlabel('Episode #')

            plt.pause(0.001)

        if len(self.trace_rw) > 1:
            self.trace_P_episode = self.trace_P_episode[-2:]
            self.trace_dP_episode = self.trace_dP_episode[-2:]
            self.trace_V_episode = self.trace_V_episode[-2:]
            self.trace_I_episode = self.trace_I_episode[-2:]
            self.trace_dPdV_episode = self.trace_dPdV_episode[-2:]

        else:
            self.trace_P_episode = [0, 0]
            self.trace_dP_episode = [0, 0]
            self.trace_V_episode = [0, 0]
            self.trace_I_episode = [0, 0]
            self.trace_dPdV_episode = [0, 0]

        self.trace_rw_episode = []

        self.sum_rw = 0

        self.steps = 0

        self.G = np.random.randint(self.G_min, self.G_max, self.G.shape)

        self.Temp = np.random.randint(self.T_min, self.T_max)

        self.pv_array = ShadedArray(self.G, self.Temp, self.V_min, self.V_max)

        self.current_max_pv_point, self.current_p_curve, self.current_i_curve = self.pv_array.pv_max, self.pv_array.p_curve, self.pv_array.i_curve

        self.max_pv_episode_so_far = [0, 0]

        self.episode_count += 1

        return np.zeros(self.state_dim)

    def normalizing_state(self, state):

        st = []
        cont = 0
        for i, m in enumerate(self.states_mask):
            if m == 1:
                st.append((2 * (state[cont] - self.full_st_min[i]) / (
                        self.full_st_max[i] - self.full_st_min[i])) - 1)
                cont += 1

        st = np.array(st)
        st[st < self.min_stateValue] = self.min_stateValue
        st[st > self.max_stateValue] = self.max_stateValue

        st = (st + 1) * self.discrete_bins / 2

        st = st.astype(np.uint)

        st = st.astype(np.float32)

        st = (st * 2 / self.discrete_bins) - 1

        return st.astype(np.float32)

    def step(self, action, render=False):

        while isinstance(action, (list, tuple, np.ndarray)):
            action = action[0]

        self.steps += 1

        pv_voltage = self.trace_V_episode[-1]

        if self.discrete_actions is None:
            # continuous action
            # action is normalized
            V = pv_voltage + action * self.DeltaV_max
        else:
            # action is an index of possible discrete actions
            V = pv_voltage + self.discrete_actions[action]

        # limit voltage range
        V = np.clip(V, self.V_min, self.V_max)

        I_new, V_new, P_new = self.pv_array.data(V)

        dV = V_new - self.trace_V_episode[-1]  # pv_voltage(i) - pv_voltage(i-1)
        dP = P_new - self.trace_P_episode[-1]  # pv_power(i) - pv_power(i-1)
        dI = I_new - self.trace_I_episode[-1]  # pv_current(i) - pv_current(i-1)

        done = self.steps >= self.MaxSteps

        #reward = self.reward_function(dP, P_new, dV, V_new, done)
        if self.reward_function == 'distance':
            reward = self.distance_reward_function(P_new, V_new)
        elif self.reward_function == 'power':
            reward = self.power_reward_function(P_new, V_new)
        else:
            raise Exception('Reward function ' + self.reward_function + ' not implemented')

        self.trace_P_episode.append(P_new)
        self.trace_dP_episode.append(dP)
        self.trace_V_episode.append(V_new)
        self.trace_I_episode.append(I_new)
        self.trace_dPdV_episode.append(dP / (dV + 0.0000001))
        self.trace_rw_episode.append(reward)

        if render:
            self.render()

        #  'V', 'I', 'P', 'dV', 'dI', 'dP', 'dP/dV', 'd(dP/dV)', 'T'
        full_states = [V_new,
                       I_new,
                       P_new,
                       dV,
                       dI,
                       dP,
                       self.trace_dPdV_episode[-1],
                       self.trace_dPdV_episode[-1] - self.trace_dPdV_episode[-2],
                       self.Temp
                       ]

        state = []
        for i, m in enumerate(self.states_mask):
            if m == 1:
                state.append(full_states[i])

        state = np.reshape(
            np.hstack(state),
            (self.state_dim,))

        info = {'analog state': state.copy(),
                'full state': full_states.copy(),
                'V': V_new,
                'I': I_new,
                'P': P_new}

        self.sum_rw += reward

        n_state = self.normalizing_state(state)

        self.trace_analog_state.append(state.copy())
        self.trace_norm_state.append(n_state)

        return n_state, reward, done, info

    def render(self, mode='human', close=False):
        plt.figure(self.alg_name + ' sample').clear()

        plt.subplot(311)
        plt.plot(np.linspace(self.V_min, self.V_max, len(self.current_p_curve)), self.current_p_curve)
        plt.plot(self.trace_V_episode[-1], self.trace_P_episode[-1], 'go')
        plt.plot(self.current_max_pv_point[1], self.current_max_pv_point[0], 'ro')
        plt.xlabel('Voltage')
        plt.ylabel('Power')
        # plt.ylim([self.P_min, self.P_max])
        plt.ylim([self.P_min, self.current_max_pv_point[0]])

        plt.subplot(312)
        plt.plot(self.trace_P_episode)
        plt.ylabel('Power')
        # plt.ylim([self.P_min, self.P_max])
        plt.ylim([self.P_min, self.current_max_pv_point[0]])

        plt.subplot(313)
        plt.plot(self.trace_rw_episode)
        plt.xlabel('Step')
        plt.ylabel('reward')

        plt.pause(0.00001)

    def take_action(self, action):
        pass

    def power_reward_function(self, P, V):
        # distance to the maximum power
        r = P / (10 * self.current_max_pv_point[0])

        while isinstance(r, np.ndarray):
            r = r[0]

        return r

    def distance_reward_function(self, P, V):
        # -distance to Vmax
        r = - np.abs(self.current_max_pv_point[1] - V) / 100

        if P < self.current_max_pv_point[0] * 0.1:
            r = -1

        while isinstance(r, np.ndarray):
            r = r[0]

        return r
