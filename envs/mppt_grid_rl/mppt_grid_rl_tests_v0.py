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
        self.Pmax = self.Iscr * self.Voc  # W

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


'''
Test 1: Random temperature and irradiance, both are kept constant for the entire episode. Episode duration is 1000 steps
Test 2: 
'''


class MpptEnvGridTest_V0(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, alg_name, test_name, states_mask, panels_series, panels_parallel,
                 discrete_bins=50, discrete_actions=None):
        assert test_name in ['Test 1', 'Test 2', 'Test 3']

        self.alg_name = alg_name
        self.test_name = test_name

        self.n_panels = panels_series * panels_parallel

        self.G = np.zeros((panels_parallel, panels_series))  # Solar radiation array in W / sq.m
        self.V_min = 1
        self.V_max = 21 * panels_series

        self.I_min = 0
        self.I_max = Panel().Iscr

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

        self.steps = 0

        self.MaxSteps = 10000

        if self.test_name == 'Test 3':
            self.MaxSteps = 1000

        self.trace_rw = []
        self.trace_rw_episode = []
        self.trace_P_episode = []
        self.trace_P_max_episode = []
        self.trace_dP_episode = []
        self.trace_V_episode = []
        self.trace_I_episode = []
        self.trace_dPdV_episode = []
        self.trace_efficiency_episode = []

        self.trace_V_max_dist = []
        self.trace_P_avg = []
        self.trace_P_max_std = []
        self.trace_P_max_avg = []
        self.trace_P_max_steps = []
        self.trace_P_max_steps_avg = []

        self.trace_analog_state = []
        self.trace_norm_state = []

        self.trace_irradiance = []
        self.trace_temperature = []

        self.current_max_pv_point = [0, 0]
        self.current_p_curve = []
        self.current_i_curve = []

        self.sum_rw = 0

        # initialize random temperature and irradiance
        self.Temp = np.random.randint(self.T_min, self.T_max)
        self.G = np.random.randint(100, 1000, (panels_parallel, panels_series))

        self.episode_count = 0

    def reset(self):
        if len(self.trace_P_episode) > 0:
            avg_p_episode = np.average(self.trace_P_episode[:])
            max_v_dist = np.abs(self.current_max_pv_point[1]
                                - np.average(self.trace_V_episode[-10:]))

            max_steps = np.argmax(self.trace_P_episode)
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

        # initialize random temperature and irradiance
        self.Temp = np.random.randint(self.T_min, self.T_max)
        self.G = np.random.randint(self.G_min, self.G_max, self.G.shape)
        self.G_centered = np.random.randint(100, 900, self.G.shape)

        # update model
        self.pv_array = ShadedArray(self.G, self.Temp, self.V_min, self.V_max)
        self.current_max_pv_point, self.current_p_curve, self.current_i_curve = self.pv_array.pv_max, self.pv_array.p_curve, self.pv_array.i_curve

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

    def bell_curve(self, x, mean, std, scale, offset):
        y_out = np.exp(- (x - mean) ** 2 / (2 * std ** 2))

        return y_out * scale + offset

    def step(self, action, render=False, analog_action=False):

        while isinstance(action, (list, tuple, np.ndarray)):
            action = action[0]

        self.steps += 1

        if self.test_name == 'Test 2':
            self.Temp = self.bell_curve(self.steps % 1000, 500, 100, 70, 0)

            # update model
            self.pv_array = ShadedArray(self.G, self.Temp, self.V_min, self.V_max)
            self.current_max_pv_point, self.current_p_curve, self.current_i_curve = self.pv_array.pv_max, self.pv_array.p_curve, self.pv_array.i_curve

        elif self.test_name == 'Test 3':
            self.Temp = self.bell_curve(self.steps, 500, 100, 70, 0)

            self.G = self.bell_curve(self.steps, self.G_centered, 300, 900, 100)

            # update model
            self.pv_array = ShadedArray(self.G, self.Temp, self.V_min, self.V_max)
            self.current_max_pv_point, self.current_p_curve, self.current_i_curve = self.pv_array.pv_max, self.pv_array.p_curve, self.pv_array.i_curve

            # self.render()

        if self.steps % 1000 == 0 and (self.test_name == 'Test 1' or self.test_name == 'Test 2'):
            # random reset of irradiance in this test
            self.G = np.random.randint(self.G_min, self.G_max, self.G.shape)

            # update model
            self.pv_array = ShadedArray(self.G, self.Temp, self.V_min, self.V_max)
            self.current_max_pv_point, self.current_p_curve, self.current_i_curve = self.pv_array.pv_max, self.pv_array.p_curve, self.pv_array.i_curve

            # self.render()

        pv_voltage = self.trace_V_episode[-1]

        if (analog_action is True) or (self.discrete_actions is None):
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

        # no reward in the testing environment
        reward = 0

        self.trace_P_episode.append(P_new)
        self.trace_dP_episode.append(dP)
        self.trace_V_episode.append(V_new)
        self.trace_I_episode.append(I_new)
        self.trace_dPdV_episode.append(dP / (dV + 0.0000001))
        self.trace_rw_episode.append(reward)

        self.trace_irradiance.append(self.G[0][0:3])
        self.trace_temperature.append(self.Temp)
        self.trace_P_max_episode.append(self.current_max_pv_point[0])
        self.trace_efficiency_episode.append(P_new / self.current_max_pv_point[0])

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
                'P': P_new,
                'MaxP': self.current_max_pv_point[0]}

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
        plt.ylim([self.P_min, self.P_max])
        # plt.ylim([self.P_min, self.current_max_pv_point[0]])

        plt.subplot(312)
        plt.plot(self.trace_P_episode)
        plt.ylabel('Power')
        plt.ylim([self.P_min, self.P_max])
        # plt.ylim([self.P_min, self.current_max_pv_point[0]])

        plt.subplot(313)
        plt.plot(self.trace_efficiency_episode)
        plt.xlabel('Step')
        plt.ylabel('Efficieny')

        # plt.plot(self.trace_P_max_episode)
        # plt.xlabel('Step')
        # plt.ylabel('P$_{max}$(W)')
        #
        # r_id = np.random.randint(1000)
        # plt.savefig(self.test_name + '_' + str(r_id) + '_max_p.png', dpi=500)

        fig = plt.figure(self.alg_name + ' Irradiance and Temperature')
        fig.clear()
        ax1 = fig.add_subplot(111)

        color = 'tab:orange'
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Irradiance (W/m$^2$)', color=color)
        ax1.plot(list(range(len(self.trace_temperature))), self.trace_irradiance, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        color = 'tab:blue'
        ax2 = ax1.twinx()
        ax2.set_ylabel(r'Temperature ($\degree$C)', color=color)
        ax2.plot(list(range(len(self.trace_temperature))), self.trace_temperature, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()

        plt.pause(0.01)
        #plt.savefig(self.test_name + '_' + str(r_id) +'_temp_irr.png', dpi=500)

    def take_action(self, action):
        pass
