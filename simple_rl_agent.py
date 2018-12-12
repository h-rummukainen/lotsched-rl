import numpy as np
from copy import deepcopy

class SimpleRLAgent:
    def __init__(self, genv,
                 alpha_init=1e-5, alpha_steps_rel=1/500.0,
                 beta_init=1e-5, beta_steps_rel=1/500.0,
                 eps_init=0.1, eps_steps_rel=1/100.0, callbacks=[],
                 multiagent=False, debug=False):
        def make_agent():
            return SmartAlgorithm(genv, alpha_init, alpha_steps_rel,
                                  beta_init, beta_steps_rel,
                                  eps_init, eps_steps_rel, debug)
        self.genv = genv
        self._multiagent = multiagent
        n_agents = genv.n_products+1 if multiagent else 1
        self.agents = [make_agent() for i in range(n_agents)]
        # indices: (current product, action, product queue)
        k = genv.observation_space.shape[0] - genv.n_product_obs
        self._weights = np.zeros((genv.n_products+1, genv.action_space.n, k))
        self._callbacks = callbacks

    def fit(self, total_steps, restore=False):
        for agent in self.agents:
            agent.initialize_fit(total_steps)
        state = self.genv.reset()
        si = self.genv.decode_state()
        log_init = {'state_arrays': [[]], 'rewards': [], 'durations': [],
                    'actions': []}
        log = deepcopy(log_init)
        for iteration in range(total_steps):
            i_agent = self._select_agent(si)
            weights = self._weights[i_agent]
            action = self.agents[i_agent].iterate_fit(si, weights)
            state, r, done, info = self.genv.step(action)
            t = info.get('step_time', 1.0)
            si = self.genv.decode_state()
            if done:
                state = self.genv.reset()
                si = self.genv.decode_state()
                for agent in self.agents:
                    agent.reset_iteration()

                for cb in self._callbacks:
                    cb.on_rollout(log, iteration)
                log = deepcopy(log_init)
            else:
                split_reward = info.get('split_reward', None)
                if self._multiagent and split_reward is not None:
                    for agent,s in zip(self.agents, split_reward):
                        agent.accumulate(s, t)
                else:
                    for agent in self.agents:
                        agent.accumulate(r, t)

                log['state_arrays'][0].append(state)
                log['rewards'].append(r)
                log['durations'].append(t)
                log['actions'].append(action)

        for cb in self._callbacks:
            cb.on_rollout(log, iteration)

    def _select_agent(self, si):
        if self._multiagent:
            if si.obs_idle:
                i_agent = self.genv.n_products
            else:
                i_agent = si.obs_product
        else:
            i_agent = 0
        return i_agent

    def restore(self):
        pass

    def select_action(self, denv_state):
        state = np.concatenate(denv_state)
        si = self.genv.decode_state(state)
        i_agent = self._select_agent(si)
        a, qvalue = self.agents[i_agent].select_action_value(
            si, self._weights[i_agent])
        return a


class SmartAlgorithm:
    def __init__(self, genv, alpha_init, alpha_steps_rel,
                 beta_init, beta_steps_rel, eps_init, eps_steps_rel, debug):
        self.gamma = 1.0 - genv.discount_rate
        self.alpha_init = alpha_init
        self.alpha_steps = None
        self.alpha_steps_rel = alpha_steps_rel
        self.beta_init = beta_init
        self.beta_steps = None
        self.beta_steps_rel = beta_steps_rel
        self.eps_init = eps_init
        self.eps_steps = None
        self.eps_steps_rel = eps_steps_rel
        self._debug = debug

    def initialize_fit(self, total_steps):
        self.alpha_steps = total_steps * self.alpha_steps_rel
        self.eps_steps = total_steps * self.eps_steps_rel
        if self.gamma == 1.0:
            self.beta_steps = total_steps * self.beta_steps_rel
            self.total_duration = 0.0
            self.reward_rate = 0.0
        self.iteration = 0
        self.reset_iteration()

    def iterate_fit(self, state_info, weights):
        self.iteration += 1
        if self.current_state is not None:
            self.update_weights(
                self.current_state, self.current_action, self.exploring,
                state_info, self.current_reward, self.current_duration,
                weights)
        a, exploring = self.select_action_or_random(
            self.iteration, state_info, weights)
        self.current_state = state_info
        self.current_action = a
        self.exploring = exploring
        self.current_reward = 0.0
        self.current_duration = 0.0
        return a

    def reset_iteration(self):
        self.current_state = None
        self.current_action = None
        self.exploring = False
        self.current_reward = 0.0
        self.current_duration = 0.0

    def accumulate(self, reward, duration):
        if self.current_state is not None:
            self.current_reward += self.gamma ** self.current_duration * reward
            self.current_duration += duration

    def update_weights(self, state_info, action, exploring,
                       next_state_info, reward, duration, weights):
        p_current = state_info.obs_product
        x = state_info.obs_rest
        Q_state_action = self.qvalue(state_info, action, weights)
        _,V_next_state = self.select_action_value(next_state_info, weights)
        if self.gamma == 1.0:
            e = (reward - self.reward_rate * duration + V_next_state
                 - Q_state_action)
            if not exploring:
                sum_duration = self.total_duration + duration
                ru = (self.total_duration * self.reward_rate
                      + reward) / sum_duration
                self.total_duration = sum_duration
                beta = self.beta_init / (1 + self.iteration / self.beta_steps)
                self.reward_rate = (1.0 - beta) * self.reward_rate + beta * ru
        else:
            e = reward + self.gamma ** duration * V_next_state - Q_state_action
        alpha = self.alpha_init / (1 + self.iteration / self.alpha_steps)
        if self._debug:
            print("it={} prod={} a={} i={}(q={}) r={} t={} j={}(v={}) e={} weights={} +{}\n"
                  .format(self.iteration, p_current, action,
                          state_info.format_brief(), Q_state_action,
                          reward, duration,
                          next_state_info.format_brief(), V_next_state,
                          e, weights[action],
                          alpha*e*x))
        weights[action] += alpha * e * x

    def select_action_or_random(self, iteration, state_info, weights):
        prod = state_info.obs_product
        a, qvalue = self.select_action_value(state_info, weights)
        other_actions = [i for i in state_info.feasible_actions if i != a]
        if len(other_actions) > 0:
            eps = self.eps_init / (1 + self.iteration / self.eps_steps)
            if np.random.uniform() < eps:
                return (np.random.choice(other_actions), True)
        return (a, False)

    def select_action_value(self, state_info, weights):
        qvalue_best = -np.infty
        a_best = 0
        for a in state_info.feasible_actions:
            qvalue = self.qvalue(state_info, a, weights)
            if qvalue > qvalue_best:
                qvalue_best = qvalue
                a_best = a
        return (a_best, qvalue_best)

    def qvalue(self, si, action, weights):
        return np.dot(si.obs_rest, weights[action])
