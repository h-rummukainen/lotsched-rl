import argparse, math, uuid
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Deep-learning on JaamSim lot-scheduling model")
    parser.add_argument('-i', '--id', default=uuid.uuid4().hex[:8])
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('-j', '--jaamsim', action='store_true')
    parser.add_argument('--continue', dest='continve', action='store_true')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--test-runs', type=int, default=30)
    parser.add_argument('--episode-days', type=int, default=5)
    parser.add_argument('--order-scv', choices=["large", "small"])
    parser.add_argument('--single-product', action='store_true')
    parser.add_argument('--max-idle', type=float, default=120)
    parser.add_argument('--action')
    parser.add_argument('--draw-q', action='store_true')
    parser.add_argument('--discount', type=float, default=0.1) # per hour
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--multi-agent', action='store_true')
    parser.add_argument('--single-layer', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('method', choices=["ppo", "a3c", "rotation",
                                           "simple-rl", "simple-rl-tiled"])
    args = parser.parse_args()

    # start fork server before import of lot_scheduling leads to JVM start-up
    if args.parallel:
        import multiprocessing as mp, logging
        if args.debug:
            logger = mp.log_to_stderr()
            logger.setLevel(logging.DEBUG)
        mp.set_start_method('forkserver')

import numpy, sys, csv
numpy.seterr(all='raise')
from copy import deepcopy
from collections import defaultdict
import gym
import deepchem.rl
from deepchem.models.tensorgraph.layers import \
  Reshape, Dense, SoftMax, Repeat, Stack, Gather, Variable, Dropout, Concat
from deepchem.models.tensorgraph.optimizers import Adam
from dc_layers import AddConstant, InsertBatchIndex
import tensorflow
import lot_scheduling
from utils import make_breakpoints, \
    format_actions, format_observations, format_losses
from utils import join_episode_totals, print_episode_totals, iso8601stamp
from graphs import draw_ac
from simple_rl_agent import SimpleRLAgent


def convert_env(genv):
    return deepchem.rl.GymEnvironment(
        'lotsched', genv,
        state_dtype=[numpy.int32, numpy.float32],
        state_slices=[(slice(0,1),),
                      (slice(1,None),)])

def _parse_action(a):
    action = None
    if a:
        action_args = a.split(':', 1)
        if len(action_args) == 1:
            action = action_args[0]
        else:
            avalues = [int(s) for s in action_args[1].split(',')]
            action = (action_args[0], avalues)
    return action

def make_envs(args):
    qmin, qmax = (lot_scheduling.QMIN, lot_scheduling.QMAX)
    order_scv = 1.0 if args.order_scv == "large" else 0.25
    action = _parse_action(args.action)
    breakpoints = make_breakpoints(qmin, qmax, 4, 0.02)
    if args.method == "rotation" or args.method == "simple-rl":
        breakpoints = None
    genv = lot_scheduling.make_env(
        args.discount, order_scv=order_scv,
        single_product=args.single_product, one_hot=False,
        mark_idle=args.multi_agent,
        visualize=args.visualize,
        breakpoints=breakpoints,
        #xbreakpoints=[(2*step, make_breakpoints(qmin, qmax, step))
        #              for step in [16, 64, 256]],
        #piecewise_queues2=make_breakpoints(qmin, qmax, 32),
        #piecewise_tilewidth2=64,
        action=action,
        reseed=True, debug=args.debug, proxy=args.parallel,
        jaamsim=args.jaamsim
        #, logfilename='dc.env.log'
        )
    return (genv, convert_env(genv))

class MyPolicy(deepchem.rl.Policy):
    def __init__(self, n_products, n_observations, single_layer):
        super(MyPolicy, self).__init__()
        self.n_products = n_products
        self.n_queue_obs = n_observations - 1
        self.single_layer = single_layer

    def create_layers(self, state, **kwargs):
        i = Reshape(in_layers=[state[0]], shape=(-1, 1))
        i = AddConstant(-1, in_layers=[i])
        i = InsertBatchIndex(in_layers=[i])
        # shape(i) = (batch_size, 1)

        q = Reshape(in_layers=[state[1]], shape=(-1, self.n_queue_obs))
        # shape(q) = (batch_size, n_queue_obs)
        #q = Dense(16, in_layers=[q], activation_fn=tensorflow.nn.relu)
        ## shape(q) = (batch_size, 16)

        x = q
        if not self.single_layer:
            for j in range(1):
                x1 = Dense(8, in_layers=[x], activation_fn=tensorflow.nn.relu)
                x = Concat(in_layers=[q, x1])
        # 1) shape(x) = (batch_size, n_queue_obs)
        # 2) shape(x) = (batch_size, n_queue_obs + 8)

        ps = []
        for j in range(self.n_products):
            p = Dense(n_actions, in_layers=[x])
            ps.append(p)
        p = Stack(in_layers=ps, axis=1)
        # shape(p) = (batch_size, n_products, n_actions)
        p = Gather(in_layers=[p, i])
        # shape(p) = (batch_size, n_actions)
        p = SoftMax(in_layers=[p])

        vs = []
        for j in range(self.n_products):
            v = Dense(1, in_layers=[x])
            vs.append(v)
        v = Stack(in_layers=vs, axis=1)
        # shape(v) = (batch_size, n_products, 1)
        v = Gather(in_layers=[v, i])
        # shape(v) = (batch_size, 1)

        return {'action_prob': p, 'value': v}

class LogCallback:
    def __init__(self, uncook_observation):
        self.rollout_count = 0
        self.uncook_observation = uncook_observation
        self.losses_text = ''
        self.losses = defaultdict(float)
        self.batch_count = 0
        self._check_best_reward = 0
        self._check_best_duration = 0
        self._best_reward_rate = -numpy.infty

    def on_training_batch(self, losses, at_steps):
        self.losses_text = ''
        for k,v in losses.items():
            self.losses[k] += v
        self.batch_count += 1

    def on_rollout(self, r, at_steps):
        state_arrays = r['state_arrays']
        if len(state_arrays) == 1:
            obs_of_step = state_arrays[0]
        elif len(state_arrays) == 2:
            obs_of_step = [numpy.concatenate([p,q])
                           for p,q in zip(state_arrays[0], state_arrays[1])]
        else:
            raise ValueError("Invalid state_arrays format")
        obs_text = format_observations([self.uncook_observation(obs)
                                        for obs in obs_of_step])
        reward = numpy.sum(r['rewards']) * lot_scheduling.REWARDSCALE
        duration = numpy.sum(r['durations'])
        r_rate = reward / (duration / 3600.0) if duration > 0 else 0.0
        actions_text = format_actions(r['actions'])
        ep_steps = len(r['rewards'])
        if self.losses_text == '':
            self.losses_text = format_losses(self.losses, self.batch_count)
            self.losses = defaultdict(float)
            self.batch_count = 0

        print("{} episode {} at {} steps: reward={:.3f} rate={:.3f} duration={:.3f} steps={} {} {} {}"
              .format(iso8601stamp(),
                      self.rollout_count, at_steps, reward, r_rate, duration,
                      ep_steps, actions_text, obs_text, self.losses_text))
        ep = self.rollout_count
        self.rollout_count += 1
        return self.check_best(ep, at_steps, reward, duration)

    def check_best(self, episode, at_steps, reward, duration):
        self._check_best_reward += reward
        self._check_best_duration += duration
        best = False
        if episode % 100 == 99:
            reward_rate = (self._check_best_reward / self._check_best_duration
                           * 3600.0 if self._check_best_duration > 0
                           else -np.infty)
            self._check_best_reward = 0
            self._check_best_duration = 0
            if reward_rate > self._best_reward_rate:
                self._best_reward_rate = reward_rate
                best = True
            print("{} episode {} at {} steps: reward rate = {}{}"
                  .format(iso8601stamp(), episode, at_steps, reward_rate,
                          " -> new best" if best else ""))
        return best

class RotationAgent:
    def __init__(self, rotation, base_levels):
        self._rotation = rotation
        self._base_levels = base_levels
        self._index = len(self._rotation) - 1

    def restore(self):
        pass

    def select_action(self, state):
        qlevels = state[1]
        i = self._index
        a = self._rotation[i]
        p = a - 1
        while qlevels[p] >= self._base_levels[p]:
            i = (i + 1) % len(self._rotation)
            if i == self._index:
                a = 0
                break
            a = self._rotation[i]
            p = a - 1
        self._index = i
        return a

def do_test(agent, genv, denv, warmup_s, i, visualize, close):
    genv.seed(i)
    denv.reset()
    t = 0.
    reward = 0.
    duration = 0.
    actions = []
    qlmin = None
    qlmax = None
    log = []
    done = False
    while not done:
        a = agent.select_action(denv.state)
        r, info = denv.step_info(a)
        d = info.get('step_time', 1.0)
        done = denv.terminated
        ql = genv.decode_state().queue_levels
        log.append([t, a, r * lot_scheduling.REWARDSCALE, d,
                    info.get('n_short_steps', 1),
                    int(t+d >= warmup_s)] + list(ql))
        t += d
        if t >= warmup_s:
            reward += r * lot_scheduling.REWARDSCALE
            duration += d
            actions.append(a)
            if not done:
                qlmin = numpy.minimum(ql, qlmin) if qlmin is not None else ql
                qlmax = numpy.maximum(ql, qlmax) if qlmax is not None else ql
    qlevels = (dict((i+1, ql) for i, ql in enumerate(zip(qlmin, qlmax)))
               if qlmin is not None else None)
    print("{} episode {}: reward={:.3f} rate={:.3f} duration={:.3f} {} qlevels={}"
          .format(iso8601stamp(),
                  i, reward, reward / (duration / 3600.0), duration,
                  format_actions(actions), qlevels))
    with open('testrun.{}.csv'.format(i), 'w') as f:
        w = csv.writer(f)
        w.writerow(['t', 'a', 'r', 'd', 'steps', 'active']
                   + ['q{}'.format(i) for i in range(len(log[0]) - 6)])
        for row in log:
            w.writerow(row)
    return { 'reward': [reward], 'duration': [duration] }

if __name__ == '__main__':
    genv, denv = make_envs(args)
    gamma = 1.0 - genv.discount_rate

    genv.root().configure("ServerIdle ServiceTime {{ {:.1f}  s }}".format(
        (args.max_idle - 0.1) if args.max_idle > 0 else 1e9))
    genv.root().configure("Simulation RunDuration {{ {:f} d }}".format(
        args.episode_days if args.episode_days > 0 else 1e6))

    n_actions = genv.action_space.n
    n_products = genv.n_products
    n_observations = genv.observation_space.shape[0]
    print("n_products={} n_observations={} n_actions={}".format(
        n_products, n_observations, n_actions))

    callbacks = [LogCallback(genv.unprocess_observation)]
    if args.method == "ppo":
        agent = deepchem.rl.PPO(
            denv, MyPolicy(n_products, n_observations, args.single_layer),
            max_rollout_length=10000,
            optimization_rollouts=(8 if args.parallel else 1),
            #optimization_epochs=4,
            discount_factor=gamma,
            advantage_lambda=0.98,
            entropy_weight=0.0,
            value_weight=1.0e-4,
            optimizer=Adam(learning_rate=1e-4),
            model_dir="data.ppo."+args.id,
            zero_terminal=False,
            callbacks=callbacks)
    elif args.method == "a3c":
        agent = deepchem.rl.A3C(
            denv, MyPolicy(n_products, n_observations, args.single_layer),
            max_rollout_length=10000,
            discount_factor=gamma,
            advantage_lambda=0.98,
            entropy_weight=0.0,
            value_weight=1.0e-4,
            optimizer=Adam(learning_rate=1e-4),
            model_dir="data.a3c."+args.id,
            worker_count=16,
            zero_terminal=False,
            callbacks=callbacks)
    elif args.method == "rotation":
        action = _parse_action(args.action)
        if action is None or action[0] != "base-stock":
            raise Exception("Use rotation with base-stock action")
        base_levels = action[1]
        agent = RotationAgent([1,2,1,3], base_levels)
    elif args.method == "simple-rl" or args.method == "simple-rl-tiled":
        agent = SimpleRLAgent(genv, callbacks=callbacks,
                              multiagent=args.multi_agent, debug=args.debug)
    else:
        raise ValueError("Unknown method "+args.method)

    if args.steps > 0:
        agent.fit(args.steps, restore=args.continve)
        agent.restore()
    else:
        # read latest checkpoint file in model_dir
        agent.restore()

    if args.draw_q:
        def get_p_and_v(obs):
            pv = agent.predict(denv._mangle_state(obs))
            return pv[:-1], pv[-1]
        draw_ac(genv, get_p_and_v, args.id)

    if args.test_runs > 0:
        print("Starting test runs...")
        warmup_s = 100 * 24 * 60 * 60
        genv.root().configure("Simulation RunDuration { 500 d }")
        genv.set_discount_rate(0, 1)
        denv.reset()
        if args.parallel:
            import multiprocessing.dummy as mp
            pool = mp.Pool(8)
            test_args = []
            for i in range(args.test_runs):
                genv2 = deepcopy(genv)
                denv2 = convert_env(genv2)
                test_args.append((agent, genv2, denv2, warmup_s, i,
                                  (args.visualize and i == 0), args.parallel))
            totals = pool.starmap(do_test, test_args)
        else:
            totals = [do_test(agent, genv, denv, warmup_s, i,
                              (args.visualize and i == 0), False)
                      for i in range(args.test_runs)]
        print_episode_totals(join_episode_totals(totals))
