import os, re, math, bisect
import numpy as np
import gym
from gym import error, spaces

_EVENT_END = 0
_EVENT_ORDER = 1

class EventQueue:
    def __init__(self):
        self._times = []
        self._events = []

    def clear(self):
        self._times.clear()
        self._events.clear()

    def add(self, time, event):
        i = bisect.bisect_left(self._times, -time)
        self._times.insert(i, -time)
        self._events.insert(i, event)

    def peek_time(self):
        return -self._times[-1] if len(self._times) > 0 else None

    def peek_event(self):
        return self._events[-1] if len(self._events) > 0 else None

    def pop(self):
        self._times.pop()
        return self._events.pop()

def make_lognormal_generator(mean, scv, scale=1.0):
    normal_mean = math.log(mean / math.sqrt(1.0 + scv))
    normal_stdev = math.sqrt(math.log(1.0 + scv))
    def generate(random):
        return random.lognormal(normal_mean, normal_stdev) * scale
    return generate

def make_unif_int_generator(low, high):
    def generate(random):
        return random.randint(low, high)
    return generate

class LotSchedulingEnv(gym.Env):
    def __init__(self, model, observation_space,
                 discount_rate=0.0, seed=None, config=None,
                 debug=False, logfile=None):
        super().__init__()
        model = os.path.basename(model)
        if model == "lot-scheduling-control.cfg":
            order_scv = 1.0
            init_range = 300
        elif model == "lot-scheduling-control-025.cfg":
            order_scv = 0.25
            init_range = 100
        else:
            raise Exception("Unrecognized model filename {}".format(model))
        self._debug = debug
        self._logfile = logfile

        self.discount_rate = discount_rate
        self._integrator_dt = 5.0
        self._n_products = 3
        self.action_space = spaces.Discrete(self._n_products + 1)
        self.observation_space = observation_space

        self._event_queue = EventQueue()
        self._seed = seed or 0
        self._random = np.random.RandomState(self._seed)
        self._closed = False
        self._runduration = 5 * 24 * 60 * 60

        production_rates_per_h = [  45,   30,   35]
        setup_times_h          = [0.10, 0.20, 0.15]
        setup_costs            = [ 5.0, 10.0, 10.0]
        holding_costs_per_h    = [ 1.0,  1.0,  1.0]
        backlog_costs_per_h    = [25.0, 12.5, 20.0]
        order_means            = [25.0, 10.0, 10.0]
        def float64array(a):
            return np.array(a, dtype=np.float64)
        self._production_times = 3600.0 / float64array(production_rates_per_h)
        self._setup_times = float64array(setup_times_h) * 3600.0
        self._setup_costs = float64array(setup_costs)
        self._holding_costs = float64array(holding_costs_per_h) / 3600.0
        self._backlog_costs = float64array(backlog_costs_per_h) / 3600.0
        self._order_size_generators = [
            make_lognormal_generator(order_mean, order_scv)
            for order_mean in order_means]
        self._order_interval_generator = \
            make_lognormal_generator(mean=2.0, scv=0.25, scale=3600.0)
        self._queue_min = [-500] * self._n_products
        self._queue_max = [500] * self._n_products
        self._idletime = 120.0
        self._init_generator = make_unif_int_generator(-init_range, init_range)

        self.reset()

    def set_discount_rate(self, rate, time_constant_s):
        self.discount_rate = -math.expm1(math.log1p(-rate) / time_constant_s)
        return self.discount_rate

    def get_discount_rate(self):
        return self.discount_rate

    def seed(self, seed=None):
        if seed is None:
            self._seed = int.from_bytes(os.urandom(4), 'big')
        elif not isinstance(seed, int):
            raise error.Error('Seed must be an integer')
        else:
            self._seed = seed
        self._random.seed(self._seed)
        if self._debug:
            print("seed({}) = {}".format(seed, self._seed))
        return [self._seed]

    def step(self, action):
        if self._closed:
            raise error.Error('Environment closed')
        action = int(action)
        self._step_start_time = self._time
        t0 = self._time
        r0 = self._total_reward
        s0 = np.copy(self._split_reward)
        if self._debug:
            print("step(action={}) at t={:.3f}, p={}, q={}".format(
                action, t0, self._current_product, self._queue_levels))
        if action == 0:
            # Idle action: Wait for next event, at most for a fixed idle time.
            done = self._advance(min(self._event_queue.peek_time(),
                                     t0 + self._idletime))
        elif action <= self._n_products:
            # Product action: Set up if necessary, then produce one item.
            p = action - 1
            t_finish = t0 + self._production_times[p]
            if p != self._current_product:
                # Set up
                t_finish += self._setup_times[p]
                self._current_product = p
                self._total_setups[p] += 1
                self._total_reward -= self._setup_costs[p]
            done = self._advance(t_finish)
            self._queue_levels[p] = min(self._queue_levels[p] + 1,
                                        self._queue_max[p])
        else:
            raise error.Error('Invalid action {}'.format(action))
        reward = self._total_reward - r0
        duration = self._time - t0
        split_reward = self._split_reward - s0
        info = {'step_time': duration, 'split_reward': split_reward}
        if self._debug:
            print("  t={:.3f} done={} reward={:.3f} duration={:.3f}".format(
                self._time, done, reward, duration))
        if self._logfile:
            self._logfile.write('{},{},{},{},{},{},{}\n'.format(
                self._time, action, done, reward, duration,
                self._current_product + 1,
                ','.join(str(ql) for ql in self._queue_levels)))
        return (self._get_state(), reward, done, info)

    def reset(self):
        if self._closed:
            raise error.Error('Environment closed')
        self._random.seed(self._seed)
        self._queue_levels = np.array([self._init_generator(self._random)
                                       for i in range(self._n_products)],
                              dtype=np.float64)
        self._current_product = 1
        self._time = 0.0
        self._integrator_time = 0.0
        self._total_reward = 0.0
        self._split_reward = np.zeros(self._n_products + 1)
        self._total_setups = [0] * self._n_products

        self._event_queue.clear()
        self._event_queue.add(self._runduration, _EVENT_END)
        self._schedule_next_order()

        if self._logfile:
            self._logfile.write('# reset seed={} discount_rate={}\n'.format(
                self._seed, self.discount_rate))
            self._logfile.write('{},{},{},{},{},{}\n'.format(
                self._time, -1, False, 0, 0, self._current_product + 1,
                ','.join(str(ql) for ql in self._queue_levels)))

        return self._get_state()

    def _get_state(self):
        return np.concatenate(([self._current_product + 1],
                               self._queue_levels))

    def _advance(self, t_finish):
        done = False
        t_event = self._event_queue.peek_time()
        if self._debug:
            print("  _advance: t={:.3f} t_finish={:.3f} t_event={:.3f}".format(
                self._time, t_finish, t_event))
        r0 = self._total_reward
        while not done and t_event <= t_finish:
            r = self._integrate_reward(self._time, t_event)
            self._time = t_event
            if self._debug:
                print("    t={:.3f} event={} reward={:.3f}".format(
                    self._time, self._event_queue.peek_event(), r))
            done = self._process_event()
            t_event = self._event_queue.peek_time()
        self._integrate_reward(self._time, t_finish)
        self._time = t_finish
        if self._debug:
            print("    t={:.3f} event=(action) reward={:.3f}".format(
                self._time, self._total_reward - r0))
        return done

    def _integrate_reward(self, t0, t1):
        w = 0.0
        t = self._integrator_time
        while t < t1:
            w += self._integrator_dt * math.exp(
                (self._step_start_time - t) * self.discount_rate)
            t += self._integrator_dt
        self._integrator_time = t

        cost_rate = 0.0
        total_backlog_cr = 0.0
        split_holding_cr = np.zeros(self._n_products + 1)
        for p,q in enumerate(self._queue_levels):
            if q > 0:
                r = self._holding_costs[p] * q
                split_holding_cr[p] += r
            else:
                r = self._backlog_costs[p] * -q
                total_backlog_cr += r
            cost_rate += r
        self._total_reward += (-w) * cost_rate
        self._split_reward += (-w) * (split_holding_cr + total_backlog_cr)
        return cost_rate

    def _process_event(self):
        e = self._event_queue.pop()
        if e == _EVENT_ORDER:
            self._process_order()
            return False
        elif e == _EVENT_END:
            return True
        else:
            raise error.Error('Invalid event {}'.format(e))

    def _process_order(self):
        amounts = [np.floor(self._order_size_generators[p](self._random))
                   for p in range(self._n_products)]
        for p,n in enumerate(amounts):
            self._queue_levels[p] = max(self._queue_levels[p] - n,
                                        self._queue_min[p])
        self._schedule_next_order()

    def _schedule_next_order(self):
        wait = self._order_interval_generator(self._random)
        if self._debug:
            print("      _schedule_next_order:"
                  " t={:.3f}, wait={:.3f}, t_order={:.3f}".format(
                      self._time, wait, self._time + wait))
        self._event_queue.add(self._time + wait, _EVENT_ORDER)

    def close(self):
        self._closed = True

    def render(self, mode='human', close=False):
        return False

    def configure(self, line):
        if self._logfile:
            self._logfile.write('# configure {}.\n'.format(line))
        m = re.match("ServerIdle ServiceTime { ([-+\d.eE]+) +s }", line)
        if m:
            self._idletime = float(m.group(1))
            return
        m = re.match("Simulation RunDuration { ([-+\d.eE]+) d }", line)
        if m:
            self._runduration = float(m.group(1)) * 86400.0
            return
        raise Exception('Configuration line not recognized: '+line)

    # An open file is not picklable, so passing the env to another process
    # means the other process will not log.
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_logfile'] = None
        return state

if __name__ == '__main__':
    env = LotSchedulingEnv(model="data/lot-scheduling-control.cfg",
                           observation_space = spaces.Box(
                               np.array([1, -100, -100, -100]),
                               np.array([3, 100, 100, 100])),
                           debug=False)
    env.seed(1)
    #env.render()
    #env.set_discount_rate(0.1, 3600.0)

    print("action space: ", env.action_space)
    print("observation space: ", env.observation_space)

    obs = env.reset()
    reward = 0
    for run in range(5):
        print()
        total_reward = 0.0
        total_duration = 0.0
        for step in range(10000):
            a = step % env.action_space.n
            #print(run, ":", step, ": r=", reward, " s=", obs, " a=", a)
            obs, reward, done, info = env.step(a)
            total_reward += reward
            total_duration += info.get('step_time', 1)
            if done:
                break
        print("total reward={:.3f}, duration={:.3f}, rate={:.3f}, steps={}"
              .format(total_reward, total_duration,
                      total_reward/total_duration*3600, step))
        env.seed(run+2)
        obs = env.reset()
        reward = 0

    print()
    print("Closing...")
    env.close()
    print("Done.")
