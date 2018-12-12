import numpy, math, gym
from gym import spaces
from collections import defaultdict

QMIN = -500
QMAX = 500
REWARDSCALE = 1000


class StateInfo(object):
    def __init__(self, obs_product, obs_idle, obs_rest,
                 queue_levels, feasible_actions):
        self.obs_product = obs_product
        self.obs_idle = obs_idle
        self.obs_rest = obs_rest
        self.queue_levels = queue_levels
        self.feasible_actions = feasible_actions

    def format_brief(self):
        return "{{p:{} q:{}}}".format(self.obs_product, self.queue_levels)

class EnvWrapper(gym.Env):
    _owns_render = False

    def __init__(self, parent, action_space, observation_space,
                 n_products=None, n_product_obs=None):
        self.parent = parent
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_products = n_products or parent.n_products
        self.n_product_obs = n_product_obs or parent.n_product_obs
        self.recurse = isinstance(parent, EnvWrapper)
        self._state = None
        self._state_info = None

    def process_action(self, a):
        a = self._process_action(a)
        if self.recurse:
            a = self.parent.process_action(a)
        return a

    def process_observation(self, obs):
        if self.recurse:
            obs = self.parent.process_observation(obs)
        return self._process_observation(obs)

    def process_reward(self, r):
        if self.recurse:
            r = self.parent.process_reward(r)
        return self._process_reward(r)

    def unprocess_observation(self, obs):
        obs = self._unprocess_observation(obs)
        if self.recurse:
            obs = self.parent.unprocess_observation(obs)
        return obs

    def process_product(self, p):
        if self.recurse:
            p = self.parent.process_product(p)
        return self._process_product(p)

    def _process_action(self, a):
        return a

    def _process_observation(self, obs):
        return obs

    def _process_reward(self, r):
        return r

    def _unprocess_observation(self, obs):
        return obs

    def _process_product(self, p):
        return p

    def _reset_hook(self):
        pass

    def step(self, action):
        observation, r, d, info = self.parent.step(
            self._process_action(action))
        self._state = self._process_observation(observation)
        self._state_info = None
        return (self._state, self._process_reward(r), d, info)

    def reset(self):
        self._reset_hook()
        observation = self.parent.reset()
        self._state = self._process_observation(observation)
        self._state_info = None
        return self._state

    def close(self):
        if hasattr(self, 'parent'):
            self.parent.close()

    def seed(self, seed=None):
        return self.parent.seed(seed)

    metadata = {'render.modes': ['human']}
    def render(self, mode='human', close=False):
        return self.parent.render(mode, close)

    @property
    def discount_rate(self):
        return self.parent.discount_rate

    def decode_state(self, state=None):
        if state is None or numpy.allclose(state, self._state):
            if self._state_info is None:
                self._state_info = self._decode_raw_state(
                    self._get_raw_state(), self._state)
            return self._state_info
        else:
            return self._decode_raw_state(
                self.unprocess_observation(state), state)

    def _decode_raw_state(self, raw_state, cooked_state):
        obs_prod = self.process_product(
            numpy.round(raw_state[0]).astype(int)) - 1
        qlevels = raw_state[1:]
        obs_idle = (self.n_product_obs == 2 and cooked_state[1]) #XXX kludge
        obs_rest = cooked_state[self.n_product_obs:]
        actions = self._find_feasible_actions(obs_prod, qlevels)
        return StateInfo(obs_prod, obs_idle, obs_rest, qlevels, actions)

    def _find_feasible_actions(self, prod, qlevels):
        # by default all actions are feasible
        return list(range(self.action_space.n))

    def _get_raw_state(self):
        if self.recurse:
            return self.parent._get_raw_state()
        return self._state

    def root(self):
        if self.recurse:
            return self.parent.root()
        return self.parent

    def set_discount_rate(self, rate, time_constant_s):
        return self.parent.set_discount_rate(rate, time_constant_s)


### Observation modifications ###

def make_float32_obs_space(os):
    if isinstance(os, spaces.Box):
        return spaces.Box(os.low, os.high, dtype=numpy.float32)
    else:
        raise Exception("Unexpected observation space type: " + str(os))

# Convert observations to float32
class EnvWrapperFloat(EnvWrapper):
    def __init__(self, parent, action_space=None, observation_space=None):
        super().__init__(
            parent, action_space=(action_space or parent.action_space),
            observation_space=make_float32_obs_space(
                (observation_space or parent.observation_space)))

    def _process_observation(self, obs):
        return obs.astype(numpy.float32)

    def _process_reward(self, r):
        return r / REWARDSCALE

# Reseed automatically at reset
class EnvWrapperReseed(EnvWrapper):
    def __init__(self, parent, action_space=None, observation_space=None):
        super().__init__(
            parent, action_space=(action_space or parent.action_space),
            observation_space=(observation_space or parent.observation_space))
        self._next_seed = 100

    def _reset_hook(self):
        self.parent.seed(self._next_seed)
        self._next_seed += 1

    def seed(self, value):
        self._next_seed = value + 1
        return self.parent.seed(value)

# Consider only product 2.
# You need to manually modify the JaamSim cfg file to eliminate orders of
# the other products.
class EnvWrapperSingle(EnvWrapper):
    def __init__(self, parent):
        super().__init__(parent,
                         action_space=spaces.Discrete(2),
                         observation_space=spaces.Box(
                             numpy.array([0, QMIN]), numpy.array([1, QMAX]),
                             dtype=numpy.float32),
                         n_products=1, n_product_obs=1)

    def _process_action(self, action):
        return action * 2

    def _process_observation(self, obs):
        return numpy.array([obs[0]//2, obs[2]], dtype=obs.dtype)

    def _unprocess_observation(self, obs):
        return numpy.array([obs[0]*2, 0, obs[1], 0], dtype=obs.dtype)

    def _process_product(self, p):
        return p // 2


def make_unit_box_space(d):
    return spaces.Box(numpy.array([0]*d), numpy.array([1]*d),
                      dtype=numpy.float32)

# Convert product from index to "one-hot" vector
class EnvWrapperProductFlags(EnvWrapper):
    def __init__(self, parent):
        n_products = parent.n_products
        n_obs = parent.observation_space.shape[0] - 1 + n_products
        super().__init__(parent,
                         action_space=parent.action_space,
                         observation_space=make_unit_box_space(n_obs),
                         n_products=n_products, n_product_obs=n_products)
        self._n_obs = n_obs

    def _process_observation(self, obs):
        n_products = self.n_products
        prod = int(obs[0])
        res = numpy.zeros(self._n_obs, dtype=obs.dtype)
        res[n_products:] = obs[1:]
        if prod > 0:
            res[prod - 1] = 1
        return res

    def _unprocess_observation(self, obs):
        n_products = self.n_products
        assert len(obs) == self._n_obs
        res = numpy.zeros(self._n_obs - n_products + 1, dtype=obs.dtype)
        prod = numpy.argmax(obs[0:n_products])
        res[0] = prod+1 if obs[prod] > 0 else 0
        res[1:] = obs[n_products:]
        return res

def make_tile_step(b, bw_down, bw_up, xs):
    if b > 0:
        b_prev = b - bw_down
        return numpy.clip((xs - b_prev) / bw_down, 0.0, 1.0)
    elif b < 0:
        return 1.0 - numpy.clip((xs - b) / bw_up, 0.0, 1.0)
    else:
        b_prev = b - bw_down
        up = numpy.clip((xs - b_prev) / bw_down, 0.0, 1.0)
        down = 1.0 - numpy.clip((xs - b) / bw_up, 0.0, 1.0)
        neg = (xs > 0)
        res = up
        res[neg] = down[neg]
        return res

# Replace current product by special value after idle action
class EnvWrapperCurrentIdle(EnvWrapper):
    def __init__(self, parent):
        n_products = parent.n_products
        assert parent.n_product_obs == 1
        pbox = parent.observation_space
        obs_space = spaces.Box(
            numpy.concatenate((pbox.low[0:1], [0.0], pbox.low[1:])),
            numpy.concatenate((pbox.high[0:1], [1.0], pbox.high[1:])),
            dtype=numpy.float32)
        super().__init__(parent,
                         action_space=parent.action_space,
                         observation_space=obs_space,
                         n_products=n_products, n_product_obs=2)
        self._prev_action = 1

    def _process_observation(self, obs):
        idle = (self._prev_action == 0)
        return numpy.concatenate((obs[0:1], [idle], obs[1:]))

    def _unprocess_observation(self, obs):
        return numpy.concatenate((obs[0:1], obs[2:]))

    def step(self, action):
        self._prev_action = action
        return super().step(action)

    def _reset_hook(self):
        self._prev_action = 1


def make_tiles_step(bp, xs):
    bp = numpy.array(bp, dtype=numpy.float64)
    xs = numpy.array(xs, dtype=numpy.float64)
    bw = bp[1:] - bp[:-1]
    bw_down = numpy.concatenate(([bw[0]], bw))
    bw_up = numpy.concatenate((bw, [bw[-1]]))
    return numpy.stack([make_tile_step(b, wd, wu, xs).astype(numpy.float32)
                        for b, wd, wu in zip(bp, bw_down, bw_up)])

def make_tile2(bp, x, y, w):
    d = (bp - x) / w
    e = (bp - y) / w
    a = (d**2)[:,numpy.newaxis] + (e**2)[numpy.newaxis,:]
    a -= numpy.min(a)
    f = numpy.exp(numpy.clip(-a**2, -700, 700))
    return (f / math.sqrt(numpy.sum(f))).astype(numpy.float32)

class EnvWrapperTiled(EnvWrapper):
    def __init__(self, parent, breakpoints, xbreakpoints=None,
                 breakpoints2=None, tilewidth2=None, action_space=None):
        n_products = parent.n_products
        n_product_obs = parent.n_product_obs
        nbp = len(breakpoints)
        nbpx = sum(len(xb) for tw,xb in xbreakpoints) if xbreakpoints else 0
        nbp2 = len(breakpoints2) if breakpoints2 else 0
        n_obs = (n_product_obs + n_products * nbp + n_products * nbpx
                 + n_products * (n_products - 1) // 2 * nbp2 * nbp2)
        super().__init__(parent,
                         action_space=(action_space or parent.action_space),
                         observation_space=make_unit_box_space(n_obs))
        self.breakpoints = numpy.array(breakpoints, dtype=numpy.int32)
        assert(numpy.all(numpy.diff(self.breakpoints) > 0))
        self._qmin = min(breakpoints)
        self._qmax = max(breakpoints)
        qs = numpy.array(range(self._qmin, self._qmax + 1))
        self._basis = make_tiles_step(self.breakpoints, qs).transpose()
        if xbreakpoints:
            self.xbreakpoints = [
                (tw, numpy.array(xb, dtype=numpy.int32),
                 make_tiles_step(xb, qs))
                 for tw,xb in xbreakpoints]
        else:
            self.xbreakpoints = None
        if breakpoints2:
            self.breakpoints2 = numpy.array(breakpoints2, dtype=numpy.int32)
            self._basis2 = [[make_tile2(self.breakpoints2, x, y, tilewidth2)
                             .flatten()
                             for x in range(self._qmin, self._qmax + 1)]
                            for y in range(self._qmin, self._qmax + 1)]
        else:
            self.breakpoints2 = None
            self._basis2 = None
        self._backmap = defaultdict(list)
        for i,v in enumerate(self._basis):
            self._backmap[numpy.argmax(v)].append(i)

    def _process_observation(self, obs):
        n_products = self.n_products
        n_product_obs = self.n_product_obs
        prod = obs[0:n_product_obs]
        ql = obs[n_product_obs:]
        qi = numpy.rint(ql - self._qmin).astype(numpy.int32)
        vectors = [prod] + [self._basis[qi[p]]
                            for p in range(n_products)]
        if self.xbreakpoints:
            vectors += [basis[qi[p]]
                        for tw,xa,basis in self.xbreakpoints
                        for p in range(n_products)]
        if self._basis2:
            vectors += [self._basis2[qi[p1]][qi[p2]]
                        for p1 in range(n_products-1)
                        for p2 in range(p1+1, n_products)]
        return numpy.hstack(vectors)

    def _unprocess_observation(self, obs):
        n_products = self.n_products
        n_product_obs = self.n_product_obs
        n_breakpoints = len(self.breakpoints)
        assert len(obs) >= n_product_obs + n_products * n_breakpoints
        res = numpy.zeros(n_product_obs + n_products, dtype=numpy.float32)
        res[0:n_product_obs] = obs[0:n_product_obs]
        i0 = n_product_obs
        for p in range(n_products):
            i1 = i0 + n_breakpoints
            v = obs[i0:i1]
            best_bi = -1
            best_e2 = numpy.inf
            for bi in self._backmap[numpy.argmax(v)]:
                e = self._basis[bi] - v
                e2 = numpy.dot(e, e)
                if e2 < best_e2:
                    best_e2 = e2
                    best_bi = bi
            res[1+p] = self._qmin + best_bi
            i0 = i1
        return res


### Action modifications ###

class EnvWrapperContinueAction(EnvWrapper):
    def __init__(self, parent, observation_space=None):
        super().__init__(parent,
                         action_space=spaces.Discrete(2 + parent.n_products),
                         observation_space=(observation_space
                                            or parent.observation_space),
                         n_products=parent.n_products)
        self._prev_product = 1

    def _process_action(self, a):
        if a == 1 + self.n_products:
            return self._prev_product
        else:
            return a

    def step(self, action):
        a = self._process_action(action)
        if 0 < a < 1 + self.n_products:
            self._prev_product = a
        obs, reward, done, info = self.parent.step(a)
        self._state = self._process_observation(obs)
        self._state_info = None
        return (self._state, self._process_reward(reward), done, info)

    def _reset_hook(self):
        self._prev_product = 1

class EnvWrapperRotateAction(EnvWrapper):
    def __init__(self, parent, rotation=None, observation_space=None):
        super().__init__(parent,
                         action_space=spaces.Discrete(3),
                         observation_space=(observation_space
                                            or parent.observation_space))
        initial_product = 1
        self.rotation = rotation or range(1, self.n_products + 1)
        self.i_rotation = self.rotation.index(initial_product)

    def _process_action(self, a):
        if a == 0:
            return 0
        elif a == 1:
            return self.rotation[self.i_rotation]
        else:
            self.i_rotation = (self.i_rotation + 1) % len(self.rotation)
            return self.rotation[self.i_rotation]

class EnvWrapperLongStep(EnvWrapper):
    def __init__(self, parent, step_lengths, observation_space=None):
        super().__init__(parent,
                         action_space=spaces.Discrete(
                             1 + parent.n_products * len(step_lengths)),
                         observation_space=(observation_space
                                            or parent.observation_space))
        self.step_lengths = step_lengths
        self.gamma = 1.0 - self.discount_rate

    def _process_action(self, a):
        if a == 0:
            return a
        else:
            return 1 + ((a-1) // len(self.step_lengths))

    def step(self, action):
        a = self._process_action(action)
        repeat_count = (1 if a == 0
                        else self.step_lengths[(a-1) % len(self.step_lengths)])
        reward = 0.
        duration = 0.
        for i in range(repeat_count):
            obs, r, done, info = self.parent.step(a)
            reward += self.gamma ** duration * self._process_reward(r)
            duration += info.get('step_time', 1.0)
            self._state = self._process_observation(obs)
            self._state_info = None
            if done:
                break
        info['step_time'] = duration
        return (self._state, reward, done, info)

    def set_discount_rate(self, rate, time_constant_s):
        r = self.parent.set_discount_rate(rate, time_constant_s)
        self.gamma = 1.0 - r
        return r

# Run a base stock policy where the input actions indicate the product
# which is then automatically manufactured up to the product's base level
# before the next action choice comes up.
class EnvWrapperBaseStock(EnvWrapper):
    def __init__(self, parent, base_levels, arrival_trigger_levels=0,
                 action_space=None, observation_space=None):
        super().__init__(parent,
                         action_space=(action_space or parent.action_space),
                         observation_space=(observation_space
                                            or parent.observation_space),
                         n_products=parent.n_products)
        self._base_levels = base_levels
        self._arrival_trigger_levels = arrival_trigger_levels
        self.gamma = 1.0 - self.discount_rate

    def step(self, action):
        i_product = None
        if action > 0:
            i_product = action - 1
            base_level = self._base_levels[i_product]
            ql = self.decode_state().queue_levels
            if ql[i_product] >= base_level:
                action = 0
                i_product = None
        a = self._process_action(action)
        reward = 0.
        duration = 0.
        n_steps = 0
        while True:
            n_steps += 1
            obs, r, done, info = self.parent.step(a)
            reward += self.gamma ** duration * self._process_reward(r)
            duration += info.get('step_time', 1.0)
            self._state = self._process_observation(obs)
            self._state_info = None
            if done or i_product is None:
                 break
            prev_ql = ql
            ql = self.decode_state().queue_levels
            if (ql[i_product] >= base_level
                or (numpy.all(ql >= self._arrival_trigger_levels)
                    and numpy.any(ql < prev_ql))):
                break
        info['step_time'] = duration
        info['n_short_steps'] = n_steps
        return (self._state, reward, done, info)

    def _find_feasible_actions(self, prod, qlevels):
        return [0] + [i+1 for i,b in enumerate(self._base_levels)
                      if qlevels[i] < b]

    def set_discount_rate(self, rate, time_constant_s):
        r = self.parent.set_discount_rate(rate, time_constant_s)
        self.gamma = 1.0 - r
        return r

# Backorders are always filled at once, up to stock level 0.
# Otherwise production decisions are done in single steps.
class EnvWrapperStayOnBackorder(EnvWrapper):
    def __init__(self, parent, trigger_on_arrival=False,
                 action_space=None, observation_space=None):
        super().__init__(parent,
                         action_space=(action_space or parent.action_space),
                         observation_space=(observation_space
                                            or parent.observation_space),
                         n_products=parent.n_products)
        self._trigger_on_arrival = trigger_on_arrival
        self.gamma = 1.0 - self.discount_rate

    def step(self, action):
        i_product = action - 1 if action > 0 else None
        ql = self.decode_state().queue_levels
        i_others = [i != i_product for i in range(len(ql))]
        a = self._process_action(action)
        reward = 0.
        duration = 0.
        n_steps = 0
        while True:
            n_steps += 1
            obs, r, done, info = self.parent.step(a)
            reward += self.gamma ** duration * self._process_reward(r)
            duration += info.get('step_time', 1.0)
            self._state = self._process_observation(obs)
            self._state_info = None
            if done or i_product is None:
                 break
            prev_ql = ql
            ql = self.decode_state().queue_levels
            if (ql[i_product] >= 0
                or (self._trigger_on_arrival
                    and numpy.any(ql[i_others] < prev_ql[i_others]))):
                break
        info['step_time'] = duration
        info['n_short_steps'] = n_steps
        return (self._state, reward, done, info)

    def _find_feasible_actions(self, prod, qlevels):
        return [0] + [i+1 for i,b in enumerate(self._base_levels)
                      if qlevels[i] < b]

    def set_discount_rate(self, rate, time_constant_s):
        r = self.parent.set_discount_rate(rate, time_constant_s)
        self.gamma = 1.0 - r
        return r

def make_env(discount_per_h, order_scv=1.0,
             single_product=False, one_hot=True, mark_idle=False,
             visualize=False,
             breakpoints=None, xbreakpoints=None,
             piecewise_queues2=None, piecewise_tilewidth2=None,
             reseed=False, action=None, jaamsim=False,
             proxy=False, debug=False, logfilename=None):
    n_products = 3
    if order_scv == 1.0:
        modelfile = "lot-scheduling-control.cfg"
    elif order_scv == 0.25:
        modelfile = "lot-scheduling-control-025.cfg"
    else:
        raise NotImplementedError("unsupported order_scv value: {}"
                                  .format(order_scv))
    obs_space = spaces.Box(numpy.array([1, QMIN, QMIN, QMIN]),
                           numpy.array([n_products, QMAX, QMAX, QMAX]),
                           dtype=numpy.float32)
    if jaamsim:
        from gym_jaamsim.envs import JaamSimEnv, JaamSimEnvProxy
        constructor = JaamSimEnvProxy if proxy else JaamSimEnv
    else:
        from simulation import LotSchedulingEnv
        constructor = LotSchedulingEnv
    logfile = open(logfilename, 'w') if logfilename else None
    env = constructor(model=modelfile, observation_space=obs_space,
                      debug=debug, logfile=logfile)
    env.seed(1)
    env.set_discount_rate(discount_per_h, 3600)
    if visualize:
        env.render()
    setattr(env, 'n_products', n_products)
    setattr(env, 'n_product_obs', 1)
    env = EnvWrapperFloat(env, action_space=spaces.Discrete(1+n_products),
                          observation_space=obs_space)
    if reseed:
        env = EnvWrapperReseed(env)
    if single_product:
        env = EnvWrapperSingle(env)
    if one_hot:
        env = EnvWrapperProductFlags(env)
    if breakpoints is not None:
        env = EnvWrapperTiled(env, breakpoints=breakpoints,
                              xbreakpoints=xbreakpoints,
                              breakpoints2=piecewise_queues2,
                              tilewidth2=piecewise_tilewidth2)
    if mark_idle:
        env = EnvWrapperCurrentIdle(env)
    if action == None:
        pass
    elif action == "continue":
        env = EnvWrapperContinueAction(env)
    elif isinstance(action, tuple) and action[0] == "rotate":
        env = EnvWrapperRotateAction(env, action[1])
    elif isinstance(action, tuple) and action[0] == "long-step":
        env = EnvWrapperLongStep(env, action[1])
    elif isinstance(action, tuple) and action[0] == "base-stock":
        env = EnvWrapperBaseStock(env, action[1])
    elif action == "stay-on-backorder":
        env = EnvWrapperStayOnBackorder(env, trigger_on_arrival=False)
    elif action == "stay-on-backorder-or-new":
        env = EnvWrapperStayOnBackorder(env, trigger_on_arrival=True)
    else:
        raise ValueError("Unrecognized action: "+str(action))
    return env
