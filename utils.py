import numpy as np
import math, datetime

def fmt(a, f="{:.3f}"):
    return [f.format(x) for x in a]

def format_observations(observations):
    obs_mean = np.mean(observations, axis=0)
    obs_min = np.min(observations, axis=0)
    obs_max = np.max(observations, axis=0)
    items = ["{{{}: {:.3f} [{:.3f}, {:.3f}]}}".format(*args)
             for args in zip(range(len(obs_mean)), obs_mean, obs_min, obs_max)]
    return "observations: " + ", ".join(items)

def format_actions(actions):
    try:
        counts = np.bincount(np.array(actions, dtype=int).flatten())
    except ValueError:
        return ""
    return "action counts: " + str(dict(zip(range(len(counts)), counts)))

def format_losses(losses, divisor):
    return ' '.join(["{}={:.5f}".format(k, losses[k]/divisor)
                     for k in sorted(losses.keys())])

def join_episode_totals(episode_totals):
    joined = dict()
    for key in episode_totals[0].keys():
        joined[key] = [item for et in episode_totals
                       for item in et[key]]
    return joined

def print_episode_totals(episode_totals):
    rewards = np.array(episode_totals['reward'], dtype=np.float64)
    durations = np.array(episode_totals['duration'], dtype=np.float64)
    mean_rewards = rewards / (durations / (60 * 60))
    print('reward rate mean={}, stdev={}'.format(
        np.mean(mean_rewards), np.std(mean_rewards)))

def make_breakpoints(bottom, top, min_abs_step, min_rel_step=0):
    def make_bp_seq(bp, end):
        bps = []
        while bp < end:
            bps.append(bp)
            bp += int(max(math.ceil(bp * min_rel_step),
                          min_abs_step))
        return bps
    assert bottom < 0 and top > 0
    up = make_bp_seq(min_abs_step, top)
    down = [-p for p in make_bp_seq(min_abs_step, -bottom)]
    return [bottom] + list(reversed(down)) + [0] + up + [top]

def iso8601stamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
