import numpy

def draw_ac(env, predict_on_state, id):
    import matplotlib
    matplotlib.use('Cairo')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['lines.linewidth'] = .5
    n_products = env.n_products
    n_actions = env.action_space.n
    fig, axs = plt.subplots(n_products, 1)
    vvs = dict()
    ql = numpy.array(range(-100, 101))
    for p in range(n_products):
        ax = axs[p] if n_products > 1 else axs
        pv = numpy.zeros(ql.shape + (n_actions,))
        vv = numpy.zeros(ql.shape)
        obs_raw = numpy.zeros(1+n_products, dtype=numpy.float32)
        obs_raw[0] = p + 1
        for i,j in enumerate(ql):
            obs_raw[1 + p] = j
            obs_proc = env.process_observation(obs_raw)
            pred_p, pred_v = predict_on_state(obs_proc)
            if pred_p:
                pv[i,:] = numpy.array(pred_p, dtype=numpy.float32)
            vv[i] = pred_v
        vvs[p] = vv
        for a in range(n_actions):
            ax.plot(ql, pv[:,a].flatten(), '-', clip_on=False, markevery=1,
                    label="action={}".format(a))
        ax.set(title="current={}".format(obs_raw[0]))
        ax.grid()
        ax.legend()
    fig.savefig("action_prob.{}.pdf".format(id))
    plt.show()

    fig, ax = plt.subplots()
    for p in range(n_products):
        ax.plot(ql, vvs[p], '-', clip_on=False, markevery=1,
                label="current={}".format(p))
    ax.grid()
    ax.legend()
    fig.savefig("values.{}.pdf".format(id))
    plt.show()
