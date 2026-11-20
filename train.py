import sys, json, time, pickle, numpy as np
import scen, cnsim as cs

VARIANTS = {
 'full_mlp':   dict(pred='mlp',   usephase=1, reward='cct'),
 'ma':         dict(pred='ma',    usephase=1, reward='cct'),
 'oracle':     dict(pred='oracle',usephase=1, reward='cct'),
 'sched':      dict(pred='sched', usephase=1, reward='cct'),
 'nopred':     dict(pred='none',  usephase=1, reward='cct'),
 'fct_reward': dict(pred='mlp',   usephase=1, reward='fct'),
 'nophase':    dict(pred='mlp',   usephase=0, reward='cct'),
}

def fitness(theta, cfgs, v, mdl):
    f = 0.0
    for cfg in cfgs:
        r = scen.run(cfg, 'Ours', pred=v['pred'], theta=theta, mdl=mdl, usephase=v['usephase'])
        perf = r['cct_sd'] if v['reward'] == 'cct' else r['fct'] / r['ideal']
        f += perf + 0.05 * r['pfc']
    return -f / len(cfgs)

def make_cfgs(seed, n, K=10, shifted=False):
    rng = np.random.default_rng(seed)
    return [scen.build(scen.sample_scenario(rng, shifted), K, int(rng.integers(1, 10**6))) for _ in range(n)]

def train(name, gens=30, pairs=8, sigma=0.1, lr=0.06, seed=0):
    v = VARIANTS[name]
    mdl = pickle.load(open('mlp_pred.pkl', 'rb'))
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.05, cs.NPAR)
    m = np.zeros_like(theta); vv = np.zeros_like(theta)
    val = make_cfgs(999, 6, K=8)
    best, bestf = theta.copy(), fitness(theta, val, v, mdl)
    curve = []
    t0 = time.time()
    for g in range(gens):
        cfgs = make_cfgs(1000 + g, 6, K=8)
        eps = rng.standard_normal((pairs, cs.NPAR))
        fp = np.array([fitness(theta + sigma * e, cfgs, v, mdl) for e in eps])
        fn = np.array([fitness(theta - sigma * e, cfgs, v, mdl) for e in eps])
        allf = np.r_[fp, fn]
        ranks = np.argsort(np.argsort(allf)).astype(float) / (len(allf) - 1) - 0.5
        rp, rn = ranks[:pairs], ranks[pairs:]
        grad = ((rp - rn)[:, None] * eps).sum(0) / (2 * pairs * sigma)
        m = 0.9 * m + 0.1 * grad; vv = 0.999 * vv + 0.001 * grad ** 2
        mh = m / (1 - 0.9 ** (g + 1)); vh = vv / (1 - 0.999 ** (g + 1))
        theta = theta + lr * mh / (np.sqrt(vh) + 1e-8)
        fv = fitness(theta, val, v, mdl)
        curve.append((g, fitness(theta, cfgs, v, mdl), fv))
        if fv > bestf:
            bestf, best = fv, theta.copy()
        print(name, g, round(curve[-1][1], 4), round(fv, 4), round(time.time() - t0), flush=True)
    np.savez(f'theta_{name}.npz', theta=best, curve=np.array(curve), bestf=bestf)

if __name__ == '__main__':
    if sys.argv[1] == 'mlp':
        mdl = scen.train_mlp(0)
        pickle.dump(mdl, open('mlp_pred.pkl', 'wb')); print('mlp done')
    else:
        for n in sys.argv[1:]:
            train(n)
