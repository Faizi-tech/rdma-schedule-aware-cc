import numpy as np
from sklearn.neural_network import MLPRegressor
import cnsim as cs

METHODS = {'DCQCN': 0, 'HPCC': 1, 'Barre-style': 2, 'ACC-style': 3, 'Ours': 4}
WARM = 2


def gen_comp(base, amp, per, phi, rho, sig, K, rng, spike_p=0.04):
    ar = 0.0
    out = np.zeros(K)
    for k in range(K):
        ar = rho * ar + sig * rng.standard_normal()
        v = base * (1.0 + amp * np.sin(2 * np.pi * k / per + phi) + ar)
        if rng.random() < spike_p:
            v *= 1.4
        out[k] = max(v, 0.3 * base)
    return out


def make_job(kind, n, size, base, rng, amp=0.12, per=4, rho=0.6, sig=0.03, offset=0.0):
    if kind == 'ar':
        nph = 2 * (n - 1)
        nfl = [n] * nph
        bsz = [size] * nph
    else:
        nph = 2
        nfl = [n] * 2
        bsz = [size] * 2
    return dict(kind=kind, n=n, size=size, base=base, nph=nph, nfl=nfl, bsz=bsz, amp=amp,
                per=per, rho=rho, sig=sig, offset=offset, phi=rng.uniform(0, 6.28))


def named_scenarios(rng):
    return {
        'S1 AllReduce x1': [make_job('ar', 8, 1.0e6, 12000, rng)],
        'S2 AllToAll x1': [make_job('a2a', 16, 4.0e6, 15000, rng)],
        'S3 Mixed x2': [make_job('ar', 8, 1.0e6, 14000, rng), make_job('a2a', 16, 3.0e6, 16000, rng, offset=3000)],
        'S4 Mixed x3': [make_job('ar', 8, 1.0e6, 10000, rng), make_job('a2a', 12, 3.0e6, 12000, rng, offset=2000),
                        make_job('ar', 6, 1.0e6, 8000, rng, offset=5000)],
    }


def sample_scenario(rng, shifted=False, nj=None):
    if nj is None:
        nj = int(rng.integers(1, 4))
    jobs = []
    for _ in range(nj):
        base = rng.uniform(6000, 20000)
        if shifted:
            per = int(rng.choice([5, 7, 8]))
            rho = rng.uniform(0.8, 0.95)
            sig = rng.uniform(0.04, 0.1)
        else:
            per = int(rng.choice([2, 3, 4, 6]))
            rho = rng.uniform(0.3, 0.8)
            sig = rng.uniform(0.01, 0.05)
        amp = rng.uniform(0.05, 0.2)
        if rng.random() < 0.5:
            n = int(rng.choice([4, 6, 8]))
            jobs.append(make_job('ar', n, rng.uniform(0.5, 1.5) * 1e6, base, rng, amp, per, rho, sig, rng.uniform(0, base)))
        else:
            n = int(rng.choice([8, 12, 16]))
            jobs.append(make_job('a2a', n, rng.uniform(2, 5) * 1e6, base, rng, amp, per, rho, sig, rng.uniform(0, base)))
    return jobs


# ------------------------------------------------------------------ predictors
W = 8


def pred_ma(comp, win=5, prior=14000.0):
    out = np.zeros(len(comp))
    for k in range(len(comp)):
        h = comp[max(0, k - win):k]
        out[k] = h.mean() if len(h) else prior
    return out


def train_mlp(seed=0, nseq=400, K=40, win=W):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(nseq):
        base = rng.uniform(6000, 20000)
        c = gen_comp(base, rng.uniform(0.05, 0.2), int(rng.choice([2, 3, 4, 6])), rng.uniform(0, 6.28),
                     rng.uniform(0.3, 0.8), rng.uniform(0.01, 0.05), K, rng)
        for k in range(win, K):
            h = c[k - win:k]
            m = h.mean()
            X.append(np.r_[h / m, k % 12 / 12.0 * 0])
            Y.append(c[k] / m)
    mdl = MLPRegressor(hidden_layer_sizes=(16,), max_iter=600, random_state=seed, learning_rate_init=0.01)
    mdl.fit(np.array(X), np.array(Y))
    return mdl


def pred_mlp(comp, mdl, win=W):
    out = pred_ma(comp)
    for k in range(win, len(comp)):
        h = comp[k - win:k]
        m = h.mean()
        out[k] = mdl.predict(np.r_[h / m, 0.0][None, :])[0] * m
    return out


def predict(comp, kind, mdl=None, noise=0.0, rng=None):
    if kind == 'oracle':
        p = comp.copy()
    elif kind == 'ma':
        p = pred_ma(comp)
    elif kind == 'mlp':
        p = pred_mlp(comp, mdl)
    else:
        p = np.zeros(len(comp))
    if noise > 0 and rng is not None and kind != 'none':
        p = p + rng.normal(0, noise, len(comp))
    return np.maximum(p, 0.0)


# ------------------------------------------------------------------ build and run
def build(jobs, K, seed):
    rng = np.random.default_rng(seed)
    nj = len(jobs)
    nph = np.zeros(nj, np.int64)
    nfl = np.zeros((nj, cs.PM), np.int64)
    bsz = np.zeros((nj, cs.PM))
    comp = np.zeros((nj, K))
    offs = np.zeros(nj)
    for j, jb in enumerate(jobs):
        nph[j] = jb['nph']
        for p in range(jb['nph']):
            nfl[j, p] = jb['nfl'][p]
            bsz[j, p] = jb['bsz'][p]
        comp[j] = gen_comp(jb['base'], jb['amp'], jb['per'], jb['phi'], jb['rho'], jb['sig'], K, rng)
        offs[j] = jb['offset']
    return dict(nj=nj, nph=nph, nfl=nfl, bsz=bsz, comp=comp, offs=offs, K=K, seed=seed)


def run(cfg, method, pred='none', theta=None, mdl=None, usephase=1, anchor_only=0, noise=0.0, trace=0, seed=None):
    nj, K = cfg['nj'], cfg['K']
    mode = METHODS[method]
    prng = np.random.default_rng(cfg['seed'] + 7)
    pc = np.zeros((nj, K))
    pmode = 0
    if mode == 4 and pred == 'sched':
        pmode = 2
    elif mode == 4 and pred != 'none':
        pmode = 1
        for j in range(nj):
            pc[j] = predict(cfg['comp'][j], pred, mdl, noise, prng)
    if theta is None:
        theta = np.zeros(cs.NPAR)
    sd = cfg['seed'] if seed is None else seed
    cct, itt, fct, st, tt, tq, tr, ideal = cs.simulate(
        nj, cfg['nph'], cfg['nfl'], cfg['bsz'], cfg['comp'], pc, cfg['offs'], K, mode, pmode,
        theta, usephase, anchor_only, sd, trace)
    sl = cct[:, WARM:] / ideal[:, None]
    ittn = itt[:, WARM:]
    res = dict(ideal=ideal.mean(), cct_sd=sl.mean(), p99_sd=np.percentile(sl, 99), itt=ittn.mean(), pfc=100 * st[0] / st[6],
               pfc_ev=st[1], qavg=st[2], qmax=st[3], jain=st[4], util=st[5],
               fct=fct.mean() if len(fct) else 0.0, sl=sl.ravel())
    if trace:
        res['trace'] = (tt, tq, tr)
    return res
