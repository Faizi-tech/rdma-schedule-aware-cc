import sys, json, pickle, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import scen, cnsim as cs
from train import VARIANTS

os.makedirs('figs', exist_ok=True)
K = 14
mdl = pickle.load(open('mlp_pred.pkl', 'rb'))
TH = {n: np.load(f'theta_{n}.npz')['theta'] for n in VARIANTS if os.path.exists(f'theta_{n}.npz')}
R = {}


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def runv(cfg, name, **kw):
    """name: baseline method or learned variant key"""
    if name in scen.METHODS and name != 'Ours':
        return scen.run(cfg, name)
    if name == 'Rule (schedule only)':
        return scen.run(cfg, 'Ours', pred='sched', anchor_only=1)
    if name == 'Rule (predicted, no RL)':
        return scen.run(cfg, 'Ours', pred='mlp', mdl=mdl, anchor_only=1)
    key = {'Ours': 'full_mlp', 'Ours-sched': 'sched'}.get(name, name)
    v = VARIANTS[key]
    th = TH[key]
    return scen.run(cfg, 'Ours', pred=v['pred'], theta=th, mdl=mdl, usephase=v['usephase'], **kw)


def summarize(rows):
    out = {}
    for k in ['cct_sd', 'p99_sd', 'itt', 'pfc', 'jain', 'util', 'fct']:
        a = np.array([r[k] for r in rows])
        out[k] = (float(a.mean()), boot_ci(a))
    return out


# ---------------------------------------------------------------- E1 predictor accuracy
def e1():
    res = {}
    for fam in ['in-family', 'shifted']:
        rng = np.random.default_rng(123 if fam == 'in-family' else 456)
        seqs = []
        for _ in range(150):
            base = rng.uniform(6000, 20000)
            if fam == 'in-family':
                c = scen.gen_comp(base, rng.uniform(.05, .2), int(rng.choice([2, 3, 4, 6])), rng.uniform(0, 6.28),
                                  rng.uniform(.3, .8), rng.uniform(.01, .05), 40, rng)
            else:
                c = scen.gen_comp(base, rng.uniform(.05, .2), int(rng.choice([5, 7, 8])), rng.uniform(0, 6.28),
                                  rng.uniform(.8, .95), rng.uniform(.04, .1), 40, rng)
            seqs.append(c)
        d = {}
        for nm in ['ma', 'mlp']:
            e = [np.abs(scen.predict(c, nm, mdl) - c)[8:] / c[8:] for c in seqs]
            d[nm] = 100 * float(np.concatenate(e).mean())
        d['oracle'] = 0.0
        res[fam] = d
    R['e1'] = res
    return res


def e1_window():
    sw = {}
    for w in [2, 4, 8, 16]:
        m = scen.train_mlp(1, win=w)
        rng = np.random.default_rng(123)
        errs = []
        for _ in range(100):
            base = rng.uniform(6000, 20000)
            c = scen.gen_comp(base, rng.uniform(.05, .2), int(rng.choice([2, 3, 4, 6])), rng.uniform(0, 6.28),
                              rng.uniform(.3, .8), rng.uniform(.01, .05), 40, rng)
            for k in range(16, 40):
                h = c[k - w:k]; mm = h.mean()
                x = np.r_[h / mm, 0.0][None, :]
                errs.append(abs(m.predict(x)[0] * mm - c[k]) / c[k])
        sw[w] = 100 * float(np.mean(errs))
    R['e1_window'] = sw
    return sw


# ---------------------------------------------------------------- E2 main comparison
METH = ['DCQCN', 'HPCC', 'Barre-style', 'ACC-style', 'Rule (schedule only)', 'Ours', 'Ours-sched']


def e2():
    out = {}
    named = {}
    for s in range(10):
        sc = scen.named_scenarios(np.random.default_rng(100 + s))
        for nm, jobs in sc.items():
            named.setdefault(nm, []).append(scen.build(jobs, K, 5000 + s))
    allrows = {m: [] for m in METH}
    for nm, cfgs in named.items():
        out[nm] = {}
        for m in METH:
            rows = [runv(c, m) for c in cfgs]
            out[nm][m] = summarize(rows)
            allrows[m] += rows
    # random held-out sets
    for fam, shifted, seed in [('random', False, 777), ('shifted', True, 888)]:
        rng = np.random.default_rng(seed)
        cfgs = [scen.build(scen.sample_scenario(rng, shifted), K, int(rng.integers(1, 10**6))) for _ in range(40)]
        out[fam] = {}
        rr = {}
        for m in METH:
            rr[m] = [runv(c, m) for c in cfgs]
            out[fam][m] = summarize(rr[m])
        out[fam]['pooled_sl'] = {m: np.concatenate([r['sl'] for r in rr[m]]).tolist() for m in METH}
        for tgt, key in [('Ours', 'stats'), ('Ours-sched', 'stats_sched')]:
            st = {}
            o = np.array([r['cct_sd'] for r in rr[tgt]])
            for m in METH:
                if m == tgt:
                    continue
                b = np.array([r['cct_sd'] for r in rr[m]])
                try:
                    p = wilcoxon(o, b).pvalue
                except Exception:
                    p = 1.0
                st[m] = dict(mean_gain_pct=float(100 * (b.mean() - o.mean()) / b.mean()), p=float(p),
                             wins=int((o < b - 1e-9).sum()), n=len(o))
            out[fam][key] = st
    R['e2'] = out
    return out


# ---------------------------------------------------------------- E3 ablations
def e3():
    names = ['DCQCN', 'Rule (schedule only)', 'Rule (predicted, no RL)', 'nopred', 'sched', 'ma', 'oracle',
             'fct_reward', 'nophase', 'full_mlp']
    rng = np.random.default_rng(777)
    cfgs = [scen.build(scen.sample_scenario(rng, False), K, int(rng.integers(1, 10**6))) for _ in range(40)]
    out = {}
    for n in names:
        rows = [runv(c, n) for c in cfgs]
        out[n] = summarize(rows)
    R['e3'] = out
    return out


# ---------------------------------------------------------------- E4 sensitivity
def multi_cfgs(seed, n=20, scale=1.0, sig=None):
    rng = np.random.default_rng(seed)
    cfgs = []
    for _ in range(n):
        jobs = scen.sample_scenario(rng, False, nj=int(rng.integers(2, 4)))
        for jb in jobs:
            jb['base'] *= scale
            jb['offset'] *= scale
            if sig is not None:
                jb['sig'] = sig
        cfgs.append(scen.build(jobs, K, int(rng.integers(1, 10**6))))
    return cfgs


def e4():
    out = {}
    cf = multi_cfgs(2024)
    out['pred_noise'] = {}
    for nz in [0, 200, 500, 1000, 2000, 5000]:
        rows = [scen.run(c, 'Ours', pred='mlp', theta=TH['full_mlp'], mdl=mdl, noise=nz) for c in cf]
        out['pred_noise'][nz] = float(np.mean([r['cct_sd'] for r in rows]))
    out['pred_noise']['DCQCN'] = float(np.mean([scen.run(c, 'DCQCN')['cct_sd'] for c in cf]))
    out['pred_noise']['HPCC'] = float(np.mean([scen.run(c, 'HPCC')['cct_sd'] for c in cf]))
    out['load'] = {}
    for sc_ in [0.25, 0.5, 1.0, 2.0]:
        cf2 = multi_cfgs(2025, scale=sc_)
        out['load'][sc_] = {m: float(np.mean([runv(c, m)['cct_sd'] for c in cf2])) for m in ['DCQCN', 'HPCC', 'Barre-style', 'Ours', 'Ours-sched']}
    out['jitter'] = {}
    for sg in [0.01, 0.05, 0.1, 0.2]:
        cf3 = multi_cfgs(2026, sig=sg)
        out['jitter'][sg] = {m: float(np.mean([runv(c, m)['cct_sd'] for c in cf3])) for m in ['DCQCN', 'HPCC', 'Ours', 'Ours-sched']}
        out['jitter'][sg]['Ours-ma'] = float(np.mean([runv(c, 'ma')['cct_sd'] for c in cf3]))
    R['e4'] = out
    return out


# ---------------------------------------------------------------- E5 quantization and budget
def quant(theta, bits=8):
    q = 2 ** (bits - 1) - 1
    s = np.abs(theta).max() / q
    return np.round(theta / s) * s


def e5():
    rng = np.random.default_rng(777)
    cfgs = [scen.build(scen.sample_scenario(rng, False), K, int(rng.integers(1, 10**6))) for _ in range(40)]
    th = TH['full_mlp']
    out = {}
    for lab, t in [('float64', th), ('int8', quant(th, 8)), ('int6', quant(th, 6)), ('int4', quant(th, 4))]:
        rows = [scen.run(c, 'Ours', pred='mlp', theta=t, mdl=mdl) for c in cfgs]
        out[lab] = dict(cct=float(np.mean([r['cct_sd'] for r in rows])), pfc=float(np.mean([r['pfc'] for r in rows])))
    out['params'] = int(cs.NPAR)
    out['macs'] = int(cs.NF * cs.HID + cs.HID)
    out['bytes_int8'] = int(cs.NPAR)
    R['e5'] = out
    return out


# ---------------------------------------------------------------- E6 sanity and traces
def e6():
    # long-flow validation of the DCQCN model: 8 long flows
    jb = [scen.make_job('a2a', 8, 6.0e7, 100.0, np.random.default_rng(0), amp=0, sig=0, rho=0)]
    cfg = scen.build(jb, 3, 1)
    r = scen.run(cfg, 'DCQCN', trace=4000)
    tt, tq, tr = r['trace']
    R['e6'] = dict(qavg_KB=float(tq[len(tq) // 3:].mean() / 1e3), util=float(r['util']), jain=float(r['jain']),
                   rate_frac=float(tr[len(tr) // 3:].mean() / cs.CAP * 8))
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.4))
    ax[0].plot(tt / 1e3, tq / 1e3); ax[0].axhline(100, ls='--', c='gray'); ax[0].axhline(400, ls='--', c='gray')
    ax[0].set_xlabel('time (ms)'); ax[0].set_ylabel('queue (KB)'); ax[0].set_title('DCQCN, 8 long flows')
    ax[1].plot(tt / 1e3, tr / cs.CAP * 8); ax[1].set_xlabel('time (ms)'); ax[1].set_ylabel('rate / fair share'); ax[1].set_title('mean per-flow rate')
    plt.tight_layout(); plt.savefig('figs/validation.png', dpi=200); plt.close()
    # like-for-like burst trace on S2 (single job, identical start time for all methods)
    sc = scen.named_scenarios(np.random.default_rng(100))
    cfg = scen.build(sc['S2 AllToAll x1'], K, 5000)
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
    tr_stats = {}
    for m, c in [('DCQCN', 'tab:red'), ('HPCC', 'tab:green'), ('Barre-style', 'tab:orange'), ('Ours', 'tab:blue')]:
        rr = runv(cfg, m, trace=8000) if m.startswith('Ours') else scen.run(cfg, m, trace=8000)
        tt, tq, tr = rr['trace']
        t0 = cfg['comp'][0, 0]
        sel = (tt > t0 - 200) & (tt < t0 + 9000)
        ax[0].plot((tt[sel] - t0) / 1e3, tq[sel] / 1e3, label=m, c=c, lw=0.9)
        ax[1].plot((tt[sel] - t0) / 1e3, tr[sel] / cs.CAP, c=c, lw=0.9)
        tr_stats[m] = dict(qpeak_KB=float(tq[sel].max() / 1e3), qmean_KB=float(tq[sel].mean() / 1e3))
    R['trace_stats'] = tr_stats
    ax[0].set_ylabel('queue (KB)'); ax[1].set_ylabel('mean rate / line'); ax[1].set_xlabel('time since burst start (ms)')
    ax[0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig('figs/trace.png', dpi=200); plt.close()


# ---------------------------------------------------------------- plots
def plots():
    global R
    R = json.loads(json.dumps(R, default=float))
    e2 = R['e2']
    names = ['S1 AllReduce x1', 'S2 AllToAll x1', 'S3 Mixed x2', 'S4 Mixed x3', 'random', 'shifted']
    labels = ['S1', 'S2', 'S3', 'S4', 'Rand', 'Shift']
    fig, ax = plt.subplots(figsize=(7, 2.6))
    w = 0.115
    for i, m in enumerate(METH):
        v = [e2[n][m]['cct_sd'][0] for n in names]
        lo = [e2[n][m]['cct_sd'][0] - e2[n][m]['cct_sd'][1][0] for n in names]
        hi = [e2[n][m]['cct_sd'][1][1] - e2[n][m]['cct_sd'][0] for n in names]
        ax.bar(np.arange(len(names)) + (i - 3) * w, v, w, yerr=[lo, hi], label=m, capsize=1.5)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(labels); ax.set_ylabel('mean CCT slowdown')
    ax.legend(fontsize=6, ncol=3); plt.tight_layout(); plt.savefig('figs/main_cct.png', dpi=200); plt.close()
    # CDF
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for m in ['DCQCN', 'HPCC', 'Barre-style', 'ACC-style', 'Ours', 'Ours-sched']:
        x = np.sort(e2['random']['pooled_sl'][m]); ax.plot(x, np.arange(len(x)) / len(x), label=m)
    ax.set_xlim(1, 6); ax.set_xlabel('CCT slowdown'); ax.set_ylabel('CDF'); ax.legend(fontsize=6)
    plt.tight_layout(); plt.savefig('figs/cdf.png', dpi=200); plt.close()
    # PFC bars
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for i, m in enumerate(METH):
        ax.bar(i, e2['random'][m]['pfc'][0], yerr=[[e2['random'][m]['pfc'][0] - e2['random'][m]['pfc'][1][0]], [e2['random'][m]['pfc'][1][1] - e2['random'][m]['pfc'][0]]], capsize=2)
    ax.set_xticks(range(len(METH))); ax.set_xticklabels(['DCQCN', 'HPCC', 'Barre', 'ACC', 'Rule', 'Ours', 'Ours\nsched'], fontsize=6)
    ax.set_ylabel('PFC paused time (%)'); plt.tight_layout(); plt.savefig('figs/pfc.png', dpi=200); plt.close()
    # ablation
    e3 = R['e3']
    keys = list(e3.keys())
    lab = ['DCQCN', 'Rule\nsched', 'Rule\npred', 'No\npred', 'Sched\nonly', 'MA', 'Oracle', 'FCT\nreward', 'No\nphase', 'Full']
    fig, ax = plt.subplots(figsize=(7, 2.5))
    v = [e3[k]['cct_sd'][0] for k in keys]
    err = [[e3[k]['cct_sd'][0] - e3[k]['cct_sd'][1][0] for k in keys], [e3[k]['cct_sd'][1][1] - e3[k]['cct_sd'][0] for k in keys]]
    ax.bar(range(len(keys)), v, yerr=err, capsize=2, color=['gray'] + ['tab:orange'] * 2 + ['tab:blue'] * 7)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(lab, fontsize=7); ax.set_ylabel('mean CCT slowdown')
    lo = min(v) - 0.1; ax.set_ylim(lo, max(v) + 0.1)
    plt.tight_layout(); plt.savefig('figs/ablation.png', dpi=200); plt.close()
    # sensitivity
    e4 = R['e4']
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    xs = ['0', '200', '500', '1000', '2000', '5000']
    ax[0].plot(range(6), [e4['pred_noise'][x] for x in xs], 'o-', label='Ours')
    ax[0].axhline(e4['pred_noise']['DCQCN'], c='r', ls='--', label='DCQCN'); ax[0].axhline(e4['pred_noise']['HPCC'], c='g', ls='--', label='HPCC')
    ax[0].set_xticks(range(6)); ax[0].set_xticklabels([str(x) for x in xs], fontsize=6); ax[0].set_xlabel('forecast error (us)'); ax[0].set_ylabel('mean CCT slowdown'); ax[0].legend(fontsize=6)
    ls = sorted(e4['load'].keys(), key=float)
    for m in ['DCQCN', 'HPCC', 'Barre-style', 'Ours', 'Ours-sched']:
        ax[1].plot(range(len(ls)), [e4['load'][l][m] for l in ls], 'o-', label=m)
    ax[1].set_xticks(range(len(ls))); ax[1].set_xticklabels([str(l) for l in ls], fontsize=6); ax[1].set_xlabel('compute time scale'); ax[1].legend(fontsize=6)
    js = sorted(e4['jitter'].keys(), key=float)
    for m in ['DCQCN', 'HPCC', 'Ours', 'Ours-sched', 'Ours-ma']:
        ax[2].plot(range(len(js)), [e4['jitter'][j][m] for j in js], 'o-', label=m)
    ax[2].set_xticks(range(len(js))); ax[2].set_xticklabels([str(j) for j in js], fontsize=6); ax[2].set_xlabel('compute jitter sigma'); ax[2].legend(fontsize=6)
    plt.tight_layout(); plt.savefig('figs/sensitivity.png', dpi=200); plt.close()
    # training curves
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for n in TH:
        c = np.load(f'theta_{n}.npz')['curve']
        ax.plot(c[:, 0], c[:, 2], label=n, lw=0.9)
    ax.set_xlabel('ES generation'); ax.set_ylabel('validation fitness'); ax.legend(fontsize=5)
    plt.tight_layout(); plt.savefig('figs/training.png', dpi=200); plt.close()
    # predictor
    e1 = R['e1']
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.2))
    x = np.arange(3)
    for i, fam in enumerate(['in-family', 'shifted']):
        ax[0].bar(x + i * 0.35, [e1[fam]['ma'], e1[fam]['mlp'], e1[fam]['oracle']], 0.35, label=fam)
    ax[0].set_xticks(x + 0.17); ax[0].set_xticklabels(['Moving avg', 'Lag-window MLP', 'Oracle']); ax[0].set_ylabel('MAPE of next compute time (%)'); ax[0].legend(fontsize=6)
    sw = R['e1_window']
    ax[1].plot([int(k) for k in sw], list(sw.values()), 'o-'); ax[1].set_xlabel('history window'); ax[1].set_ylabel('MAPE (%)')
    plt.tight_layout(); plt.savefig('figs/predictor.png', dpi=200); plt.close()


if __name__ == '__main__':
    steps = sys.argv[1:] or ['e1', 'e1w', 'e2', 'e3', 'e4', 'e5', 'e6', 'plots']
    if os.path.exists('results.json'):
        R.update(json.load(open('results.json')))
    for s in steps:
        {'e1': e1, 'e1w': e1_window, 'e2': e2, 'e3': e3, 'e4': e4, 'e5': e5, 'e6': e6, 'plots': plots}[s]()
        json.dump(R, open('results.json', 'w'), default=float)
        print('done', s, flush=True)
