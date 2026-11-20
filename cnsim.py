"""
cnsim.py : flow-level simulator of a shared RDMA bottleneck carrying periodic
collective (AI training) traffic. Units: bytes and microseconds.

Controllers (mode): 0 DCQCN, 1 HPCC-style, 2 Barre-style, 3 ACC-style adaptive ECN,
                    4 learned policy (predictive anchor + MLP correction)
"""
import numpy as np
from numba import njit

FM, PM, JM = 16, 16, 4          # max flows per job, phases per job, jobs
DT, CAP, RL, RMIN = 5.0, 12500.0, 12500.0, 12.5   # step (us), link 100 Gbps, NIC 100 Gbps, min rate
RTT, D = 20.0, 4                # feedback delay of D steps
XOFF, XON = 1.5e6, 1.0e6        # PFC thresholds (bytes)
KMIN0, KMAX0, PMAXM = 1.0e5, 4.0e5, 0.2
PKT = 4096.0
RAI, GAI = 5.0, 1.0 / 16.0
RAIH, ETA = 25.0, 0.95
CTRL = 4                        # learned policy control interval in steps (20 us)
NF = 8                          # number of policy features
HID = 12
NPAR = NF * HID + HID + HID + 1

RC, RT, AL, STG, TINC, TAL, CG, CON, REM, QI, DREM, ACT, LAST, ECN, FST, EFF, HR, CUT = range(18)


@njit(cache=True)
def start_phase(j, p, t, nfl, bsz, S):
    for f in range(nfl[j, p]):
        i = j * FM + f
        S[i, ACT] = 1.0
        S[i, REM] = bsz[j, p]
        S[i, DREM] = bsz[j, p]
        S[i, QI] = 0.0
        S[i, FST] = t
        if p == 0:
            idle = t - S[i, LAST]
            if idle > 500.0:
                S[i, RC] = RL
                S[i, RT] = RL
                S[i, AL] = 0.5
                S[i, STG] = 0.0
                S[i, HR] = RL
                S[i, EFF] = RL
                S[i, CON] = 0.0
                S[i, CUT] = 0.0


@njit(cache=True)
def simulate(nj, nph, nfl, bsz, comp, pcomp, offs, K, mode, pmode, theta, usephase,
             anchor_only, seed, trace_n):
    np.random.seed(seed)
    n = nj * FM
    S = np.zeros((n, 18))
    for i in range(n):
        S[i, RC] = RL
        S[i, RT] = RL
        S[i, AL] = 0.5
        S[i, CG] = 1000.0
        S[i, LAST] = -1e9
        S[i, EFF] = RL
        S[i, HR] = RL
    jp = np.zeros(nj, np.int64)
    jst = np.zeros(nj, np.int64)
    jstart = np.zeros(nj)
    jk = np.zeros(nj, np.int64)
    jcs = np.zeros(nj)
    jib = np.zeros(nj)
    pws = np.zeros(nj)
    pwe = np.zeros(nj)
    fmaxj = np.zeros(nj)
    idealj = np.zeros(nj)
    cct = np.zeros((nj, K))
    itt = np.zeros((nj, K))
    fct = np.zeros(60000)
    nfct = 0
    for j in range(nj):
        tot = 0.0
        mx = 0
        for p in range(nph[j]):
            tot += nfl[j, p] * bsz[j, p]
            if nfl[j, p] > mx:
                mx = nfl[j, p]
        fmaxj[j] = mx
        idealj[j] = tot / CAP
        jstart[j] = offs[j] + comp[j, 0]
        jib[j] = offs[j]
        pws[j] = offs[j] + pcomp[j, 0]
        pwe[j] = pws[j] + 1.3 * idealj[j]

    pbuf = np.zeros(D)
    qbuf = np.zeros(D)
    obuf = np.zeros(D)
    ptr = 0
    Q = 0.0
    paused = 0
    pfc_t = 0.0
    pfc_ev = 0
    qsum = 0.0
    qmax = 0.0
    nsteps = 0
    jsum = 0.0
    jn = 0
    busy_t = 0.0
    out_tot = 0.0
    s_acc = 1.0
    acc_q = 0.0
    acc_o = 0.0
    acc_n = 0
    tr_q = np.zeros(trace_n + 1)
    tr_r = np.zeros(trace_n + 1)
    tr_t = np.zeros(trace_n + 1)
    ntr = 0
    t = 0.0
    step = 0
    H = HID
    while True:
        done = True
        for j in range(nj):
            if jst[j] != 2:
                done = False
        if done:
            break
        for j in range(nj):
            if jst[j] == 0 and t >= jstart[j]:
                jst[j] = 1
                jp[j] = 0
                jcs[j] = t
                start_phase(j, 0, t, nfl, bsz, S)
        nact = 0
        for i in range(n):
            if S[i, ACT] > 0.5:
                nact += 1
        if nact == 0 and Q < 1.0:
            tn = 1e18
            for j in range(nj):
                if jst[j] == 0 and jstart[j] < tn:
                    tn = jstart[j]
            if tn > t and tn < 1e17:
                t = tn
            Q = 0.0
            paused = 0
            for k in range(D):
                pbuf[k] = 0.0
                qbuf[k] = 0.0
                obuf[k] = 0.0
            continue

        # ---------------- learned policy update ----------------
        newflow = False
        if mode == 4:
            for i in range(n):
                if S[i, ACT] > 0.5 and S[i, FST] == t:
                    newflow = True
        if mode == 4 and (step % CTRL == 0 or newflow):
            cown = np.zeros(nj)
            wnow = np.zeros(nj)
            wahd = np.zeros(nj)
            for j in range(nj):
                c = 0.0
                for f in range(FM):
                    if S[j * FM + f, ACT] > 0.5:
                        c += 1.0
                cown[j] = c
                if pmode == 1 and jst[j] != 2:
                    if t >= pws[j] and t <= pwe[j]:
                        wnow[j] = fmaxj[j]
                    if t + 300.0 >= pws[j] and t + 300.0 <= pwe[j]:
                        wahd[j] = fmaxj[j]
            swn = 0.0
            swa = 0.0
            for j in range(nj):
                swn += wnow[j]
                swa += wahd[j]
            qd = qbuf[(ptr + 1) % D]
            for j in range(nj):
                if jst[j] != 1:
                    continue
                for f in range(FM):
                    i = j * FM + f
                    if S[i, ACT] < 0.5:
                        continue
                    nnow = max(cown[j] + swn - wnow[j], 1.0)
                    nahd = max(cown[j] + swa - wahd[j], 1.0)
                    if pmode >= 1:
                        base = min(RL, CAP / nnow)
                    else:
                        base = S[i, RC]
                    a = 0.0
                    if anchor_only == 0:
                        x0 = S[i, EFF] / RL
                        x1 = S[i, ECN]
                        x2 = min(qd / XOFF, 2.0)
                        x3 = S[i, REM] / max(bsz[j, jp[j]], 1.0)
                        x4 = base / RL
                        x5 = 0.0
                        x7 = 0.0
                        if pmode >= 1:
                            x5 = nahd / 16.0
                            x7 = (nahd - nnow) / 16.0
                        x6 = 0.0
                        if usephase == 1:
                            x6 = min((t - jcs[j]) / 1000.0, 1.0)
                        z = theta[NF * H + H + H]
                        for h in range(H):
                            u = (theta[0 * H + h] * x0 + theta[1 * H + h] * x1 + theta[2 * H + h] * x2
                                 + theta[3 * H + h] * x3 + theta[4 * H + h] * x4 + theta[5 * H + h] * x5
                                 + theta[6 * H + h] * x6 + theta[7 * H + h] * x7 + theta[NF * H + h])
                            z += theta[NF * H + H + h] * np.tanh(u)
                        a = np.tanh(z)
                    e = base * np.exp(1.5 * a)
                    if e > RL:
                        e = RL
                    if e < RMIN:
                        e = RMIN
                    S[i, EFF] = e

        # ---------------- sending ----------------
        kmin = KMIN0
        kmax = KMAX0
        if mode == 3:
            kmin = KMIN0 * s_acc
            kmax = KMAX0 * s_acc
        qin = 0.0
        ract = 0.0
        r2 = 0.0
        na = 0
        for i in range(n):
            if S[i, ACT] > 0.5:
                if mode == 1:
                    r = S[i, HR]
                elif mode == 4:
                    r = S[i, EFF]
                else:
                    r = S[i, RC]
                if paused == 1:
                    r = 0.0
                sent = r * DT
                if sent > S[i, REM]:
                    sent = S[i, REM]
                S[i, REM] -= sent
                S[i, QI] += sent
                qin += sent
                if r > 0.0:
                    ract += r
                    r2 += r * r
                    na += 1
        Q += qin
        if Q <= kmin:
            pm = 0.0
        elif Q >= kmax:
            pm = 1.0
        else:
            pm = PMAXM * (Q - kmin) / (kmax - kmin)
        pbuf[ptr] = pm
        qbuf[ptr] = Q
        pd = pbuf[(ptr + 1) % D]
        qd = qbuf[(ptr + 1) % D]
        od = obuf[(ptr + 1) % D]

        # ---------------- congestion signals and control ----------------
        for i in range(n):
            if S[i, ACT] < 0.5:
                continue
            if mode == 1:
                if step % 4 == 0:
                    U = qd / (CAP * RTT) + od / CAP
                    if U > ETA:
                        S[i, HR] = S[i, HR] * ETA / U + RAIH
                    else:
                        S[i, HR] = S[i, HR] + RAIH
                    if S[i, HR] > RL:
                        S[i, HR] = RL
                    if S[i, HR] < RMIN:
                        S[i, HR] = RMIN
                continue
            npk = max(S[i, RC], 1.0) * DT / PKT
            if mode == 4:
                npk = max(S[i, EFF], 1.0) * DT / PKT
            cnp = 0
            if np.random.random() < 1.0 - (1.0 - pd) ** npk:
                cnp = 1
            S[i, ECN] = 0.98 * S[i, ECN] + 0.02 * cnp
            S[i, CG] += DT
            S[i, TINC] += DT
            S[i, TAL] += DT
            if cnp == 1 and S[i, CG] >= 50.0:
                S[i, RT] = S[i, RC]
                if mode == 2:
                    S[i, RC] = S[i, RC] * 0.97
                else:
                    S[i, RC] = S[i, RC] * (1.0 - S[i, AL] / 2.0)
                S[i, AL] = (1.0 - GAI) * S[i, AL] + GAI
                S[i, STG] = 0.0
                S[i, TINC] = 0.0
                S[i, TAL] = 0.0
                S[i, CG] = 0.0
                S[i, CON] = 0.0
                S[i, CUT] = 0.0
            else:
                if S[i, TAL] >= 55.0:
                    S[i, AL] = S[i, AL] * (1.0 - GAI)
                    S[i, TAL] = 0.0
                tint = 55.0
                if mode == 2:
                    tint = 40.0
                if S[i, TINC] >= tint:
                    S[i, TINC] = 0.0
                    S[i, STG] += 1.0
                    S[i, CON] += 1.0
                    if S[i, STG] <= 5.0:
                        S[i, RC] = (S[i, RT] + S[i, RC]) / 2.0
                    else:
                        stp = RAI
                        if mode == 2 and S[i, CON] > 5.0:
                            stp = RAI * min(1.0 + (S[i, CON] - 5.0), 8.0)
                        S[i, RT] = min(S[i, RT] + stp, RL)
                        S[i, RC] = (S[i, RT] + S[i, RC]) / 2.0
                if mode == 2 and S[i, CUT] < 0.5 and qd > 0.5 * XOFF:
                    S[i, RC] = S[i, RC] / 4.0
                    S[i, CUT] = 1.0
            if S[i, RC] > RL:
                S[i, RC] = RL
            if S[i, RC] < RMIN:
                S[i, RC] = RMIN

        # ---------------- draining and completion ----------------
        out = CAP * DT
        if Q < out:
            out = Q
        Qs = Q
        for i in range(n):
            if S[i, ACT] < 0.5:
                continue
            if Qs > 0.0:
                dr = out * S[i, QI] / Qs
            else:
                dr = 0.0
            S[i, QI] -= dr
            S[i, DREM] -= dr
            if S[i, DREM] <= 1.0 and S[i, REM] <= 0.0:
                S[i, ACT] = 0.0
                S[i, LAST] = t
                S[i, QI] = 0.0
                if nfct < 60000:
                    fct[nfct] = t + DT - S[i, FST]
                    nfct += 1
        Q -= out
        if Q < 0.0:
            Q = 0.0
        obuf[ptr] = out / DT
        ptr = (ptr + 1) % D
        out_tot += out
        if Q > 0.0 or out > 0.0:
            busy_t += DT

        # ---------------- PFC ----------------
        if paused == 0 and Q > XOFF:
            paused = 1
            pfc_ev += 1
        elif paused == 1 and Q < XON:
            paused = 0
        if paused == 1:
            pfc_t += DT

        # ---------------- ACC-style threshold adaptation ----------------
        if mode == 3:
            acc_n += 1
            if Q > acc_q:
                acc_q = Q
            acc_o += out
            if acc_n >= 20:
                util = acc_o / (CAP * DT * acc_n)
                if acc_q > 0.7 * XOFF:
                    s_acc = max(s_acc * 0.7, 0.25)
                elif util < 0.85 and acc_q < KMIN0 * s_acc:
                    s_acc = min(s_acc * 1.2, 4.0)
                acc_n = 0
                acc_q = 0.0
                acc_o = 0.0

        # ---------------- stats ----------------
        qsum += Q
        if Q > qmax:
            qmax = Q
        nsteps += 1
        if na >= 2 and r2 > 0.0:
            jsum += ract * ract / (na * r2)
            jn += 1
        if trace_n > 0 and step % 2 == 0 and ntr < trace_n:
            tr_q[ntr] = Q
            tr_r[ntr] = ract / max(na, 1)
            tr_t[ntr] = t
            ntr += 1

        # ---------------- job phase progress ----------------
        for j in range(nj):
            if jst[j] != 1:
                continue
            anyact = False
            for f in range(FM):
                if S[j * FM + f, ACT] > 0.5:
                    anyact = True
            if not anyact:
                jp[j] += 1
                if jp[j] < nph[j]:
                    start_phase(j, jp[j], t + DT, nfl, bsz, S)
                else:
                    k = jk[j]
                    tend = t + DT
                    cct[j, k] = tend - jcs[j]
                    itt[j, k] = tend - jib[j]
                    k += 1
                    jk[j] = k
                    if k < K:
                        jst[j] = 0
                        jib[j] = tend
                        jstart[j] = tend + comp[j, k]
                        pws[j] = tend + pcomp[j, k]
                        pwe[j] = pws[j] + 1.3 * idealj[j]
                    else:
                        jst[j] = 2
        t += DT
        step += 1

    jf = 1.0
    if jn > 0:
        jf = jsum / jn
    stats = np.zeros(8)
    stats[0] = pfc_t
    stats[1] = pfc_ev
    stats[2] = qsum / max(nsteps, 1)
    stats[3] = qmax
    stats[4] = jf
    stats[5] = out_tot / max(busy_t * CAP, 1.0)
    stats[6] = t
    stats[7] = nfct
    return cct, itt, fct[:nfct], stats, tr_t[:ntr], tr_q[:ntr], tr_r[:ntr], idealj
