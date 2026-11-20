import json, re
R = json.load(open('results.json'))
e2, e3, e4, e5, e6, e1 = R['e2'], R['e3'], R['e4'], R['e5'], R['e6'], R['e1']
M = ['DCQCN','HPCC','Barre-style','ACC-style','Rule (schedule only)','Ours','Ours-sched']
LAB = {'DCQCN':'DCQCN','HPCC':'HPCC-style','Barre-style':'Barre-style','ACC-style':'ACC-style','Rule (schedule only)':'Rule (schedule only)','Ours':'Ours (predictive)','Ours-sched':'Ours (schedule-aware)'}
def ci(v, d=2): return f"{v[0]:.{d}f} [{v[1][0]:.{d}f}, {v[1][1]:.{d}f}]"
def f(x, d=2): return f"{x:.{d}f}"
T = {}
# main per-scenario table
rows = []
for m in M:
    cells = [f(e2[k][m]['cct_sd'][0]) for k in ['S1 AllReduce x1','S2 AllToAll x1','S3 Mixed x2','S4 Mixed x3','random','shifted']]
    rows.append(LAB[m] + ' & ' + ' & '.join(cells) + r' \\')
T['MAINTAB'] = '\n'.join(rows)
# detailed metrics on random set
rows = []
for m in M:
    d = e2['random'][m]
    rows.append(f"{LAB[m]} & {ci(d['cct_sd'])} & {f(d['p99_sd'][0])} & {f(d['itt'][0]/1000,2)} & {f(d['pfc'][0],2)} & {f(d['jain'][0],3)} & {f(d['util'][0],2)} & {f(d['fct'][0]/1000,2)} \\\\")
T['DETAILTAB'] = '\n'.join(rows)
# stats
rows = []
for fam, nm in [('random','Random'),('shifted','Shifted')]:
    for tgt, key, lab in [('Ours','stats','Pred.'),('Ours-sched','stats_sched','Sched.')]:
        for m in M:
            if m == tgt or m.startswith('Ours') and False: continue
            s = e2[fam][key][m]
            p = '<0.001' if s['p'] < 0.001 else f(s['p'],3)
            sh = {'Ours':'Pred.','Ours-sched':'Sched.','Rule (schedule only)':'Rule','Barre-style':'Barre','ACC-style':'ACC','HPCC':'HPCC','DCQCN':'DCQCN'}[m]
            rows.append(f"{nm} & {lab} vs {sh} & {f(s['mean_gain_pct'],1)} & {s['wins']}/{s['n']} & {p} \\\\")
T['STATTAB'] = '\n'.join(rows)
# ablation
names = [('DCQCN','DCQCN (no prediction, no learning)'),('Rule (schedule only)','Rule: own schedule, no learning'),('Rule (predicted, no RL)','Rule: predicted windows, no learning'),
 ('nopred','Learned, no schedule or forecast (DCQCN anchor)'),('sched','Learned, own schedule only'),('ma','Learned, moving average forecast'),('oracle','Learned, oracle forecast'),
 ('fct_reward','Learned, FCT reward instead of CCT'),('nophase','Learned, no phase feature'),('full_mlp','Learned, MLP forecast (full)')]
rows = [f"{lab} & {ci(e3[k]['cct_sd'])} & {f(e3[k]['p99_sd'][0])} & {f(e3[k]['pfc'][0],2)} \\\\" for k, lab in names]
T['ABLTAB'] = '\n'.join(rows)
# quantization
rows = [f"{k} & {f(e5[k]['cct'],3)} & {f(e5[k]['pfc'],2)} \\\\" for k in ['float64','int8','int6','int4']]
T['QUANTTAB'] = '\n'.join(rows)
# gains vs DCQCN used in text and comparison table
rn = e2['random']
T['G_DC'] = f(100*(rn['DCQCN']['cct_sd'][0]-rn['Ours']['cct_sd'][0])/rn['DCQCN']['cct_sd'][0],1)
T['G_HP'] = f(100*(rn['HPCC']['cct_sd'][0]-rn['Ours']['cct_sd'][0])/rn['HPCC']['cct_sd'][0],1)
T['G_BA'] = f(100*(rn['Barre-style']['cct_sd'][0]-rn['Ours']['cct_sd'][0])/rn['Barre-style']['cct_sd'][0],1)
T['G_AC'] = f(100*(rn['ACC-style']['cct_sd'][0]-rn['Ours']['cct_sd'][0])/rn['ACC-style']['cct_sd'][0],1)
T['GS_DC'] = f(100*(rn['DCQCN']['cct_sd'][0]-rn['Ours-sched']['cct_sd'][0])/rn['DCQCN']['cct_sd'][0],1)
T['GS_BA'] = f(100*(rn['Barre-style']['cct_sd'][0]-rn['Ours-sched']['cct_sd'][0])/rn['Barre-style']['cct_sd'][0],1)
T['ITT_G'] = f(100*(rn['DCQCN']['itt'][0]-rn['Ours']['itt'][0])/rn['DCQCN']['itt'][0],1)
T['ITT_GS'] = f(100*(rn['DCQCN']['itt'][0]-rn['Ours-sched']['itt'][0])/rn['DCQCN']['itt'][0],1)
s2 = e2['S2 AllToAll x1']
T['S2_G'] = f(100*(s2['DCQCN']['cct_sd'][0]-s2['Ours']['cct_sd'][0])/s2['DCQCN']['cct_sd'][0],1)
T['PFC_DC'] = f(rn['DCQCN']['pfc'][0],2); T['PFC_RULE'] = f(rn['Rule (schedule only)']['pfc'][0],2); T['PFC_OURS'] = f(rn['Ours']['pfc'][0],2)
T['P99_DC'] = f(rn['DCQCN']['p99_sd'][0],2); T['P99_OURS'] = f(rn['Ours']['p99_sd'][0],2); T['P99_SCH'] = f(rn['Ours-sched']['p99_sd'][0],2)
T['E1MA'] = f(e1['in-family']['ma'],1); T['E1ML'] = f(e1['in-family']['mlp'],1); T['E1MAS'] = f(e1['shifted']['ma'],1); T['E1MLS'] = f(e1['shifted']['mlp'],1)
pn = e4['pred_noise']
T['PN0'] = f(pn['0'],2); T['PN5000'] = f(pn['5000'],2); T['PNDC'] = f(pn['DCQCN'],2); T['PNHP'] = f(pn['HPCC'],2)
T['V_Q'] = f(e6['qavg_KB'],0); T['V_U'] = f(e6['util'],2); T['V_J'] = f(e6['jain'],3)
ts=R['trace_stats']
for k,n in [('DCQCN','DC'),('HPCC','HP'),('Barre-style','BA'),('Ours','O')]:
    T['TQ_'+n+'_P']=f(ts[k]['qpeak_KB'],0); T['TQ_'+n+'_M']=f(ts[k]['qmean_KB'],0)
T['QP'] = str(e5['params']); T['QM'] = str(e5['macs'])

for k,n in [('sched','A_SCHED'),('full_mlp','A_FULL'),('fct_reward','A_FCT'),('oracle','A_ORACLE'),('ma','A_MA'),('nopred','A_NOPRED'),('Rule (schedule only)','A_RULE'),('Rule (predicted, no RL)','A_RULEP'),('nophase','A_NOPH')]:
    T[n]=f(e3[k]['cct_sd'][0],2)
T['A_DC']=f(e3['DCQCN']['cct_sd'][0],2); T['A_NOPRED_PFC']=f(e3['nopred']['pfc'][0],2); T['A_RULE_PFC']=f(e3['Rule (schedule only)']['pfc'][0],2)
T['LD025']=f(e4['load']['0.25']['Ours-sched'],2); T['LD025D']=f(e4['load']['0.25']['DCQCN'],2); T['LD2']=f(e4['load']['2.0']['Ours-sched'],2); T['LD2D']=f(e4['load']['2.0']['DCQCN'],2)

t = open('report_tmpl.tex').read()
for k, v in T.items(): t = t.replace('@@'+k+'@@', v)
left = re.findall(r'@@(\w+)@@', t)
assert not left, left
open('report.tex','w').write(t)
print('ok')
