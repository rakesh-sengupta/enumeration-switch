#!/usr/bin/env python3
"""
dfa_alpha_vs_rt.py

Alpha vs Mean RT analysis for enumeration DFA paper:
- Loads results.mat
- Computes per-subject DFA (order1) for small and large numerosities
- Computes mean RT for each condition
- Correlations between alpha and mean RT
- Saves scatter plots
"""

import os, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats

np.random.seed(0)
outdir = './results'
os.makedirs(outdir, exist_ok=True)

# ---------- DFA function ----------
def DFA(DATA, win_length, order):
    DATA = np.asarray(DATA).astype(float)
    DATA = DATA[~np.isnan(DATA)]
    N = len(DATA)
    if N < win_length:
        return np.nan
    n = N // win_length
    N1 = n * win_length
    mean1 = np.mean(DATA[:N1])
    y = np.cumsum(DATA[:N1] - mean1)
    Yn = np.zeros(N1)
    x_axis = np.arange(1, win_length + 1)
    for j in range(n):
        seg = y[j*win_length:(j+1)*win_length]
        coeffs = np.polyfit(x_axis, seg, order)
        fitted = np.polyval(coeffs, x_axis)
        Yn[j*win_length:(j+1)*win_length] = fitted
    sum1 = np.sum((y - Yn)**2) / N1
    return math.sqrt(sum1)

def alpha_from_windows(series, window_list, order=1):
    F = []
    for w in window_list:
        F.append(DFA(series, int(w), order))
    F = np.array(F)
    valid = (~np.isnan(F)) & (F > 0)
    if valid.sum() < 3:
        return np.nan, F
    logs = np.log(np.array(window_list)[valid])
    logF = np.log(F[valid])
    slope, intercept = np.polyfit(logs, logF, 1)
    return slope, F

# ---------- Load results.mat ----------
def extract_results_struct_fields(matpath='results.mat'):
    mat = loadmat(matpath, squeeze_me=True, struct_as_record=False)
    if 'results' not in mat:
        raise KeyError("No 'results' in mat file. Keys available: " + ", ".join(mat.keys()))
    results = mat['results']
    
    subj_structs = []
    try:
        first = results.flatten()[0] if isinstance(results, np.ndarray) else results
    except Exception:
        first = results
        
    all_field = None
    if isinstance(first, np.void) and 'all' in first.dtype.names:
        all_field = first['all']
    elif hasattr(first, 'all'):
        all_field = getattr(first, 'all')
        
    if all_field is not None:
        subj_structs = list(np.atleast_1d(all_field).flat)
    else:
        subj_structs = list(np.atleast_1d(results).flat)
        
    subjects = []
    for s in subj_structs:
        subj = {}
        fields = ['rt_s','rt_l','expLog']
        for f in fields:
            try:
                if isinstance(s, np.void):
                    subj[f] = s[f] if f in s.dtype.names else None
                else:
                    subj[f] = getattr(s, f, None)
            except Exception:
                subj[f] = None
        subjects.append(subj)
    return subjects

# ---------- MAIN ANALYSIS ----------
matpath = 'results.mat'
subjects = extract_results_struct_fields(matpath)
print(f"Loaded {len(subjects)} subject entries")

win_lengths = [20,30,40,50,60,70,80]

alpha1_s_list = []
alpha1_l_list = []
meanRT_s = []
meanRT_l = []

for si, subj in enumerate(subjects):
    rt_s = subj.get('rt_s', None)
    rt_l = subj.get('rt_l', None)
    
    if rt_s is None or rt_l is None:
        alpha1_s_list.append(np.nan)
        alpha1_l_list.append(np.nan)
        meanRT_s.append(np.nan)
        meanRT_l.append(np.nan)
        continue
        
    rs = np.asarray(rt_s).astype(float).flatten()
    rl = np.asarray(rt_l).astype(float).flatten()
    
    # Compute DFA alpha (order 1)
    a_s, F_s = alpha_from_windows(rs, win_lengths, order=1)
    a_l, F_l = alpha_from_windows(rl, win_lengths, order=1)
    
    alpha1_s_list.append(a_s)
    alpha1_l_list.append(a_l)
    meanRT_s.append(np.mean(rs))
    meanRT_l.append(np.mean(rl))

# Create DataFrame
df = pd.DataFrame({
    'alpha1_small': alpha1_s_list,
    'alpha1_large': alpha1_l_list,
    'meanRT_small': meanRT_s,
    'meanRT_large': meanRT_l
})

# Remove subjects with missing data
df_clean = df.dropna()

print(f"Final sample size: {len(df_clean)} subjects")

# Group statistics
def mean_sem(x):
    x = np.asarray(x)
    return np.mean(x), stats.sem(x, ddof=1)

m_s, sem_s = mean_sem(df_clean['alpha1_small'])
m_l, sem_l = mean_sem(df_clean['alpha1_large'])
m_rt_s, sem_rt_s = mean_sem(df_clean['meanRT_small'])
m_rt_l, sem_rt_l = mean_sem(df_clean['meanRT_large'])

print(f"\n--- GROUP STATISTICS ---")
print(f"Alpha small: {m_s:.3f} ± {sem_s:.3f}")
print(f"Alpha large: {m_l:.3f} ± {sem_l:.3f}")
print(f"Mean RT small: {m_rt_s:.1f} ± {sem_rt_s:.1f} ms")
print(f"Mean RT large: {m_rt_l:.1f} ± {sem_rt_l:.1f} ms")

# Paired t-tests
t_alpha, p_alpha = stats.ttest_rel(df_clean['alpha1_large'], df_clean['alpha1_small'])
t_rt, p_rt = stats.ttest_rel(df_clean['meanRT_large'], df_clean['meanRT_small'])

print(f"\n--- PAIRED T-TESTS ---")
print(f"Alpha (large vs small): t({len(df_clean)-1}) = {t_alpha:.3f}, p = {p_alpha:.4f}")
print(f"Mean RT (large vs small): t({len(df_clean)-1}) = {t_rt:.3f}, p = {p_rt:.4f}")

# Correlations between alpha and mean RT
corr_small = stats.pearsonr(df_clean['alpha1_small'], df_clean['meanRT_small'])
corr_large = stats.pearsonr(df_clean['alpha1_large'], df_clean['meanRT_large'])

print(f"\n--- CORRELATIONS ---")
print(f"Alpha vs Mean RT (small): r = {corr_small.statistic:.3f}, p = {corr_small.pvalue:.4f}")
print(f"Alpha vs Mean RT (large): r = {corr_large.statistic:.3f}, p = {corr_large.pvalue:.4f}")

# ---------- PLOTS ----------
plt.rcParams.update({'font.size': 12})

# Scatter: Alpha vs Mean RT
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(df_clean['meanRT_small'], df_clean['alpha1_small'], alpha=0.7, s=60)
plt.xlabel('Mean RT (ms)')
plt.ylabel('DFA Exponent (α)')
plt.title('Subitization (1-6 dots)')
# Add correlation line if significant
if corr_small.pvalue < 0.05:
    z = np.polyfit(df_clean['meanRT_small'], df_clean['alpha1_small'], 1)
    p = np.poly1d(z)
    plt.plot(df_clean['meanRT_small'], p(df_clean['meanRT_small']), "r--", alpha=0.8)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(df_clean['meanRT_large'], df_clean['alpha1_large'], alpha=0.7, s=60, color='red')
plt.xlabel('Mean RT (ms)')
plt.ylabel('DFA Exponent (α)')
plt.title('Estimation (24-34 dots)')
# Add correlation line if significant
if corr_large.pvalue < 0.05:
    z = np.polyfit(df_clean['meanRT_large'], df_clean['alpha1_large'], 1)
    p = np.poly1d(z)
    plt.plot(df_clean['meanRT_large'], p(df_clean['meanRT_large']), "r--", alpha=0.8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(outdir, 'alpha_vs_meanRT_scatter.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(outdir, 'alpha_vs_meanRT_scatter.pdf'), bbox_inches='tight')
plt.close()

# Save results to CSV
results_summary = {
    'n_subjects': len(df_clean),
    'alpha_small_mean': m_s,
    'alpha_small_sem': sem_s,
    'alpha_large_mean': m_l,
    'alpha_large_sem': sem_l,
    'meanRT_small_mean': m_rt_s,
    'meanRT_small_sem': sem_rt_s,
    'meanRT_large_mean': m_rt_l,
    'meanRT_large_sem': sem_rt_l,
    't_alpha': t_alpha,
    'p_alpha': p_alpha,
    't_rt': t_rt,
    'p_rt': p_rt,
    'corr_small_r': corr_small.statistic,
    'corr_small_p': corr_small.pvalue,
    'corr_large_r': corr_large.statistic,
    'corr_large_p': corr_large.pvalue
}

# Save summary dict
import json
with open(os.path.join(outdir, 'alpha_rt_results.json'), 'w') as f:
    json.dump(results_summary, f, indent=4)

# Save clean dataframe
df_clean.to_csv(os.path.join(outdir, 'alpha_rt_data.csv'), index=False)

print(f"\nResults saved to {outdir}/")
print("Analysis complete!")
