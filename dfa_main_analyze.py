# ----- Add these imports near top of file -----
from scipy import stats

# ----- (After you computed Alpha1_s, Alpha1_l, Alpha2_s, Alpha2_l) -----

# Helper: compute SEM ignoring NaNs
def sem_nan(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return stats.sem(x, ddof=1)

# Compute means & SEMs (order 1)
mean_a1_s = np.nanmean(Alpha1_s)
sem_a1_s  = sem_nan(Alpha1_s)
mean_a1_l = np.nanmean(Alpha1_l)
sem_a1_l  = sem_nan(Alpha1_l)

# Compute means & SEMs (order 2)
mean_a2_s = np.nanmean(Alpha2_s)
sem_a2_s  = sem_nan(Alpha2_s)
mean_a2_l = np.nanmean(Alpha2_l)
sem_a2_l  = sem_nan(Alpha2_l)

# Paired t-test (order 1): include only subjects with both values
mask1 = ~np.isnan(Alpha1_s) & ~np.isnan(Alpha1_l)
n1 = int(mask1.sum())
if n1 >= 2:
    diff1 = Alpha1_l[mask1] - Alpha1_s[mask1]
    t1, p1 = stats.ttest_rel(Alpha1_l[mask1], Alpha1_s[mask1], nan_policy='omit')
    df1 = n1 - 1
    # paired Cohen's d: mean(diff)/sd(diff) with sd = std(diff, ddof=1)
    sd_diff1 = np.std(diff1, ddof=1)
    cohend1 = np.nan if sd_diff1 == 0 else np.mean(diff1) / sd_diff1
else:
    t1 = p1 = df1 = cohend1 = np.nan

# Paired t-test (order 2)
mask2 = ~np.isnan(Alpha2_s) & ~np.isnan(Alpha2_l)
n2 = int(mask2.sum())
if n2 >= 2:
    diff2 = Alpha2_l[mask2] - Alpha2_s[mask2]
    t2, p2 = stats.ttest_rel(Alpha2_l[mask2], Alpha2_s[mask2], nan_policy='omit')
    df2 = n2 - 1
    sd_diff2 = np.std(diff2, ddof=1)
    cohend2 = np.nan if sd_diff2 == 0 else np.mean(diff2) / sd_diff2
else:
    t2 = p2 = df2 = cohend2 = np.nan

# Print results (human readable)
print("\n--- DFA summary (order 1) ---")
print(f"Small: mean = {mean_a1_s:.4f}, SEM = {sem_a1_s:.4f} (N={np.sum(~np.isnan(Alpha1_s))})")
print(f"Large: mean = {mean_a1_l:.4f}, SEM = {sem_a1_l:.4f} (N={np.sum(~np.isnan(Alpha1_l))})")
if not np.isnan(t1):
    print(f"Paired t-test (large vs small): t({df1}) = {t1:.3f}, p = {p1:.4f}, Cohen's d = {cohend1:.3f}")
else:
    print("Paired t-test (order 1): insufficient paired data")

print("\n--- DFA summary (order 2) ---")
print(f"Small: mean = {mean_a2_s:.4f}, SEM = {sem_a2_s:.4f} (N={np.sum(~np.isnan(Alpha2_s))})")
print(f"Large: mean = {mean_a2_l:.4f}, SEM = {sem_a2_l:.4f} (N={np.sum(~np.isnan(Alpha2_l))})")
if not np.isnan(t2):
    print(f"Paired t-test (large vs small): t({df2}) = {t2:.3f}, p = {p2:.4f}, Cohen's d = {cohend2:.3f}")
else:
    print("Paired t-test (order 2): insufficient paired data")

# Save a small summary DataFrame
summary_df = pd.DataFrame({
    'condition': ['small_order1','large_order1','small_order2','large_order2'],
    'mean': [mean_a1_s, mean_a1_l, mean_a2_s, mean_a2_l],
    'SEM':  [sem_a1_s, sem_a1_l, sem_a2_s, sem_a2_l],
    'N':    [int(np.sum(~np.isnan(Alpha1_s))), int(np.sum(~np.isnan(Alpha1_l))),
             int(np.sum(~np.isnan(Alpha2_s))), int(np.sum(~np.isnan(Alpha2_l)))]
})
summary_csv = 'dfa_summary_stats.csv'
summary_df.to_csv(summary_csv, index=False)
print(f"\nSaved summary CSV to {summary_csv}")

# Produce LaTeX-ready lines you can paste into the Results paragraph
latex_line_order1 = ("Order 1 DFA: small: ${mean:.3f}\\pm{sem:.3f}$ (N={N}); "
                     "large: ${mean_l:.3f}\\pm{sem_l:.3f}$ (N={Nl}). "
                     "Paired t({df}) = {t:.3f}, p = {p:.4f}, Cohen's d = {d:.3f}").format(
                        mean=mean_a1_s, sem=sem_a1_s, N=int(np.sum(~np.isnan(Alpha1_s))),
                        mean_l=mean_a1_l, sem_l=sem_a1_l, Nl=int(np.sum(~np.isnan(Alpha1_l))),
                        df=df1 if not np.isnan(df1) else 0, t=t1 if not np.isnan(t1) else 0, p=p1 if not np.isnan(p1) else 1.0, d=cohend1 if not np.isnan(cohend1) else 0.0)

latex_line_order2 = ("Order 2 DFA: small: ${mean:.3f}\\pm{sem:.3f}$ (N={N}); "
                     "large: ${mean_l:.3f}\\pm{sem_l:.3f}$ (N={Nl}). "
                     "Paired t({df}) = {t:.3f}, p = {p:.4f}, Cohen's d = {d:.3f}").format(
                        mean=mean_a2_s, sem=sem_a2_s, N=int(np.sum(~np.isnan(Alpha2_s))),
                        mean_l=mean_a2_l, sem_l=sem_a2_l, Nl=int(np.sum(~np.isnan(Alpha2_l))),
                        df=df2 if not np.isnan(df2) else 0, t=t2 if not np.isnan(t2) else 0, p=p2 if not np.isnan(p2) else 1.0, d=cohend2 if not np.isnan(cohend2) else 0.0)

print("\nLaTeX-ready summary lines (paste into manuscript):\n")
print(latex_line_order1)
print()
print(latex_line_order2)

# Optionally save the LaTeX lines to file
with open('dfa_latex_lines.txt','w') as f:
    f.write(latex_line_order1 + "\n\n" + latex_line_order2 + "\n")
print("Saved LaTeX lines to dfa_latex_lines.txt")
