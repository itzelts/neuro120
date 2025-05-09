import pandas as pd
import numpy as np
from scipy import stats

# Read the goal_dual CSV data
df = pd.read_csv('results/goal_dual/rsa_results_raw.csv')

# Calculate mean values for each ROI
roi_summary = df.groupby('ROI').agg({
    'adjacency_rho': 'mean',
    'functional_rho': 'mean'
}).reset_index()

# Calculate advantage
roi_summary['advantage'] = roi_summary['adjacency_rho'] - roi_summary['functional_rho']

# Calculate percentage of subjects with functional > anatomical
results = []
for roi in df['ROI'].unique():
    roi_data = df[df['ROI'] == roi]
    func_dominant = (roi_data['functional_rho'] > roi_data['adjacency_rho']).mean() * 100
    results.append({'ROI': roi, 'percent_func_dominant': func_dominant})
func_dominance = pd.DataFrame(results)

# Perform paired t-tests
pvalues = {}
t_values = {}
for roi in df['ROI'].unique():
    roi_data = df[df['ROI'] == roi]
    t, p = stats.ttest_rel(roi_data['adjacency_rho'], roi_data['functional_rho'])
    pvalues[roi] = p
    t_values[roi] = t

# Print formatted results for the table
print('\\begin{table}[h]')
print('\\centering')
print('\\caption{ROI Performance Metrics for Anatomical and Functional Models (Goal-Dual)}')
print('\\label{tab:roi_metrics}')
print('\\begin{tabular}{|l|c|c|c|c|}')
print('\\hline')
print('\\textbf{ROI} & \\textbf{Anatomical ($\\rho$)} & \\textbf{Functional ($\\rho$)} & \\textbf{Advantage} & \\textbf{p-value} \\\\')
print('\\hline')
for i, row in roi_summary.iterrows():
    roi = row['ROI']
    adj = row['adjacency_rho']
    func = row['functional_rho']
    adv = row['advantage']
    p = pvalues[roi]
    t = t_values[roi]
    p_str = f'${p:.2e}$' if p >= 1e-8 else '$< 10^{-8}$'
    print(f'{roi} & {adj:.3f} & {func:.3f} & {adv:.3f} & {p_str} \\\\')
print('\\hline')
print('\\end{tabular}')
print('\\end{table}')

# Print percentages of subjects with functional dominance
print('\nPercentage of subjects showing functional dominance:')
for i, row in func_dominance.iterrows():
    print(f"{row['ROI']}: {row['percent_func_dominant']:.1f}%")

# Print the hierarchical gradient (difference between highest and lowest advantage)
gradient = roi_summary['advantage'].max() - roi_summary['advantage'].min()
print(f'\nHierarchical gradient: {gradient:.3f}')

print('\nDetailed ROI metrics:')
print(roi_summary.to_string(index=False))

print('\nT-values:')
for roi, t in t_values.items():
    print(f"{roi}: {t:.3f}") 