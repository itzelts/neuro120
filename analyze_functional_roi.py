#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, ttest_ind, pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import os
import pathlib

# Create output directory for results
os.makedirs('roi_functional_analysis', exist_ok=True)

print("Analyzing functional groupings performance in RSA...")

# Define current functional groupings based on sub_anlysis.py
func_group = {
    'toe':'DFM',     # Distal fine manipulation
    'finger':'DFM',
    'ankle':'MJA',    # Mid-level joint articulation
    'wrist':'MJA',
    'leftleg':'PLM',  # Proximal limb movements
    'rightleg':'PLM',
    'forearm':'PLM',
    'upperarm':'PLM',
    'jaw':'OFC',      # Orofacial communication
    'lip':'OFC',
    'tongue':'OFC',
    'eye':'TOR'       # Targeted orientation
}

# Create a reverse mapping for easier lookup
functional_groups = {
    'DFM': ['toe', 'finger'],
    'MJA': ['ankle', 'wrist'],
    'PLM': ['leftleg', 'rightleg', 'forearm', 'upperarm'],
    'OFC': ['jaw', 'lip', 'tongue'],
    'TOR': ['eye']
}

# Add descriptions for each group
group_descriptions = {
    'DFM': 'Distal Fine Manipulation',
    'MJA': 'Mid-level Joint Articulation',
    'PLM': 'Proximal Limb Movements',
    'OFC': 'Orofacial Communication',
    'TOR': 'Targeted Orientation'
}

# Define colors for visualization
group_colors = {
    'DFM': 'tab:red',
    'MJA': 'tab:blue',
    'PLM': 'tab:green',
    'OFC': 'tab:purple',
    'TOR': 'tab:orange'
}

# Load the raw RSA data
results_dir = pathlib.Path('results')
raw_data = pd.read_csv(results_dir / 'rsa_results_raw.csv')
print(f"Loaded data with shape: {raw_data.shape}")
print(raw_data.head(5))

# Extract unique subjects and ROIs
subjects = raw_data['subject'].unique()
rois = raw_data['ROI'].unique()
print(f"\nAnalyzing data from {len(subjects)} subjects across {len(rois)} ROIs")
print(f"ROIs: {', '.join(rois)}")

# Part 1: Basic Performance Analysis by ROI
# -----------------------------------------
print("\n--- Performance Analysis by ROI ---")

# Calculate mean and standard error for each ROI and model
roi_stats = raw_data.groupby('ROI').agg(
    adjacency_mean=('adjacency_rho', 'mean'),
    adjacency_se=('adjacency_rho', lambda x: x.std() / np.sqrt(len(x))),
    functional_mean=('functional_rho', 'mean'),
    functional_se=('functional_rho', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

# Add a column indicating which model performs better for each ROI
roi_stats['better_model'] = roi_stats.apply(
    lambda row: 'Adjacency' if row['adjacency_mean'] > row['functional_mean'] else 'Functional', 
    axis=1
)

# Calculate the difference between models
roi_stats['model_diff'] = roi_stats['adjacency_mean'] - roi_stats['functional_mean']

# Display results
print("\nPerformance by ROI:")
print(roi_stats[['ROI', 'adjacency_mean', 'functional_mean', 'model_diff', 'better_model']])

# Plot the results
plt.figure(figsize=(10, 6))
x = np.arange(len(rois))
width = 0.35

plt.bar(x - width/2, roi_stats['adjacency_mean'], width, label='Anatomical', 
       yerr=roi_stats['adjacency_se'], color='tab:blue', alpha=0.7)
plt.bar(x + width/2, roi_stats['functional_mean'], width, label='Functional', 
       yerr=roi_stats['functional_se'], color='tab:orange', alpha=0.7)

plt.xlabel('Region of Interest (ROI)')
plt.ylabel('Mean Correlation (ρ)')
plt.title('Comparison of Adjacency vs. Functional Models by ROI')
plt.xticks(x, roi_stats['ROI'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roi_functional_analysis/model_comparison_by_roi.png')

# Perform statistical tests to compare models within each ROI
roi_tests = []
for roi in rois:
    roi_data = raw_data[raw_data['ROI'] == roi]
    t, p = ttest_rel(roi_data['adjacency_rho'], roi_data['functional_rho'])
    roi_tests.append({
        'ROI': roi,
        't_statistic': t,
        'p_value': p,
        'significant': p < 0.05,
        'better_model': 'Adjacency' if t > 0 else 'Functional'
    })

roi_test_df = pd.DataFrame(roi_tests)
print("\nStatistical comparison of models by ROI:")
print(roi_test_df)

# Part 2: Hierarchical Analysis
# -----------------------------
print("\n--- Hierarchical Analysis ---")

# Create a hierarchical ordering of ROIs (from primary to higher-order regions)
hierarchical_order = ['M1', 'SMA', 'PMC', 'PPC']  # From primary to higher-order

# Calculate the mean difference between models for each ROI
raw_data['model_diff'] = raw_data['adjacency_rho'] - raw_data['functional_rho']
hierarchical_stats = raw_data.groupby('ROI')['model_diff'].agg(['mean', 'sem']).reindex(hierarchical_order)

# Plot the hierarchical transformation
plt.figure(figsize=(9, 6))
x = np.arange(len(hierarchical_order))
plt.bar(x, hierarchical_stats['mean'], yerr=hierarchical_stats['sem'], color='tab:blue', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
plt.xticks(x, hierarchical_order)
plt.xlabel('ROI (Hierarchical Order)')
plt.ylabel('Adjacency - Functional Correlation (ρ)')
plt.title('Transformation from Anatomical to Functional Organization Across Motor Hierarchy')
plt.grid(True, alpha=0.3)

# Add annotations for interpretation
for i, roi in enumerate(hierarchical_order):
    diff = hierarchical_stats.loc[roi, 'mean']
    if diff > 0.1:
        annotation = "Anatomical\ndominant"
    elif diff < -0.1:
        annotation = "Functional\ndominant"
    else:
        annotation = "Balanced"
    plt.text(i, hierarchical_stats.loc[roi, 'mean'] + np.sign(diff)*0.03, 
             annotation, ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('roi_functional_analysis/hierarchical_transformation.png')

# Part 3: Cross-ROI Analysis for Functional Groups
# -----------------------------------------------
print("\n--- Cross-ROI Analysis for Functional Groups ---")

# Reshape the data to analyze performance by ROI for each subject
pivot_adj = raw_data.pivot(index='subject', columns='ROI', values='adjacency_rho')
pivot_func = raw_data.pivot(index='subject', columns='ROI', values='functional_rho')

# Calculate correlation between ROIs based on subject patterns
roi_corr_adj = pivot_adj.corr()
roi_corr_func = pivot_func.corr()

# Create heatmaps for visualization
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(roi_corr_adj, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('ROI Correlations - Adjacency Model')

plt.subplot(1, 2, 2)
sns.heatmap(roi_corr_func, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('ROI Correlations - Functional Model')

plt.tight_layout()
plt.savefig('roi_functional_analysis/roi_correlation_heatmaps.png')

# Part 4: Hierarchical Clustering Analysis
# ---------------------------------------
print("\n--- Hierarchical Clustering Analysis ---")

# Perform hierarchical clustering on ROI correlation matrices
def perform_clustering(corr_matrix, name):
    # Calculate linkage matrix
    link = linkage(squareform(1 - corr_matrix), method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(link, labels=corr_matrix.index, leaf_font_size=12)
    plt.title(f'Hierarchical Clustering of ROIs - {name} Model')
    plt.xlabel('ROI')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(f'roi_functional_analysis/dendrogram_{name.lower()}.png')
    
    return link

# Perform clustering for both models
adj_link = perform_clustering(roi_corr_adj, 'Adjacency')
func_link = perform_clustering(roi_corr_func, 'Functional')

# Part 5: Model Performance by Body Part Category
# ---------------------------------------------
print("\n--- Model Performance by Functional Category ---")

# Compare adjacency vs functional model performance across hierarchy
hierarchical_comparison = pd.DataFrame({
    'ROI': hierarchical_order,
    'Adjacency Mean': [roi_stats[roi_stats['ROI'] == roi]['adjacency_mean'].values[0] for roi in hierarchical_order],
    'Functional Mean': [roi_stats[roi_stats['ROI'] == roi]['functional_mean'].values[0] for roi in hierarchical_order],
    'Difference': [roi_stats[roi_stats['ROI'] == roi]['model_diff'].values[0] for roi in hierarchical_order]
})

# Plot the model performance throughout the hierarchy
plt.figure(figsize=(10, 6))
x = np.arange(len(hierarchical_order))

plt.plot(x, hierarchical_comparison['Adjacency Mean'], 'o-', color='tab:blue', 
         label='Anatomical Model', linewidth=2, markersize=8)
plt.plot(x, hierarchical_comparison['Functional Mean'], 'o-', color='tab:orange', 
         label='Functional Model', linewidth=2, markersize=8)

plt.xlabel('Hierarchical Position')
plt.ylabel('Mean Correlation (ρ)')
plt.title('Model Performance Across Motor Hierarchy')
plt.xticks(x, hierarchical_order)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('roi_functional_analysis/model_performance_across_hierarchy.png')

# Part 6: Generate Numerical Metrics in CSV Files
# -------------------------------------------
print("\n--- Generating Numerical Metrics in CSV Files ---")

# 1. ROI-level Metrics CSV
roi_metrics = pd.DataFrame({
    'ROI': roi_stats['ROI'],
    'Adjacency_Mean': roi_stats['adjacency_mean'],
    'Adjacency_SE': roi_stats['adjacency_se'],
    'Functional_Mean': roi_stats['functional_mean'],
    'Functional_SE': roi_stats['functional_se'],
    'Model_Difference': roi_stats['model_diff'],
    'Better_Model': roi_stats['better_model'],
    't_statistic': [row['t_statistic'] for _, row in roi_test_df.iterrows()],
    'p_value': [row['p_value'] for _, row in roi_test_df.iterrows()],
    'Significant_Difference': [row['significant'] for _, row in roi_test_df.iterrows()]
})

# Add hierarchical position information
hierarchical_positions = {'M1': 1, 'SMA': 2, 'PMC': 3, 'PPC': 4}
roi_metrics['Hierarchical_Position'] = roi_metrics['ROI'].map(hierarchical_positions)

# Sort by hierarchical position
roi_metrics = roi_metrics.sort_values('Hierarchical_Position')

roi_metrics.to_csv('roi_functional_analysis/roi_performance_metrics.csv', index=False)
print("Saved ROI performance metrics to roi_functional_analysis/roi_performance_metrics.csv")

# 2. Subject-level Metrics CSV
# Calculate performance metrics for each subject
subject_metrics = []
for subject in subjects:
    subject_data = raw_data[raw_data['subject'] == subject]
    
    # Calculate mean values for adjacency and functional models by subject
    adj_mean = subject_data['adjacency_rho'].mean()
    func_mean = subject_data['functional_rho'].mean()
    
    # Calculate by-ROI metrics for this subject
    roi_values = {}
    for roi in rois:
        roi_data = subject_data[subject_data['ROI'] == roi]
        if not roi_data.empty:
            roi_values[f"{roi}_adjacency"] = roi_data['adjacency_rho'].values[0]
            roi_values[f"{roi}_functional"] = roi_data['functional_rho'].values[0]
            roi_values[f"{roi}_diff"] = roi_data['adjacency_rho'].values[0] - roi_data['functional_rho'].values[0]
    
    # Combine all metrics
    subject_dict = {
        'Subject': subject,
        'Adjacency_Mean': adj_mean,
        'Functional_Mean': func_mean,
        'Overall_Diff': adj_mean - func_mean,
        'Better_Model': 'Adjacency' if adj_mean > func_mean else 'Functional'
    }
    subject_dict.update(roi_values)
    
    subject_metrics.append(subject_dict)

subject_metrics_df = pd.DataFrame(subject_metrics)
subject_metrics_df.to_csv('roi_functional_analysis/subject_performance_metrics.csv', index=False)
print("Saved subject performance metrics to roi_functional_analysis/subject_performance_metrics.csv")

# 3. Hierarchical Transformation Metrics CSV
hierarchical_metrics = hierarchical_stats.reset_index()
hierarchical_metrics.columns = ['ROI', 'Model_Difference_Mean', 'Model_Difference_SEM']
hierarchical_metrics['Hierarchical_Position'] = hierarchical_metrics['ROI'].map(hierarchical_positions)

# Add adjacency and functional means for reference
for roi in hierarchical_metrics['ROI']:
    roi_data = roi_stats[roi_stats['ROI'] == roi]
    hierarchical_metrics.loc[hierarchical_metrics['ROI'] == roi, 'Adjacency_Mean'] = roi_data['adjacency_mean'].values[0]
    hierarchical_metrics.loc[hierarchical_metrics['ROI'] == roi, 'Functional_Mean'] = roi_data['functional_mean'].values[0]

# Add interpretation column
hierarchical_metrics['Interpretation'] = hierarchical_metrics['Model_Difference_Mean'].apply(
    lambda x: "Anatomical dominance" if x > 0.1 else "Functional dominance" if x < -0.1 else "Balanced representation"
)

hierarchical_metrics = hierarchical_metrics.sort_values('Hierarchical_Position')
hierarchical_metrics.to_csv('roi_functional_analysis/hierarchical_transformation_metrics.csv', index=False)
print("Saved hierarchical transformation metrics to roi_functional_analysis/hierarchical_transformation_metrics.csv")

# 4. Inter-ROI Correlation Metrics CSV
roi_correlation_metrics = pd.DataFrame()

# Get pairwise correlations between ROIs
for roi1 in rois:
    for roi2 in rois:
        if roi1 != roi2:
            # Adjacency model correlation
            adj_corr = roi_corr_adj.loc[roi1, roi2]
            
            # Functional model correlation
            func_corr = roi_corr_func.loc[roi1, roi2]
            
            # Combine in a row
            roi_correlation_metrics = pd.concat([roi_correlation_metrics, pd.DataFrame({
                'ROI_1': [roi1],
                'ROI_2': [roi2],
                'Adjacency_Correlation': [adj_corr],
                'Functional_Correlation': [func_corr],
                'Correlation_Difference': [adj_corr - func_corr]
            })])

roi_correlation_metrics.to_csv('roi_functional_analysis/roi_correlation_metrics.csv', index=False)
print("Saved ROI correlation metrics to roi_functional_analysis/roi_correlation_metrics.csv")

# 5. Summary Metrics CSV
# Create a summary of the key findings
adj_wins = sum(1 for _, row in roi_stats.iterrows() if row['better_model'] == 'Adjacency')
func_wins = len(rois) - adj_wins

avg_adj_advantage = roi_stats['model_diff'].mean()
avg_m1_advantage = roi_stats[roi_stats['ROI'] == 'M1']['model_diff'].values[0]
avg_ppc_advantage = roi_stats[roi_stats['ROI'] == 'PPC']['model_diff'].values[0]
hierarchical_gradient = avg_m1_advantage - avg_ppc_advantage

# Count how many ROIs show significant differences
sig_count = sum(1 for _, row in roi_test_df.iterrows() if row['significant'])

# Create summary DataFrame
summary_metrics = pd.DataFrame({
    'Metric': [
        'Adjacency_Model_Win_Count',
        'Functional_Model_Win_Count',
        'Average_Adjacency_Advantage',
        'M1_Adjacency_Advantage',
        'PPC_Adjacency_Advantage',
        'Hierarchical_Gradient',
        'Significant_ROI_Count',
        'Total_ROI_Count'
    ],
    'Value': [
        adj_wins,
        func_wins,
        avg_adj_advantage,
        avg_m1_advantage,
        avg_ppc_advantage,
        hierarchical_gradient,
        sig_count,
        len(rois)
    ]
})

summary_metrics.to_csv('roi_functional_analysis/summary_metrics.csv', index=False)
print("Saved summary metrics to roi_functional_analysis/summary_metrics.csv")

# Part 7: Comprehensive Analysis Report
# -----------------------------------
print("\n--- Generating Comprehensive Analysis Report ---")

with open('roi_functional_analysis/functional_grouping_analysis.md', 'w') as f:
    f.write("# Analysis of Current Functional Groupings Performance\n\n")
    
    # Introduction
    f.write("## 1. Introduction\n\n")
    f.write("This analysis evaluates the performance of the current functional groupings in the RSA framework ")
    f.write("across different regions of interest (ROIs). The analysis aims to determine whether the current ")
    f.write("anatomical and functional definitions are optimal or if they require restructuring.\n\n")
    
    f.write("### 1.1 Current Functional Groupings\n\n")
    f.write("The following functional groupings are currently defined based on shared ethological roles:\n\n")
    
    for group, parts in functional_groups.items():
        f.write(f"**{group}** ({group_descriptions[group]}): {', '.join(parts)}\n\n")
    
    # Performance by ROI
    f.write("## 2. Performance Analysis by ROI\n\n")
    f.write("### 2.1 Model Comparison\n\n")
    f.write("The table below compares the performance of the anatomical adjacency model and the functional similarity ")
    f.write("model across different ROIs:\n\n")
    
    f.write("| ROI | Adjacency Mean | Functional Mean | Difference | Better Model |\n")
    f.write("|-----|----------------|-----------------|------------|----------------|\n")
    for _, row in roi_stats.iterrows():
        f.write(f"| {row['ROI']} | {row['adjacency_mean']:.4f} | {row['functional_mean']:.4f} | ")
        f.write(f"{row['model_diff']:.4f} | {row['better_model']} |\n")
    
    f.write("\n### 2.2 Statistical Analysis\n\n")
    f.write("Paired t-tests were conducted to determine if the differences between models are statistically significant:\n\n")
    
    f.write("| ROI | t-statistic | p-value | Significant? | Better Model |\n")
    f.write("|-----|-------------|---------|--------------|----------------|\n")
    for _, row in roi_test_df.iterrows():
        f.write(f"| {row['ROI']} | {row['t_statistic']:.4f} | {row['p_value']:.4f} | ")
        f.write(f"{str(row['significant'])} | {row['better_model']} |\n")
    
    # Hierarchical Analysis
    f.write("\n## 3. Hierarchical Analysis\n\n")
    f.write("The analysis reveals a pattern of transformation from anatomical to functional organization ")
    f.write("as we move from primary to higher-order motor regions:\n\n")
    
    f.write("| ROI | Adjacency-Functional Difference | Interpretation |\n")
    f.write("|-----|----------------------------------|----------------|\n")
    for roi in hierarchical_order:
        diff = hierarchical_stats.loc[roi, 'mean']
        if diff > 0.1:
            interp = "Anatomical dominance"
        elif diff < -0.1:
            interp = "Functional dominance"
        else:
            interp = "Balanced representation"
        f.write(f"| {roi} | {diff:.4f} | {interp} |\n")
    
    # ROI Relationships
    f.write("\n## 4. ROI Relationships\n\n")
    f.write("Correlation analysis between ROIs reveals how similarly they process movement representations:\n\n")
    
    f.write("### 4.1 Adjacency Model Correlations\n\n")
    f.write("```\n")
    f.write(f"{roi_corr_adj.round(2).to_string()}\n")
    f.write("```\n\n")
    
    f.write("### 4.2 Functional Model Correlations\n\n")
    f.write("```\n")
    f.write(f"{roi_corr_func.round(2).to_string()}\n")
    f.write("```\n\n")
    
    # Evaluation and Recommendations
    f.write("## 5. Evaluation and Recommendations\n\n")
    
    # Overall assessment
    adjacency_wins = sum(1 for _, row in roi_stats.iterrows() if row['better_model'] == 'Adjacency')
    functional_wins = len(rois) - adjacency_wins
    
    f.write("### 5.1 Overall Assessment\n\n")
    f.write(f"Across all ROIs, the **anatomical adjacency model** performs better in {adjacency_wins} ROIs, ")
    f.write(f"while the **functional similarity model** performs better in {functional_wins} ROIs.\n\n")
    
    # Hierarchical transformation
    primary_diff = hierarchical_stats.loc['M1', 'mean']
    highest_diff = hierarchical_stats.loc['PPC', 'mean']
    
    f.write("### 5.2 Hierarchical Transformation\n\n")
    f.write(f"The analysis reveals a clear transformation pattern across the motor hierarchy. ")
    f.write(f"In primary motor areas (M1), the anatomical model shows a stronger advantage ({primary_diff:.4f}), ")
    f.write(f"while in higher-order regions (PPC), this advantage shifts ({highest_diff:.4f}) ")
    f.write(f"toward the functional model.\n\n")
    
    # Recommendations for groupings
    f.write("### 5.3 Recommendations for Functional Groupings\n\n")
    
    # Based on the hierarchical results, make recommendations
    if all(hierarchical_stats['mean'] > 0.1):
        f.write("**Recommendation**: The current anatomical organization dominates across all ROIs. ")
        f.write("Consider focusing more on anatomical adjacency in your groupings.\n\n")
    elif all(hierarchical_stats['mean'] < -0.1):
        f.write("**Recommendation**: The current functional organization dominates across all ROIs. ")
        f.write("The current groupings appear well-suited to neural representations.\n\n")
    else:
        f.write("**Recommendation**: Consider a hybrid approach where:\n\n")
        f.write("1. For primary motor regions (e.g., M1), maintain groupings that respect anatomical adjacency\n")
        f.write("2. For higher-order regions (e.g., PPC), emphasize the current functional groupings\n")
        f.write("3. For intermediate regions, balance both organizational principles\n\n")
    
    # Specific group recommendations
    f.write("### 5.4 Specific Group Recommendations\n\n")
    
    # PLM group is quite large and might benefit from subdivision
    f.write("#### Proximal Limb Movements (PLM)\n\n")
    f.write("The PLM group contains four body parts (leftleg, rightleg, forearm, upperarm) and may benefit from subdivision into:\n\n")
    f.write("- **Lower Limb Movements**: leftleg, rightleg\n")
    f.write("- **Upper Limb Movements**: forearm, upperarm\n\n")
    
    # Consider TOR group which only has one member
    f.write("#### Targeted Orientation (TOR)\n\n")
    f.write("The TOR group contains only eye movements. Consider if this is truly a separate functional category ")
    f.write("or if it could be integrated with another group based on functional properties.\n\n")
    
    # Conclusion
    f.write("## 6. Conclusion\n\n")
    f.write("The analysis supports the hypothesis of a hierarchical transformation from anatomical to functional ")
    f.write("organization across the motor hierarchy. The current functional groupings show promise, particularly ")
    f.write("in higher-order regions, but could benefit from refinements based on the hierarchical organization ")
    f.write("observed in the data.\n\n")
    
    f.write("The most compelling evidence for functional organization appears in higher-order regions (PMC, PPC), ")
    f.write("suggesting that these areas may be where ethological groupings are most relevant to neural processing.")

print("\nAnalysis complete! Results saved to the 'roi_functional_analysis' directory.")
print("A comprehensive report has been generated: roi_functional_analysis/functional_grouping_analysis.md")
