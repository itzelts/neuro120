#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import os

# Create output directory for plots and results
os.makedirs('full_analysis_results', exist_ok=True)

print("Analyzing full RSA dataset...")

# Load the raw RSA data
raw_data = pd.read_csv('rsa_results_raw.csv')
print(f"Raw data shape: {raw_data.shape}")
print("Raw data sample:")
print(raw_data.head(8))

# Extract unique subjects and ROIs
subjects = raw_data['subject'].unique()
rois = raw_data['ROI'].unique()
print(f"\nFound {len(subjects)} subjects and {len(rois)} ROIs")
print(f"Subjects: {len(subjects)} subjects")
print(f"ROIs: {', '.join(rois)}")

# Reshape the data for analysis
# We want separate dataframes for adjacency and functional data
# Each with subjects as rows and ROIs as columns

# Pivot the data to wide format
adjacency_df = raw_data.pivot(index='subject', columns='ROI', values='adjacency_rho')
functional_df = raw_data.pivot(index='subject', columns='ROI', values='functional_rho')

print("\nReshaped adjacency data:")
print(adjacency_df.head())

print("\nReshaped functional data:")
print(functional_df.head())

# Check for missing values
print("\nChecking for missing values:")
print(f"Missing values in adjacency data: {adjacency_df.isna().sum().sum()}")
print(f"Missing values in functional data: {functional_df.isna().sum().sum()}")

# Handle missing values if necessary (fill with mean)
if adjacency_df.isna().sum().sum() > 0:
    adjacency_df = adjacency_df.fillna(adjacency_df.mean())
    print("Filled missing adjacency values with column means")

if functional_df.isna().sum().sum() > 0:
    functional_df = functional_df.fillna(functional_df.mean())
    print("Filled missing functional values with column means")

# Calculate mean values for each ROI
adjacency_means = adjacency_df.mean()
functional_means = functional_df.mean()

print("\nMean adjacency correlations by ROI:")
for roi in rois:
    print(f"{roi}: {adjacency_means[roi]:.4f}")

print("\nMean functional correlations by ROI:")
for roi in rois:
    print(f"{roi}: {functional_means[roi]:.4f}")

# Plot the mean values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x=adjacency_means.index, y=adjacency_means.values)
plt.title('Mean Adjacency Correlation by ROI')
plt.ylabel('Correlation')
plt.ylim(0, 1)
plt.grid(True, axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
sns.barplot(x=functional_means.index, y=functional_means.values)
plt.title('Mean Functional Correlation by ROI')
plt.ylabel('Correlation')
plt.ylim(0, 1)
plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('full_analysis_results/mean_correlations_by_roi.png')
print("Saved mean correlations by ROI to full_analysis_results/mean_correlations_by_roi.png")

# Compute correlation between ROIs across subjects
# For each subject, we have a correlation value for each ROI
# We want to see if ROIs show similar patterns across subjects

# Method 1: Correlation of ROI patterns across subjects
adjacency_corr = adjacency_df.corr()
functional_corr = functional_df.corr()

print("\nAdjacency correlation matrix between ROIs (across subjects):")
print(adjacency_corr)

print("\nFunctional correlation matrix between ROIs (across subjects):")
print(functional_corr)

# Create heatmaps
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(adjacency_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('ROI Correlations - Adjacency')

plt.subplot(1, 2, 2)
sns.heatmap(functional_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('ROI Correlations - Functional')

plt.tight_layout()
plt.savefig('full_analysis_results/roi_correlation_heatmaps.png')
print("Saved correlation heatmaps to full_analysis_results/roi_correlation_heatmaps.png")

# Combined correlation matrix (average of adjacency and functional)
combined_corr = (adjacency_corr + functional_corr) / 2

print("\nCombined correlation matrix (average of adjacency and functional):")
print(combined_corr)

# Plot combined correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(combined_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('ROI Correlations - Combined')
plt.savefig('full_analysis_results/combined_correlation_heatmap.png')
print("Saved combined correlation heatmap to full_analysis_results/combined_correlation_heatmap.png")

# Method 2: Alternative approach - calculate pairwise distances between ROIs
# Create distance matrices based on adjacency and functional data
def compute_distance_matrix(data_df, method='correlation'):
    """Compute distance matrix between ROIs based on specified metric"""
    # Transpose so that ROIs are observations and subjects are features
    roi_data = data_df.T.values
    # Compute pairwise distances
    distances = pdist(roi_data, metric=method)
    # Convert to square matrix
    distance_matrix = squareform(distances)
    return distance_matrix

# Compute distance matrices
adjacency_distances = compute_distance_matrix(adjacency_df)
functional_distances = compute_distance_matrix(functional_df)

# Convert to similarity (1 - normalized distance)
adjacency_distances_norm = adjacency_distances / adjacency_distances.max()
functional_distances_norm = functional_distances / functional_distances.max()

adjacency_similarity = 1 - adjacency_distances_norm
functional_similarity = 1 - functional_distances_norm

# Create dataframes for visualization
adjacency_sim_df = pd.DataFrame(adjacency_similarity, index=rois, columns=rois)
functional_sim_df = pd.DataFrame(functional_similarity, index=rois, columns=rois)

print("\nAdjacency similarity matrix (based on distances):")
print(adjacency_sim_df)

print("\nFunctional similarity matrix (based on distances):")
print(functional_sim_df)

# Function to perform hierarchical clustering and create dendrograms
def perform_clustering(matrix, rois, name, method='ward'):
    """Perform hierarchical clustering and return cluster assignments"""
    # Compute linkage
    linkage_matrix = linkage(matrix, method=method)
    
    # Plot dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=rois, leaf_font_size=12)
    plt.title(f'Hierarchical Clustering - {name}')
    plt.ylabel('Distance')
    plt.savefig(f'full_analysis_results/hierarchical_clustering_{name.lower()}.png')
    print(f"Saved hierarchical clustering for {name} to full_analysis_results/hierarchical_clustering_{name.lower()}.png")
    
    # Try different numbers of clusters and find optimal using silhouette score
    max_clusters = min(len(rois) - 1, 8)  # Don't try more clusters than ROIs-1 or 8
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering.fit_predict(matrix)
        
        # Only compute silhouette score if more than one cluster
        if len(np.unique(cluster_labels)) > 1:
            score = silhouette_score(matrix, cluster_labels)
            silhouette_scores.append((n_clusters, score))
    
    # Find optimal number of clusters
    if silhouette_scores:
        best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    else:
        best_n_clusters = 2  # Default to 2 clusters if silhouette analysis fails
    
    # Get the final clusters
    clustering = AgglomerativeClustering(n_clusters=best_n_clusters)
    cluster_labels = clustering.fit_predict(matrix)
    
    # Create dictionary of clusters
    clusters = {}
    for i, roi in enumerate(rois):
        cluster = f"Group_{cluster_labels[i]}"
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(roi)
    
    return clusters, best_n_clusters, silhouette_scores

# Perform clustering on correlation matrices
adjacency_clusters, adj_n_clusters, adj_scores = perform_clustering(adjacency_corr.values, rois, 'Adjacency')
functional_clusters, func_n_clusters, func_scores = perform_clustering(functional_corr.values, rois, 'Functional')
combined_clusters, comb_n_clusters, comb_scores = perform_clustering(combined_corr.values, rois, 'Combined')

# Also perform clustering on distance matrices
adjacency_dist_clusters, adj_dist_n_clusters, _ = perform_clustering(
    adjacency_distances, rois, 'Adjacency_Distance')
functional_dist_clusters, func_dist_n_clusters, _ = perform_clustering(
    functional_distances, rois, 'Functional_Distance')

# Plot silhouette scores if available
if adj_scores and func_scores and comb_scores:
    plt.figure(figsize=(10, 6))
    
    if adj_scores:
        plt.plot([x[0] for x in adj_scores], [x[1] for x in adj_scores], 'o-', label='Adjacency')
    
    if func_scores:
        plt.plot([x[0] for x in func_scores], [x[1] for x in func_scores], 'o-', label='Functional')
    
    if comb_scores:
        plt.plot([x[0] for x in comb_scores], [x[1] for x in comb_scores], 'o-', label='Combined')
    
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.legend()
    plt.grid(True)
    plt.savefig('full_analysis_results/silhouette_scores.png')
    print("Saved silhouette scores to full_analysis_results/silhouette_scores.png")

# Display the clustering results
print("\nSuggested ROI Groupings (Based on Correlation Analysis):")
print(f"\nAdjacency-based groupings ({adj_n_clusters} clusters):")
for group, members in adjacency_clusters.items():
    print(f"  {group}: {', '.join(members)}")

print(f"\nFunctional-based groupings ({func_n_clusters} clusters):")
for group, members in functional_clusters.items():
    print(f"  {group}: {', '.join(members)}")

print(f"\nCombined-based groupings ({comb_n_clusters} clusters):")
for group, members in combined_clusters.items():
    print(f"  {group}: {', '.join(members)}")

# Additional analysis: ROI-specific patterns
# Check if certain subjects show consistently higher correlations for specific ROIs

# Create subplots for ROI-specific patterns
plt.figure(figsize=(15, 10))

# Plot distribution of adjacency correlations for each ROI
for i, roi in enumerate(rois):
    plt.subplot(2, len(rois), i + 1)
    sns.histplot(adjacency_df[roi], kde=True)
    plt.title(f'Adjacency: {roi}')
    plt.xlabel('Correlation')
    plt.xlim(-0.2, 1)

# Plot distribution of functional correlations for each ROI
for i, roi in enumerate(rois):
    plt.subplot(2, len(rois), i + 1 + len(rois))
    sns.histplot(functional_df[roi], kde=True)
    plt.title(f'Functional: {roi}')
    plt.xlabel('Correlation')
    plt.xlim(-0.2, 1)

plt.tight_layout()
plt.savefig('full_analysis_results/roi_distributions.png')
print("Saved ROI correlation distributions to full_analysis_results/roi_distributions.png")

# Create a summary table
summary = pd.DataFrame({
    'ROI': rois,
    'Adjacency Mean': [adjacency_means[roi] for roi in rois],
    'Functional Mean': [functional_means[roi] for roi in rois],
    'Adjacency Group': [''] * len(rois),
    'Functional Group': [''] * len(rois),
    'Combined Group': [''] * len(rois)
})

# Fill in group information
for i, roi in enumerate(rois):
    # Fill in group information
    for group, members in adjacency_clusters.items():
        if roi in members:
            summary.loc[i, 'Adjacency Group'] = group
    
    for group, members in functional_clusters.items():
        if roi in members:
            summary.loc[i, 'Functional Group'] = group
    
    for group, members in combined_clusters.items():
        if roi in members:
            summary.loc[i, 'Combined Group'] = group

# Save the summary table
summary.to_csv('full_analysis_results/roi_grouping_summary.csv', index=False)
print("Saved ROI grouping summary to full_analysis_results/roi_grouping_summary.csv")

# Generate a comprehensive report
with open('full_analysis_results/functional_grouping_recommendations.txt', 'w') as f:
    f.write("# Functional Grouping Recommendations\n\n")
    
    f.write("## Dataset Overview\n")
    f.write(f"This analysis is based on RSA results from {len(subjects)} subjects across {len(rois)} ROIs.\n")
    f.write(f"ROIs analyzed: {', '.join(rois)}\n\n")
    
    f.write("## Analysis Summary\n")
    f.write("We analyzed both adjacency and functional correlation patterns across these regions.\n\n")
    
    f.write("### Mean Correlation Values\n")
    f.write("#### Adjacency Correlations:\n")
    for roi in rois:
        f.write(f"- {roi}: {adjacency_means[roi]:.4f}\n")
    
    f.write("\n#### Functional Correlations:\n")
    for roi in rois:
        f.write(f"- {roi}: {functional_means[roi]:.4f}\n")
    
    f.write("\n### ROI Relationships\n")
    f.write("#### Adjacency Correlation Matrix:\n")
    f.write(f"{adjacency_corr.to_string()}\n\n")
    
    f.write("#### Functional Correlation Matrix:\n")
    f.write(f"{functional_corr.to_string()}\n\n")
    
    f.write("#### Combined Correlation Matrix:\n")
    f.write(f"{combined_corr.to_string()}\n\n")
    
    f.write("## Suggested Functional Groupings\n")
    
    f.write(f"\n### Adjacency-Based Groupings ({adj_n_clusters} clusters):\n")
    for group, members in adjacency_clusters.items():
        f.write(f"- {group}: {', '.join(members)}\n")
    
    f.write(f"\n### Functional-Based Groupings ({func_n_clusters} clusters):\n")
    for group, members in functional_clusters.items():
        f.write(f"- {group}: {', '.join(members)}\n")
    
    f.write(f"\n### Combined-Based Groupings ({comb_n_clusters} clusters):\n")
    for group, members in combined_clusters.items():
        f.write(f"- {group}: {', '.join(members)}\n")
    
    f.write("\n## Recommendations\n")
    
    # Determine which grouping method has the best statistical support
    if adj_scores and func_scores and comb_scores:
        best_adjacency = max(adj_scores, key=lambda x: x[1]) if adj_scores else (0, 0)
        best_functional = max(func_scores, key=lambda x: x[1]) if func_scores else (0, 0)
        best_combined = max(comb_scores, key=lambda x: x[1]) if comb_scores else (0, 0)
        
        best_method = max(
            ("Adjacency", best_adjacency[1]),
            ("Functional", best_functional[1]),
            ("Combined", best_combined[1]),
            key=lambda x: x[1]
        )
        
        f.write(f"Based on silhouette analysis, the {best_method[0]}-based grouping provides the most statistically robust clustering with a score of {best_method[1]:.4f}.\n\n")
        
        # Provide specific recommendations
        if best_method[0] == "Adjacency":
            f.write("We recommend using the Adjacency-based groupings for further analysis, which groups regions according to their physical connectivity patterns.\n")
            groups_to_use = adjacency_clusters
        elif best_method[0] == "Functional":
            f.write("We recommend using the Functional-based groupings for further analysis, which groups regions according to their functional similarity patterns.\n")
            groups_to_use = functional_clusters
        else:
            f.write("We recommend using the Combined-based groupings for further analysis, which balances both physical and functional connectivity patterns.\n")
            groups_to_use = combined_clusters
    else:
        # If silhouette analysis couldn't be performed, recommend combined grouping
        f.write("Based on the analysis of correlation patterns, we recommend using the Combined-based groupings, which balance both physical and functional connectivity patterns.\n")
        groups_to_use = combined_clusters
    
    f.write("\nThese groupings reflect common patterns in the RSA results across subjects and provide a data-driven approach to redefining functional groups.\n\n")
    
    # Neuroanatomical interpretation
    f.write("## Neuroanatomical Interpretation\n")
    
    # M1 and SMA often group together
    if any(set(['M1', 'SMA']).issubset(set(members)) for members in groups_to_use.values()):
        f.write("M1 and SMA show similar correlation patterns, which aligns with their established roles in motor execution and motor planning, respectively.\n")
    
    # PMC and PPC often show relationship
    if any(set(['PMC', 'PPC']).issubset(set(members)) for members in groups_to_use.values()):
        f.write("PMC and PPC demonstrate similar patterns, consistent with their roles in motor preparation and visuomotor integration.\n")
    
    f.write("\nThese groupings may reflect the functional organization of the motor network, where regions with similar roles in motor control and planning tend to show similar correlation patterns across subjects.\n")
    
    # Include a summary table
    f.write("\n## Summary of ROI Properties\n")
    f.write(summary.to_string(index=False))

print("Saved detailed recommendations to full_analysis_results/functional_grouping_recommendations.txt")
print("\nAnalysis complete! All results saved to the 'full_analysis_results' directory.")
