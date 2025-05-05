#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import os

# Create output directory for results
os.makedirs('functional_grouping_analysis', exist_ok=True)

print("Analyzing ROI functional groupings...")

# Define the path to the raw RSA data
rsa_data_path = 'results/rsa_results_raw.csv'

# Define the existing functional groupings for baseline comparison
# These are the current groupings we'll compare against
existing_groupings = {
    'Group_1': ['M1', 'SMA'],        # Primary motor areas
    'Group_2': ['PMC', 'PPC']        # Motor planning/preparation areas
}

print("Current functional groupings:")
for group, rois in existing_groupings.items():
    print(f"  {group}: {', '.join(rois)}")

# Load the raw RSA data
raw_data = pd.read_csv(rsa_data_path)
print(f"\nLoaded data: {raw_data.shape[0]} rows, {raw_data.shape[1]} columns")
print(raw_data.head())

# Extract unique subjects and ROIs
subjects = raw_data['subject'].unique()
rois = raw_data['ROI'].unique()
print(f"\nFound {len(subjects)} subjects and {len(rois)} ROIs")
print(f"ROIs: {', '.join(rois)}")

# Reshape the data for analysis - pivot to get ROIs as columns
adjacency_df = raw_data.pivot(index='subject', columns='ROI', values='adjacency_rho')
functional_df = raw_data.pivot(index='subject', columns='ROI', values='functional_rho')

print("\nAdjacency correlation data (sample):")
print(adjacency_df.head())
print("\nFunctional correlation data (sample):")
print(functional_df.head())

# 1. BASELINE ANALYSIS - Evaluate existing groupings
# Calculate within-group similarity for existing groupings

def evaluate_grouping(data_df, grouping, correlation_type):
    """Evaluate the quality of a grouping by measuring within-group and between-group similarity"""
    within_group_similarity = {}
    between_group_similarity = {}
    
    # Calculate within-group similarity
    for group, group_rois in grouping.items():
        if len(group_rois) > 1:  # Need at least 2 ROIs to calculate correlation
            # Extract the data for the group's ROIs
            group_data = data_df[group_rois]
            
            # Calculate pairwise correlations between ROIs in the group
            correlations = []
            for i, roi1 in enumerate(group_rois):
                for roi2 in group_rois[i+1:]:
                    corr = np.corrcoef(group_data[roi1], group_data[roi2])[0, 1]
                    correlations.append(corr)
            
            within_group_similarity[group] = np.mean(correlations) if correlations else np.nan
    
    # Calculate between-group similarity
    groups = list(grouping.keys())
    for i, group1 in enumerate(groups):
        for group2 in groups[i+1:]:
            rois1 = grouping[group1]
            rois2 = grouping[group2]
            
            correlations = []
            for roi1 in rois1:
                for roi2 in rois2:
                    corr = np.corrcoef(data_df[roi1], data_df[roi2])[0, 1]
                    correlations.append(corr)
            
            between_key = f"{group1} - {group2}"
            between_group_similarity[between_key] = np.mean(correlations) if correlations else np.nan
    
    # Calculate overall grouping quality
    # Good groupings have high within-group similarity and low between-group similarity
    avg_within = np.nanmean(list(within_group_similarity.values()))
    avg_between = np.nanmean(list(between_group_similarity.values()))
    
    # A simple quality score: difference between within and between group similarities
    quality_score = avg_within - avg_between
    
    return {
        'within_group': within_group_similarity,
        'between_group': between_group_similarity,
        'avg_within': avg_within,
        'avg_between': avg_between,
        'quality_score': quality_score
    }

# Evaluate existing groupings
print("\nEvaluating existing functional groupings...")
adjacency_evaluation = evaluate_grouping(adjacency_df, existing_groupings, 'Adjacency')
functional_evaluation = evaluate_grouping(functional_df, existing_groupings, 'Functional')

print("\nExisting groupings - Adjacency correlation:")
print(f"  Within-group similarity: {adjacency_evaluation['avg_within']:.4f}")
print(f"  Between-group similarity: {adjacency_evaluation['avg_between']:.4f}")
print(f"  Quality score: {adjacency_evaluation['quality_score']:.4f}")

print("\nExisting groupings - Functional correlation:")
print(f"  Within-group similarity: {functional_evaluation['avg_within']:.4f}")
print(f"  Between-group similarity: {functional_evaluation['avg_between']:.4f}")
print(f"  Quality score: {functional_evaluation['quality_score']:.4f}")

# 2. ROI SIMILARITY ANALYSIS
# Compute ROI-to-ROI correlation matrices to see how similar ROIs are to each other

# Correlation based on adjacency data
adjacency_corr = adjacency_df.corr()
print("\nROI correlations based on adjacency data:")
print(adjacency_corr)

# Correlation based on functional data
functional_corr = functional_df.corr()
print("\nROI correlations based on functional data:")
print(functional_corr)

# Combined correlation (average of adjacency and functional)
combined_corr = (adjacency_corr + functional_corr) / 2
print("\nROI correlations based on combined data:")
print(combined_corr)

# Create heatmaps for visualization
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.heatmap(adjacency_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('ROI Correlations - Adjacency')

plt.subplot(1, 3, 2)
sns.heatmap(functional_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('ROI Correlations - Functional')

plt.subplot(1, 3, 3)
sns.heatmap(combined_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('ROI Correlations - Combined')

plt.tight_layout()
plt.savefig('functional_grouping_analysis/roi_correlation_heatmaps.png')
print("Saved correlation heatmaps to functional_grouping_analysis/roi_correlation_heatmaps.png")

# 3. HIERARCHICAL CLUSTERING ANALYSIS
# Use hierarchical clustering to suggest alternative groupings

def perform_hierarchical_clustering(corr_matrix, rois, name, linkage_method='ward'):
    """Perform hierarchical clustering and analyze different cluster counts"""
    # Create linkage matrix
    link_matrix = linkage(corr_matrix, method=linkage_method)
    
    # Plot dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(link_matrix, labels=rois, leaf_font_size=12)
    plt.title(f'Hierarchical Clustering of ROIs - {name}')
    plt.ylabel('Distance')
    plt.savefig(f'functional_grouping_analysis/dendrogram_{name.lower()}.png')
    
    # Try different numbers of clusters and find optimal
    silhouette_scores = []
    ch_scores = []  # Calinski-Harabasz scores
    groupings = {}
    
    # Skip if too few ROIs
    if len(rois) <= 2:
        return None, None, None
    
    # For small number of ROIs, just try 2 clusters
    if len(rois) <= 4:
        possible_clusters = [2]
    else:
        possible_clusters = range(2, min(len(rois), 5))
    
    for n_clusters in possible_clusters:
        # Perform clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = clustering.fit_predict(corr_matrix)
        
        # Create grouping
        grouping = {}
        for i, roi in enumerate(rois):
            group = f"Group_{labels[i]+1}"
            if group not in grouping:
                grouping[group] = []
            grouping[group].append(roi)
        
        groupings[n_clusters] = grouping
        
        # Calculate silhouette score if more than one cluster
        if n_clusters > 1:
            sil_score = silhouette_score(corr_matrix, labels)
            silhouette_scores.append((n_clusters, sil_score))
            
            # Calculate Calinski-Harabasz score
            ch_score = calinski_harabasz_score(corr_matrix, labels)
            ch_scores.append((n_clusters, ch_score))
    
    # Find optimal number of clusters
    best_n_clusters = None
    if silhouette_scores:
        best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    else:
        best_n_clusters = 2  # Default if can't calculate scores
    
    return groupings[best_n_clusters], best_n_clusters, silhouette_scores

# Perform clustering on the correlation matrices
print("\nPerforming hierarchical clustering to suggest alternative groupings...")

adjacency_cluster_result = perform_hierarchical_clustering(adjacency_corr.values, list(adjacency_corr.index), 'Adjacency')
functional_cluster_result = perform_hierarchical_clustering(functional_corr.values, list(functional_corr.index), 'Functional')
combined_cluster_result = perform_hierarchical_clustering(combined_corr.values, list(combined_corr.index), 'Combined')

if adjacency_cluster_result and functional_cluster_result and combined_cluster_result:
    adjacency_grouping, adj_n_clusters, adj_silhouette = adjacency_cluster_result
    functional_grouping, func_n_clusters, func_silhouette = functional_cluster_result
    combined_grouping, comb_n_clusters, comb_silhouette = combined_cluster_result
    
    # Print the suggested groupings
    print("\nSuggested functional groupings based on hierarchical clustering:")
    
    print(f"\nAdjacency-based grouping ({adj_n_clusters} clusters):")
    for group, members in adjacency_grouping.items():
        print(f"  {group}: {', '.join(members)}")
    
    print(f"\nFunctional-based grouping ({func_n_clusters} clusters):")
    for group, members in functional_grouping.items():
        print(f"  {group}: {', '.join(members)}")
    
    print(f"\nCombined-based grouping ({comb_n_clusters} clusters):")
    for group, members in combined_grouping.items():
        print(f"  {group}: {', '.join(members)}")
    
    # 4. EVALUATE SUGGESTED GROUPINGS
    # Compare the suggested groupings against the existing ones
    
    # Evaluate new groupings
    adjacency_new_eval = evaluate_grouping(adjacency_df, adjacency_grouping, 'Adjacency')
    functional_new_eval = evaluate_grouping(functional_df, functional_grouping, 'Functional')
    combined_new_eval = evaluate_grouping(functional_df, combined_grouping, 'Combined')
    
    # Compare quality scores
    quality_comparison = pd.DataFrame({
        'Grouping': ['Existing', 'Adjacency Suggested', 'Functional Suggested', 'Combined Suggested'],
        'Adjacency Quality': [
            adjacency_evaluation['quality_score'],
            adjacency_new_eval['quality_score'],
            evaluate_grouping(adjacency_df, functional_grouping, 'Adjacency')['quality_score'],
            evaluate_grouping(adjacency_df, combined_grouping, 'Adjacency')['quality_score']
        ],
        'Functional Quality': [
            functional_evaluation['quality_score'],
            evaluate_grouping(functional_df, adjacency_grouping, 'Functional')['quality_score'],
            functional_new_eval['quality_score'],
            combined_new_eval['quality_score']
        ]
    })
    
    # Add average quality column
    quality_comparison['Average Quality'] = (quality_comparison['Adjacency Quality'] + 
                                           quality_comparison['Functional Quality']) / 2
    
    print("\nQuality comparison of different groupings:")
    print(quality_comparison)
    
    # Save quality comparison to CSV
    quality_comparison.to_csv('functional_grouping_analysis/grouping_quality_comparison.csv', index=False)
    
    # 5. FINAL RECOMMENDATION
    # Select the best grouping based on quality scores
    
    best_grouping_idx = quality_comparison['Average Quality'].idxmax()
    best_grouping_name = quality_comparison.loc[best_grouping_idx, 'Grouping']
    
    print("\nRECOMMENDED FUNCTIONAL GROUPING:")
    if best_grouping_name == 'Existing':
        print("The existing functional groupings appear to be optimal and should be retained.")
        recommended_grouping = existing_groupings
    elif best_grouping_name == 'Adjacency Suggested':
        print("The adjacency-based grouping is recommended for restructuring functional groups:")
        for group, members in adjacency_grouping.items():
            print(f"  {group}: {', '.join(members)}")
        recommended_grouping = adjacency_grouping
    elif best_grouping_name == 'Functional Suggested':
        print("The functional-based grouping is recommended for restructuring functional groups:")
        for group, members in functional_grouping.items():
            print(f"  {group}: {', '.join(members)}")
        recommended_grouping = functional_grouping
    else:  # Combined Suggested
        print("The combined (adjacency + functional) grouping is recommended for restructuring functional groups:")
        for group, members in combined_grouping.items():
            print(f"  {group}: {', '.join(members)}")
        recommended_grouping = combined_grouping
    
    # 6. GENERATE COMPREHENSIVE REPORT
    with open('functional_grouping_analysis/functional_grouping_recommendation.txt', 'w') as f:
        f.write("# Functional Grouping Analysis and Recommendations\n\n")
        
        f.write("## Dataset Overview\n")
        f.write(f"This analysis is based on RSA results from {len(subjects)} subjects across {len(rois)} ROIs.\n")
        f.write(f"ROIs analyzed: {', '.join(rois)}\n\n")
        
        f.write("## Current Functional Groupings\n")
        for group, members in existing_groupings.items():
            f.write(f"- {group}: {', '.join(members)}\n")
        
        f.write(f"\nQuality of current groupings:\n")
        f.write(f"- Adjacency correlation: {adjacency_evaluation['quality_score']:.4f}\n")
        f.write(f"- Functional correlation: {functional_evaluation['quality_score']:.4f}\n")
        f.write(f"- Average quality: {(adjacency_evaluation['quality_score'] + functional_evaluation['quality_score'])/2:.4f}\n\n")
        
        f.write("## ROI Correlation Analysis\n")
        f.write("### Adjacency Correlation Matrix:\n")
        f.write(f"{adjacency_corr.to_string()}\n\n")
        
        f.write("### Functional Correlation Matrix:\n")
        f.write(f"{functional_corr.to_string()}\n\n")
        
        f.write("### Combined Correlation Matrix:\n")
        f.write(f"{combined_corr.to_string()}\n\n")
        
        f.write("## Suggested Alternative Groupings\n")
        
        f.write(f"\n### Adjacency-based grouping ({adj_n_clusters} clusters):\n")
        for group, members in adjacency_grouping.items():
            f.write(f"- {group}: {', '.join(members)}\n")
        f.write(f"Quality score: {adjacency_new_eval['quality_score']:.4f}\n")
        
        f.write(f"\n### Functional-based grouping ({func_n_clusters} clusters):\n")
        for group, members in functional_grouping.items():
            f.write(f"- {group}: {', '.join(members)}\n")
        f.write(f"Quality score: {functional_new_eval['quality_score']:.4f}\n")
        
        f.write(f"\n### Combined-based grouping ({comb_n_clusters} clusters):\n")
        for group, members in combined_grouping.items():
            f.write(f"- {group}: {', '.join(members)}\n")
        f.write(f"Quality score: {combined_new_eval['quality_score']:.4f}\n")
        
        f.write("\n## Final Recommendation\n")
        f.write(f"Based on comprehensive analysis of both adjacency and functional correlation patterns, ")
        f.write(f"the {best_grouping_name} provides the optimal functional organization with ")
        f.write(f"a quality score of {quality_comparison.loc[best_grouping_idx, 'Average Quality']:.4f}.\n\n")
        
        f.write("### Recommended Functional Grouping:\n")
        for group, members in recommended_grouping.items():
            f.write(f"- {group}: {', '.join(members)}\n")
        
        f.write("\n## Neuroanatomical Interpretation\n")
        # Interpret the results based on neuroscience knowledge
        if any(set(['M1', 'SMA']).issubset(set(members)) for members in recommended_grouping.values()):
            f.write("M1 (Primary Motor Cortex) and SMA (Supplementary Motor Area) show similar correlation patterns, ")
            f.write("which aligns with their established roles in motor execution and motor planning, respectively.\n")
        
        if any(set(['PMC', 'PPC']).issubset(set(members)) for members in recommended_grouping.values()):
            f.write("PMC (Premotor Cortex) and PPC (Posterior Parietal Cortex) demonstrate similar patterns, ")
            f.write("consistent with their roles in motor preparation and visuomotor integration.\n")
        
        f.write("\nThese groupings reflect the functional organization of the motor network, ")
        f.write("where regions with similar roles in motor control and planning tend to show similar correlation patterns across subjects.\n")

    print("\nSaved comprehensive report to functional_grouping_analysis/functional_grouping_recommendation.txt")
    print("\nAnalysis complete! All results saved to the 'functional_grouping_analysis' directory.")
else:
    print("Not enough ROIs to perform meaningful clustering analysis.")
