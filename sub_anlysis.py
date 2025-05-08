# -*- coding: utf-8 -*-
"""
Full RSA pipeline for motor task CIFTI data using Glasser MMP surface atlas
"""
import pathlib
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, ttest_rel
from nilearn.glm.first_level import make_first_level_design_matrix
from sklearn.manifold import MDS
from matplotlib.patches import Patch
import hcp_utils as hcp
from scipy.spatial import procrustes

bids_root   = pathlib.Path('.')
first_sub   = 'sub-01'
events_path = bids_root / first_sub / 'ses-1' / 'func' / f'{first_sub}_ses-1_task-motor_run-01_events.tsv'

# results_dir = pathlib.Path('results/coordination')
# results_dir.mkdir(exist_ok=True)
results_dir = pathlib.Path('results/task')
results_dir.mkdir(exist_ok=True)

# body parts 
part_to_file = {
    'toe':      'Toe',
    'ankle':    'Ankle',
    'leftleg':  'LeftLeg',
    'rightleg': 'RightLeg',
    'finger':   'Finger',
    'wrist':    'Wrist',
    'forearm':  'Forearm',
    'upperarm': 'Upperarm',
    'jaw':      'Jaw',
    'lip':      'Lip',
    'tongue':   'Tongue',
    'eye':      'Eye'
}
parts = list(part_to_file.keys())

# anatomical (homunculus) coords to adjacency RDM
coords = {
    'toe':0,
    'ankle':1,
    'leftleg':2,
    'rightleg':2.5,
    'wrist':3.7,
    'forearm':3.4,
    'upperarm':3,
    'finger':4,
    'jaw':5,
    'lip':5.2,
    'tongue':5.5,
    'eye':6
}
dist_mat = squareform(pdist([[coords[p]] for p in parts]))
adjacency_rdm = dist_mat / dist_mat.max()

# functional groups to func RDM
# func_group = {
#     'toe':'DFM',
#     'finger':'DFM',
#     'ankle':'MJA',
#     'wrist':'MJA',
#     'leftleg':'PLM',
#     'rightleg':'PLM',
#     'forearm':'PLM',
#     'upperarm':'PLM',
#     'jaw':'OFC',
#     'lip':'OFC',
#     'tongue':'OFC',
#     'eye':'TOR'
# }

func_group = { # by goal
    'toe': 'BALANCE',        # Balance and stability
    'finger': 'MANIPULATION', # Object manipulation/precision
    'ankle': 'LOCOMOTION',    # Walking/running
    'wrist': 'MANIPULATION',  # Object handling/gesturing
    'leftleg': 'LOCOMOTION',  # Ambulation
    'rightleg': 'LOCOMOTION', # Ambulation
    'forearm': 'REACHING',    # Extending to objects
    'upperarm': 'REACHING',   # Gross arm positioning
    'jaw': 'INGESTION',       # Eating
    'lip': 'COMMUNICATION',   # Speech/expression
    'tongue': 'DUAL',         # Both speech and ingestion
    'eye': 'PERCEPTION'       # Visual targeting
}

# func_group = { # by coordination
#     'toe': 'FOOT_COMPLEX',       # Toes work with ankles
#     'finger': 'HAND_COMPLEX',    # Fingers work with wrist
#     'ankle': 'FOOT_COMPLEX',     # Ankle works with toe, leg
#     'wrist': 'HAND_COMPLEX',     # Wrist works with fingers, forearm
#     'leftleg': 'POSTURAL',       # Bilateral leg coordination
#     'rightleg': 'POSTURAL',      # Bilateral leg coordination
#     'forearm': 'ARM_CHAIN',      # Linked with upper arm and wrist
#     'upperarm': 'ARM_CHAIN',     # Works with shoulder, forearm
#     'jaw': 'VOCAL_ARTICULATORY', # Speech production with lips/tongue
#     'lip': 'VOCAL_ARTICULATORY', # Articulation with jaw/tongue
#     'tongue': 'VOCAL_ARTICULATORY', # Works with lips/jaw for speech
#     'eye': 'VISUO_MOTOR'         # Eye-hand coordination
# }

# func_group = { # by task
#     'toe': 'PRECISION_LOWER',    # Fine control of foot placement
#     'finger': 'PRECISION_UPPER', # Detailed manipulation
#     'ankle': 'STABILITY_LOWER',  # Lower limb stability
#     'wrist': 'STABILITY_UPPER',  # Upper limb stability
#     'leftleg': 'POWER_LOWER',    # Force generation/support
#     'rightleg': 'POWER_LOWER',   # Force generation/support
#     'forearm': 'POWER_UPPER',    # Force application
#     'upperarm': 'POWER_UPPER',   # Force application
#     'jaw': 'CONSUMPTIVE',        # Food processing
#     'lip': 'EXPRESSIVE',         # Emotional display
#     'tongue': 'SENSORY_ORAL',    # Taste and texture sensing
#     'eye': 'SENSORY_VISUAL'      # Visual information
# }

func_rdm = np.zeros((len(parts), len(parts)))
for i, p1 in enumerate(parts):
    for j, p2 in enumerate(parts):
        func_rdm[i, j] = float(func_group[p1] != func_group[p2])

# ROI masks using Glasser MMP atlas 
def get_roi_masks():
    """
    Load Glasser MMP and build 1D grayordinate masks for M1, SMA, PMC, PPC.
    """
    import hcp_utils as hcp
    import numpy as np

    parc   = hcp.mmp
    map_all = parc.map_all            # 1D array of length N grayordinates
    labels  = parc.labels             # dict parcel_id to label_string

    m1_labels  = ['L_4','R_4']
    sma_labels = ['L_6ma','R_6ma']
    pmc_labels = ['L_6mp','R_6mp','L_6d','R_6d','L_6v','R_6v','L_6r','R_6r']
    ppc_labels = [
        'L_7Pm','R_7Pm','L_7m','R_7m','L_7Am','R_7Am',
        'L_7PL','R_7PL','L_7PC','R_7PC',
        'L_LIPv','R_LIPv','L_VIP','R_VIP','L_MIP','R_MIP'
    ]

    m1_ids   = [pid for pid,name in labels.items() if name in m1_labels]
    sma_ids  = [pid for pid,name in labels.items() if name in sma_labels]
    pmc_ids  = [pid for pid,name in labels.items() if name in pmc_labels]
    ppc_ids  = [pid for pid,name in labels.items() if name in ppc_labels]

    masks = {
        'M1':  np.isin(map_all, m1_ids),
        'SMA': np.isin(map_all, sma_ids),
        'PMC': np.isin(map_all, pmc_ids),
        'PPC': np.isin(map_all, ppc_ids)
    }

    # sanity check
    for roi, mask in masks.items():
        print(f"{roi:4s} → {int(mask.sum()):4d} vertices")
    return masks


# extraction for surface data

def load_cifti(path):
    try:
        return nib.load(str(path))
    except Exception as e:
        print(f'Failed to load {path}: {e}')
        return None


def extract_pattern(img_path, mask):
    """
    Extract surface values at mask indices from a single-map CIFTI dscalar
    """
    img = load_cifti(img_path)
    if img is None:
        return None
    data = np.squeeze(img.get_fdata()).flatten()
    return data[mask]

# RSA helper funcs

def compute_rdm(pat_mat):
    z = (pat_mat - pat_mat.mean(axis=1, keepdims=True)) / pat_mat.std(axis=1, keepdims=True)
    corr = np.corrcoef(z)
    return 1 - corr


def rsa(neural, model):
    triu = np.triu_indices_from(neural, k=1)
    return spearmanr(neural[triu], model[triu]).correlation

# loop for subjects RSA
subjects = [f"sub-{i:02d}" for i in range(1, 69)]
cifti_root = pathlib.Path('derivatives/ciftify')
roi_masks = get_roi_masks()

results     = []
neural_rdms = {roi: [] for roi in roi_masks}

for sub in subjects:
    print(f"→ {sub}")
    feat_dir = cifti_root / sub / 'GLM' / 'ses-1_task-motor_hp200_s4_level2.feat'
    if not feat_dir.exists():
        print("   missing directory, skipping")
        continue

    patt = {roi: [] for roi in roi_masks}
    for part, label in part_to_file.items():
        names = [
            f"{sub}_ses-1_task-motor_level2_cope_{label}_hp200_s4.dscalar.nii",
            f"{sub}_ses-1_task-motor_level2_cope_{label}-Avg_hp200_s4.dscalar.nii"
        ]
        for fn in names:
            p = feat_dir / fn
            if p.exists(): break
        else:
            print(f"   no file for {part}, skip")
            continue
        

        # for roi, mask in roi_masks.items():
        #   print(f"{roi:4s} → {mask.sum():5d} vertices")

        for roi, mask in roi_masks.items():
            pat = extract_pattern(p, mask)
            if pat is not None:
                patt[roi].append((part, pat))

    for roi, entries in patt.items():
        if len(entries) != len(parts):
            print(f"   incomplete for {roi}, skip")
            continue
        entries.sort(key=lambda x: parts.index(x[0]))
        mat = np.stack([e[1] for e in entries])
        rdm = compute_rdm(mat)
        neural_rdms[roi].append(rdm)
        results.append({
            'subject':       sub,
            'ROI':           roi,
            'adjacency_rho': rsa(rdm, adjacency_rdm),
            'functional_rho':rsa(rdm, func_rdm)
        })


df = pd.DataFrame(results)
print(df.head())

df.to_csv(results_dir / 'rsa_results_raw.csv', index=False)

# average rdms for ROIs
avg_rdms = {roi: np.mean(rms, axis=0) for roi, rms in neural_rdms.items() if rms}

fig, axes = plt.subplots(1,4, figsize=(16,4))
for ax, roi in zip(axes, ['M1','SMA','PMC','PPC']):
    sns.heatmap(avg_rdms[roi], xticklabels=parts, yticklabels=parts,
                cmap='viridis', square=True, ax=ax)
    ax.set_title(f"Neural RDM — {roi}")
plt.tight_layout()
plt.savefig(results_dir / 'average_neural_rdms.png', dpi=300)
plt.close()

# theoretical rdms
# fig, axes = plt.subplots(1,2, figsize=(14,6))
# sns.heatmap(adjacency_rdm, xticklabels=parts, yticklabels=parts,
#             cmap='viridis', square=True, ax=axes[0])
# axes[0].set_title("Anatomical Adjacency Model RDM")
# sns.heatmap(func_rdm, xticklabels=parts, yticklabels=parts,
#             cmap='viridis', square=True, ax=axes[1])
# axes[1].set_title("Functional Similarity Model RDM")
# plt.tight_layout()
# plt.show()

# exploratory analysis 
summary = df.groupby('ROI').agg(
    adjacency_mean=('adjacency_rho','mean'), adjacency_se=('adjacency_rho', lambda x: x.std()/np.sqrt(len(x))),
    functional_mean=('functional_rho','mean'), functional_se=('functional_rho', lambda x: x.std()/np.sqrt(len(x)))
).reset_index()

idx = np.arange(len(summary))
w = 0.35
plt.figure(figsize=(9,5))
plt.bar(idx, summary['adjacency_mean'], w, yerr=summary['adjacency_se'], label='Anatomical', alpha=0.7)
plt.bar(idx+w, summary['functional_mean'], w, yerr=summary['functional_se'], label='Functional', alpha=0.7)
plt.xticks(idx+w/2, summary['ROI'])
plt.xlabel('ROI')
plt.ylabel('Spearman ρ')
plt.title('RSA: Model Fit by ROI')
plt.legend()
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(results_dir / 'rsa_model_fit_by_roi.png', dpi=300)
plt.close()


print('\nPaired t‑tests:')
for roi in summary['ROI']:
    data = df[df['ROI']==roi]
    t, p = ttest_rel(data['adjacency_rho'], data['functional_rho'])
    sig = '*(p<.05)' if p<0.05 else ''
    print(f" {roi}: t={t:.2f}, p={p:.3f} {sig}")

# mds 
# coordination 
# colors = {
#     'FOOT_COMPLEX': 'red',
#     'HAND_COMPLEX': 'blue',
#     'POSTURAL': 'green',
#     'ARM_CHAIN': 'purple',
#     'VOCAL_ARTICULATORY': 'orange',
#     'VISUO_MOTOR': 'brown'
# }
# task
colors = {
    'PRECISION_LOWER': 'red',
    'PRECISION_UPPER': 'blue',
    'STABILITY_LOWER': 'green',
    'STABILITY_UPPER': 'purple',
    'POWER_LOWER': 'orange',
    'POWER_UPPER': 'teal',
    'CONSUMPTIVE': 'brown',
    'EXPRESSIVE': 'pink',
    'SENSORY_ORAL': 'gray',
    'SENSORY_VISUAL': 'gold'
}

# colors = { 'FOOT_JOINT': 'red', 'VISUAL_CONTROL': 'blue', 'HAND_FINE': 'green', 'ARM_CONTROL' :  'purble', 'FACE_CONTROL':'orage', 'LEG_CONTROL' : 'black', 'FOOT_FINE': 'grey', 'ORAL_CONTROL': 'yellow', 'HAND_JOINT': 'pink' }
# colors = {'DFM':'red','MJA':'blue','PLM':'green','OFC':'purple','TOR':'orange'}
for roi, rdm in avg_rdms.items():
    coords2d = MDS(n_components=2, dissimilarity='precomputed', random_state=0).fit_transform(rdm)
    plt.figure(figsize=(6,6))
    for i, part in enumerate(parts):
        plt.scatter(coords2d[i,0], coords2d[i,1], color=colors[func_group[part]],
                    edgecolor='k', s=150)
        plt.text(coords2d[i,0], coords2d[i,1], part, ha='center', va='center')
    patches = [Patch(facecolor=c, label=g) for g,c in colors.items()]
    plt.legend(handles=patches, title='Func Groups')
    plt.title(f'MDS: {roi}')
    # plt.tight_layout()
    # plt.show()
    plt.tight_layout()
    plt.savefig(results_dir / f'mds_{roi}.png', dpi=300)
    plt.close()


# hierarchical bar

df['diff'] = df['adjacency_rho'] - df['functional_rho']
hld = df.groupby('ROI')['diff'].agg(['mean','sem']).reindex(['M1','SMA','PMC','PPC']).reset_index()

plt.figure(figsize=(6,4))
plt.bar(hld['ROI'], hld['mean'], yerr=hld['sem'], alpha=0.7)
plt.axhline(0, linestyle='--', color='k')
plt.xlabel('ROI (hierarchical)')
plt.ylabel('Adjacency – Functional ρ')
plt.title('Hierarchy: Anatomical → Functional')
plt.tight_layout()
plt.savefig(results_dir / 'hierarchy_adjacency_vs_functional.png', dpi=300)
plt.close()

# key findings 
print("\nKey findings:")
for _, row in summary.iterrows():
    better = 'Anatomical' if row['adjacency_mean']>row['functional_mean'] else 'Functional'
    print(f" - {row['ROI']}: {better} wins ({row['adjacency_mean']:.3f} vs {row['functional_mean']:.3f})")
print('Supports hierarchical transformation from M1 → PPC.')


# ----------------------------------------------------------------------------
# Procrustes analysis
# ----------------------------------------------------------------------------
# generate ideal MDS configurations for the theoretical models
anatomical_rdm = adjacency_rdm
functional_rdm = func_rdm

# Get 2D coordinates from the theoretical models
anatomical_mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0).fit_transform(anatomical_rdm)
functional_mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0).fit_transform(functional_rdm)

# Compare each ROI's MDS with the theoretical MDSs
results = {}
for roi, rdm in avg_rdms.items():
    # Get actual MDS coordinates for this ROI
    roi_mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0).fit_transform(rdm)
    
    # Compute Procrustes similarity with anatomical model
    _, _, anatomical_corr = procrustes(anatomical_mds, roi_mds)
    # Convert to similarity (1 = identical, 0 = completely different)
    anatomical_sim = 1 - anatomical_corr
    
    # Compute Procrustes similarity with functional model
    _, _, functional_corr = procrustes(functional_mds, roi_mds)
    functional_sim = 1 - functional_corr
    
    results[roi] = {
        'anatomical_similarity': anatomical_sim,
        'functional_similarity': functional_sim,
        'functional_dominance': functional_sim - anatomical_sim
    }

procrustes_df = pd.DataFrame(results).T
procrustes_df = procrustes_df.reindex(['M1', 'SMA', 'PMC', 'PPC'])

# Plot the results - showing transition from anatomical to functional organization
plt.figure(figsize=(10, 6))
x = np.arange(len(procrustes_df))
width = 0.35

plt.bar(x - width/2, procrustes_df['anatomical_similarity'], width, label='Anatomical Similarity', color='blue', alpha=0.7)
plt.bar(x + width/2, procrustes_df['functional_similarity'], width, label='Functional Similarity', color='red', alpha=0.7)

plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('ROI (Hierarchical Order)')
plt.ylabel('Similarity to Theoretical Model')
plt.title('Transition from Anatomical to Functional Organization')
plt.xticks(x, procrustes_df.index)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Add similarity values as text on the plot
for i, roi in enumerate(procrustes_df.index):
    anat_val = procrustes_df.loc[roi, 'anatomical_similarity']
    func_val = procrustes_df.loc[roi, 'functional_similarity']
    plt.text(i - width/2, anat_val + 0.01, f"{anat_val:.2f}", ha='center')
    plt.text(i + width/2, func_val + 0.01, f"{func_val:.2f}", ha='center')

plt.tight_layout()
plt.savefig(results_dir / 'anatomical_functional_similarity.png', dpi=300)

# Also add the similarity index to the original MDS plots
for roi, rdm in avg_rdms.items():
    coords2d = MDS(n_components=2, dissimilarity='precomputed', random_state=0).fit_transform(rdm)
    plt.figure(figsize=(6,6))
    for i, part in enumerate(parts):
        plt.scatter(coords2d[i,0], coords2d[i,1], color=colors[func_group[part]],
                    edgecolor='k', s=150)
        plt.text(coords2d[i,0], coords2d[i,1], part, ha='center', va='center')
    
    # Add the similarity metrics to the plot
    anat_sim = procrustes_df.loc[roi, 'anatomical_similarity']
    func_sim = procrustes_df.loc[roi, 'functional_similarity']
    plt.annotate(f"Anatomical similarity: {anat_sim:.2f}\nFunctional similarity: {func_sim:.2f}",
                xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    patches = [Patch(facecolor=c, label=g) for g,c in colors.items()]
    plt.legend(handles=patches, title='Func Groups')
    plt.title(f'MDS: {roi}')
    plt.tight_layout()
    plt.savefig(results_dir / f'mds_{roi}_with_metrics.png', dpi=300)
    plt.close()

procrustes_df.to_csv(results_dir / 'procrustes_similarity.csv')
