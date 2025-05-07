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

# ----------------------------------------------------------------------------
# Settings and File Paths
# ----------------------------------------------------------------------------
bids_root   = pathlib.Path('.')
first_sub   = 'sub-01'
events_path = bids_root / first_sub / 'ses-1' / 'func' / f'{first_sub}_ses-1_task-motor_run-01_events.tsv'

results_dir = pathlib.Path('results')
results_dir.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
#  Design Matrix -- Exploratory 
# ----------------------------------------------------------------------------
# events = pd.read_csv(events_path, sep='\t')
# TR = 1.5
# n_scans = 280
# frame_times = np.arange(n_scans) * TR
# design_matrix = make_first_level_design_matrix(
#     frame_times,
#     events,
#     hrf_model='spm',
#     drift_model='cosine'
# )
# plt.figure(figsize=(12,5))
# design_matrix.plot()
# plt.title('Design Matrix: Motor Task (run‑01)')
# plt.tight_layout()
# plt.show()

# ----------------------------------------------------------------------------
# Part 2: Define Body Parts & Theoretical RDMs
# ----------------------------------------------------------------------------
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

func_group = {
    'ankle':'FOOT_JOINT',
    'eye':'VISUAL_CONTROL',
    'finger':'HAND_FINE',
    'forearm':'ARM_CONTROL',
    'jaw':'FACE_CONTROL',
    'leftleg':'LEG_CONTROL',
    'lip':'FACE_CONTROL',
    'rightleg':'LEG_CONTROL',
    'toe':'FOOT_FINE',
    'tongue':'ORAL_CONTROL',
    'upperarm':'ARM_CONTROL',
    'wrist':'HAND_JOINT',
}
func_rdm = np.zeros((len(parts), len(parts)))
for i, p1 in enumerate(parts):
    for j, p2 in enumerate(parts):
        func_rdm[i, j] = float(func_group[p1] != func_group[p2])

# ----------------------------------------------------------------------------
# ROI Masks via Glasser MMP Surface Atlas
# ----------------------------------------------------------------------------
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
    Extracts surface values at mask indices from a single-map CIFTI dscalar.
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

# ----------------------------------------------------------------------------
# Loop fo subjects and RSA
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Average Neural RDMs & Theoretical RDMs 
# ----------------------------------------------------------------------------
avg_rdms = {roi: np.mean(rms, axis=0) for roi, rms in neural_rdms.items() if rms}

fig, axes = plt.subplots(1,4, figsize=(16,4))
for ax, roi in zip(axes, ['M1','SMA','PMC','PPC']):
    sns.heatmap(avg_rdms[roi], xticklabels=parts, yticklabels=parts,
                cmap='viridis', square=True, ax=ax)
    ax.set_title(f"Neural RDM — {roi}")
# plt.tight_layout()
# plt.show()
plt.tight_layout()
plt.savefig(results_dir / 'average_neural_rdms.png', dpi=300)
plt.close()


# fig, axes = plt.subplots(1,2, figsize=(14,6))
# sns.heatmap(adjacency_rdm, xticklabels=parts, yticklabels=parts,
#             cmap='viridis', square=True, ax=axes[0])
# axes[0].set_title("Anatomical Adjacency Model RDM")
# sns.heatmap(func_rdm, xticklabels=parts, yticklabels=parts,
#             cmap='viridis', square=True, ax=axes[1])
# axes[1].set_title("Functional Similarity Model RDM")
# plt.tight_layout()
# plt.show()

# ----------------------------------------------------------------------------
# Some stats and more plots!
# ----------------------------------------------------------------------------
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
# plt.tight_layout()
# plt.show()
plt.tight_layout()
plt.savefig(results_dir / 'rsa_model_fit_by_roi.png', dpi=300)
plt.close()


print('\nPaired t‑tests:')
for roi in summary['ROI']:
    data = df[df['ROI']==roi]
    t, p = ttest_rel(data['adjacency_rho'], data['functional_rho'])
    sig = '*(p<.05)' if p<0.05 else ''
    print(f" {roi}: t={t:.2f}, p={p:.3f} {sig}")

# ----------------------------------------------------------------------------
# MDS & Hierarchical difference
# ----------------------------------------------------------------------------
colors = { 'FOOT_JOINT': 'red', 'VISUAL_CONTROL': 'blue', 'HAND_FINE': 'green', 'ARM_CONTROL' :  'purble', 'FACE_CONTROL':'orage', 'LEG_CONTROL' : 'black', 'FOOT_FINE': 'grey', 'ORAL_CONTROL': 'yellow', 'HAND_JOINT': 'pink' }
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


# Hierarchical bar

df['diff'] = df['adjacency_rho'] - df['functional_rho']
hld = df.groupby('ROI')['diff'].agg(['mean','sem']).reindex(['M1','SMA','PMC','PPC']).reset_index()

plt.figure(figsize=(6,4))
plt.bar(hld['ROI'], hld['mean'], yerr=hld['sem'], alpha=0.7)
plt.axhline(0, linestyle='--', color='k')
plt.xlabel('ROI (hierarchical)')
plt.ylabel('Adjacency – Functional ρ')
plt.title('Hierarchy: Anatomical → Functional')
# plt.tight_layout()
# plt.show()
plt.tight_layout()
plt.savefig(results_dir / 'hierarchy_adjacency_vs_functional.png', dpi=300)
plt.close()


# ----------------------------------------------------------------------------
# quick summary 
# ----------------------------------------------------------------------------
print("\nKey findings:")
for _, row in summary.iterrows():
    better = 'Anatomical' if row['adjacency_mean']>row['functional_mean'] else 'Functional'
    print(f" - {row['ROI']}: {better} wins ({row['adjacency_mean']:.3f} vs {row['functional_mean']:.3f})")
print('Supports hierarchical transformation from M1 → PPC.')
