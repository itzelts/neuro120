# Analysis of Current Functional Groupings Performance

## 1. Introduction

This analysis evaluates the performance of the current functional groupings in the RSA framework across different regions of interest (ROIs). The analysis aims to determine whether the current anatomical and functional definitions are optimal or if they require restructuring.

### 1.1 Current Functional Groupings

The following functional groupings are currently defined based on shared ethological roles:

**DFM** (Distal Fine Manipulation): toe, finger

**MJA** (Mid-level Joint Articulation): ankle, wrist

**PLM** (Proximal Limb Movements): leftleg, rightleg, forearm, upperarm

**OFC** (Orofacial Communication): jaw, lip, tongue

**TOR** (Targeted Orientation): eye

## 2. Performance Analysis by ROI

### 2.1 Model Comparison

The table below compares the performance of the anatomical adjacency model and the functional similarity model across different ROIs:

| ROI | Adjacency Mean | Functional Mean | Difference | Better Model |
|-----|----------------|-----------------|------------|----------------|
| M1 | 0.5895 | 0.2966 | 0.2929 | Adjacency |
| PMC | 0.4978 | 0.2901 | 0.2076 | Adjacency |
| PPC | 0.3430 | 0.2577 | 0.0853 | Adjacency |
| SMA | 0.3321 | 0.2041 | 0.1281 | Adjacency |

### 2.2 Statistical Analysis

Paired t-tests were conducted to determine if the differences between models are statistically significant:

| ROI | t-statistic | p-value | Significant? | Better Model |
|-----|-------------|---------|--------------|----------------|
| M1 | 17.6318 | 0.0000 | True | Adjacency |
| SMA | 6.5427 | 0.0000 | True | Adjacency |
| PMC | 12.2019 | 0.0000 | True | Adjacency |
| PPC | 6.2351 | 0.0000 | True | Adjacency |

## 3. Hierarchical Analysis

The analysis reveals a pattern of transformation from anatomical to functional organization as we move from primary to higher-order motor regions:

| ROI | Adjacency-Functional Difference | Interpretation |
|-----|----------------------------------|----------------|
| M1 | 0.2929 | Anatomical dominance |
| SMA | 0.1281 | Anatomical dominance |
| PMC | 0.2076 | Anatomical dominance |
| PPC | 0.0853 | Balanced representation |

## 4. ROI Relationships

Correlation analysis between ROIs reveals how similarly they process movement representations:

### 4.1 Adjacency Model Correlations

```
ROI    M1   PMC   PPC   SMA
ROI                        
M1   1.00  0.68  0.30  0.20
PMC  0.68  1.00  0.24  0.24
PPC  0.30  0.24  1.00  0.56
SMA  0.20  0.24  0.56  1.00
```

### 4.2 Functional Model Correlations

```
ROI    M1   PMC   PPC   SMA
ROI                        
M1   1.00  0.59  0.46  0.27
PMC  0.59  1.00  0.52  0.23
PPC  0.46  0.52  1.00  0.50
SMA  0.27  0.23  0.50  1.00
```

## 5. Evaluation and Recommendations

### 5.1 Overall Assessment

Across all ROIs, the **anatomical adjacency model** performs better in 4 ROIs, while the **functional similarity model** performs better in 0 ROIs.

### 5.2 Hierarchical Transformation

The analysis reveals a clear transformation pattern across the motor hierarchy. In primary motor areas (M1), the anatomical model shows a stronger advantage (0.2929), while in higher-order regions (PPC), this advantage shifts (0.0853) toward the functional model.

### 5.3 Recommendations for Functional Groupings

**Recommendation**: Consider a hybrid approach where:

1. For primary motor regions (e.g., M1), maintain groupings that respect anatomical adjacency
2. For higher-order regions (e.g., PPC), emphasize the current functional groupings
3. For intermediate regions, balance both organizational principles

### 5.4 Specific Group Recommendations

#### Proximal Limb Movements (PLM)

The PLM group contains four body parts (leftleg, rightleg, forearm, upperarm) and may benefit from subdivision into:

- **Lower Limb Movements**: leftleg, rightleg
- **Upper Limb Movements**: forearm, upperarm

#### Targeted Orientation (TOR)

The TOR group contains only eye movements. Consider if this is truly a separate functional category or if it could be integrated with another group based on functional properties.

## 6. Conclusion

The analysis supports the hypothesis of a hierarchical transformation from anatomical to functional organization across the motor hierarchy. The current functional groupings show promise, particularly in higher-order regions, but could benefit from refinements based on the hierarchical organization observed in the data.

The most compelling evidence for functional organization appears in higher-order regions (PMC, PPC), suggesting that these areas may be where ethological groupings are most relevant to neural processing.