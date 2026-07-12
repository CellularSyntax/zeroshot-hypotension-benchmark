**Table 7. Cross-dataset transfer of impending-hypotension prediction (covariate-free, M0; AUROC, all cases). Supervised baselines are shown in-domain (held-out 5-fold OOF on the training cohort) and transferred (all-train checkpoint applied to the other cohort); TiRex-2 is zero-shot on both. Cells marked 'pending' await the corresponding in-domain cross-validation run. 95% CIs are in Fig. 6. VitalDB in-domain is at 15 s cadence; all transferred and MOVER results are at the harmonised 60 s cadence.**

| Test cohort | Model (training source) | 1 min | 3 min | 5 min | 7 min | 10 min | 15 min |
|---|---|---|---|---|---|---|---|
| VitalDB | TiRex-2 (zero-shot) | 0.986 | 0.953 | 0.924 | 0.900 | 0.872 | 0.846 |
|  | TFT (in-domain) | 0.983 | 0.945 | 0.913 | 0.888 | 0.858 | 0.831 |
|  | TFT (trained MOVER) | 0.964 | 0.924 | 0.897 | 0.874 | 0.849 | 0.825 |
|  | PatchTST (in-domain) | 0.982 | 0.947 | 0.920 | 0.897 | 0.870 | 0.846 |
|  | PatchTST (trained MOVER) | 0.967 | 0.928 | 0.900 | 0.877 | 0.851 | 0.832 |
| MOVER | TiRex-2 (zero-shot) | 0.948 | 0.919 | 0.899 | 0.886 | 0.872 | 0.855 |
|  | TFT (in-domain) | 0.945 | 0.922 | 0.903 | 0.889 | 0.876 | 0.860 |
|  | TFT (trained VitalDB) | 0.940 | 0.914 | 0.895 | 0.881 | 0.866 | 0.846 |
|  | PatchTST (in-domain) | 0.945 | 0.919 | 0.901 | 0.888 | 0.877 | 0.862 |
|  | PatchTST (trained VitalDB) | 0.939 | 0.912 | 0.895 | 0.881 | 0.868 | 0.851 |
