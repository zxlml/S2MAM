Summary of the complete experiment reproduction (paper protocol: partial
labeling + massive redundant dims + extreme noisy dims N(100,100), seed 0).

Columns:
- selected_features: indices of the features selected by S2MAM
- precision/recall: feature-selection quality vs. the true informative dims
- metric_name: test_mse (regression, lower is better) / test_acc (classification)
- s2mam_metric: S2MAM performance with the selected features
- full_feature_baseline / oracle_baseline: baselines (all features / true dims)
- s2mam_equals_oracle: whether S2MAM matches the oracle performance
- engine: upper-level solver (implicit = exact hypergradient; reinforce = PBCS-style)

Reproduce with:
  python main.py --task regression --dataset additive
  python main.py --task regression --dataset friedman
  python main.py --task classification --dataset additive
  python main.py --task classification --dataset moon --engine reinforce --max_outer_iter 40
