# Skeleton-normalised FK features vs. a tiny MLP for a legs-spread pose metric

## Motivation
"Pose metrics: a new paradigm for character motion edition" (2023) proposes a modular pipeline of tiny neural networks, one per pose metric, each trained on a single standardised humanoid skeleton. Such per-metric networks working on raw joint rotations may not transfer when limb proportions change. This experiment checks whether skeleton-normalised forward-kinematics features help.

## Question
Can a metric regressor on skeleton-normalised forward-kinematics features beat a tiny MLP on raw joint rotations at predicting a legs-spread pose metric when the skeleton proportions change?

## Dataset
Simulated, not the paper's data. A 16-joint humanoid (hips, knees, ankles, spine, arms) is posed with forward kinematics in numpy: 20,000 random correlated poses. Legs spread = foot-to-foot distance / leg length; knee bend is a second, overlapping metric. Training uses the standard skeleton only. The held-out test poses are evaluated on the standard skeleton and on skeletons with changed limb proportions (main test).

## Baseline vs improved
- **Baseline:** tiny scikit-learn `MLPRegressor` on raw joint rotations (stand-in for a per-metric tiny network), tuned with cross-validation.
- **Improved:** LightGBM on rotations plus leg-length-normalised FK positions and angles between limb vectors, tuned with cross-validation using the same search budget.

Both use the same split, training data and evaluation; seeds 42, 43, 44. Secondary metrics: RMSE on the standard skeleton and the correlation of the legs-spread error with knee bend.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> L Victor, A Meyer, S Bouakaz (2023). *Pose metrics: a new paradigm for character motion edition*. arXiv preprint arXiv:2301.06514, 2023 - arxiv.org. https://arxiv.org/abs/2301.06514

Paper author: Prof. Dr. Alexandre Meyer, Université Claude Bernard Lyon 1 (alexandre.meyer@univ-lyon1.fr)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
