"""Legs-spread regression: tiny MLP on raw joint rotations vs tuned LightGBM on skeleton-normalised FK features."""
import json, os, sys, warnings
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED, N, SEEDS, N_ITER = 42, 20000, [42, 43, 44], 8
rng = np.random.default_rng(SEED)

# ---- humanoid: 16 joints (pelvis, 3 spine/head, 2x3 arm, 2x3 leg), y-up, rest-pose offsets in parent frame
PARENT = [-1, 0, 1, 2, 2, 4, 5, 2, 7, 8, 0, 10, 11, 0, 13, 14]
J = len(PARENT)
BASE = np.array([[0, 0, 0], [0, .2, 0], [0, .25, 0], [0, .3, 0], [.2, .15, 0], [0, -.3, 0], [0, -.25, 0],
                 [-.2, .15, 0], [0, -.3, 0], [0, -.25, 0], [.1, -.05, 0], [0, -.45, 0], [0, -.42, 0],
                 [-.1, -.05, 0], [0, -.45, 0], [0, -.42, 0]])
LK, LA, RK, RA, LH, RH = 11, 12, 14, 15, 10, 13  # knee / ankle / hip joint ids

def make_skel(n, changed):
    """Standard skeleton, or per-sample changed limb proportions (spine, arms, thigh, shin, hip width, global)."""
    off = np.tile(BASE.astype(float), (n, 1, 1))
    if changed:
        s = rng.uniform([.8, .7, .7, .7, .6, .8], [1.2, 1.3, 1.4, 1.4, 1.6, 1.2], (n, 6))
        off[:, [1, 2, 3]] *= s[:, 0, None, None]
        off[:, 4:10] *= s[:, 1, None, None]
        off[:, [LK, RK]] *= s[:, 2, None, None]
        off[:, [LA, RA]] *= s[:, 3, None, None]
        off[:, [LH, RH], 0] *= s[:, 4, None]
        off *= s[:, 5, None, None]
    return off

def rotmat(a):
    """Euler (Rz Ry Rx) rotation matrices for angles (N,J,3) -> (N,J,3,3)."""
    cx, cy, cz = np.cos(a[..., 0]), np.cos(a[..., 1]), np.cos(a[..., 2])
    sx, sy, sz = np.sin(a[..., 0]), np.sin(a[..., 1]), np.sin(a[..., 2])
    rows = [[cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy, sx * cy, cx * cy]]
    return np.stack([np.stack(r, -1) for r in rows], -2)

def fk(ang, off):
    """Forward kinematics -> world joint positions (N,J,3)."""
    R = rotmat(ang)
    Rw, pos = np.zeros_like(R), np.zeros(ang.shape)
    Rw[:, 0] = R[:, 0]
    for j in range(1, J):
        p = PARENT[j]
        pos[:, j] = pos[:, p] + np.einsum("nab,nb->na", Rw[:, p], off[:, j])
        Rw[:, j] = Rw[:, p] @ R[:, j]
    return pos

def angle(a, b):
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    return np.arccos(np.clip(c, -1, 1))

def limbs(pos):
    v = lambda a, b: pos[:, b] - pos[:, a]
    return dict(thL=v(LH, LK), thR=v(RH, RK), shL=v(LK, LA), shR=v(RK, RA), uaL=v(4, 5), uaR=v(7, 8),
                faL=v(5, 6), faR=v(8, 9), spine=v(0, 2), down=np.tile([0., -1., 0.], (len(pos), 1)))

def metrics(ang, off):
    """Targets: legs spread = foot-to-foot distance / leg length; knee bend = mean knee flexion angle."""
    pos, l = fk(ang, off), None
    l = np.linalg.norm(off[:, LK], axis=-1) + np.linalg.norm(off[:, LA], axis=-1)
    spread = np.linalg.norm(pos[:, LA] - pos[:, RA], axis=-1) / l
    b = limbs(pos)
    return spread, (angle(b["thL"], b["shL"]) + angle(b["thR"], b["shR"])) / 2

def feats_raw(ang, off):
    return ang[:, 1:].reshape(len(ang), -1)  # raw joint rotations only (no skeleton info)

def feats_fk(ang, off):
    """Raw rotations + FK joint positions normalised by leg length + angles between limb vectors."""
    pos = fk(ang, off)
    leg = (np.linalg.norm(off[:, LK], axis=-1) + np.linalg.norm(off[:, LA], axis=-1))[:, None, None]
    P = ((pos[:, 1:] - pos[:, :1]) / leg).reshape(len(ang), -1)
    b = limbs(pos)
    pairs = [("thL", "thR"), ("shL", "shR"), ("thL", "shL"), ("thR", "shR"), ("thL", "down"), ("thR", "down"),
             ("shL", "down"), ("shR", "down"), ("uaL", "uaR"), ("uaL", "spine"), ("uaR", "spine"),
             ("faL", "uaL"), ("faR", "uaR")]
    A = np.stack([angle(b[x], b[y]) for x, y in pairs], 1)
    return np.hstack([feats_raw(ang, off), P, A])

# ---- correlated random poses (latent factors mixed into all DOFs, then joint-limit clipping)
MEAN, STD = np.zeros((J, 3)), np.full((J, 3), .15)
STD[0] = 0                                                  # pelvis fixed
STD[[4, 7]], MEAN[[5, 8], 0], STD[[5, 8]] = [.5, .3, .5], .6, [.5, .1, .1]
STD[[LH, RH]], MEAN[[LH, RH], 0] = [.5, .2, .45], .1
MEAN[[LK, RK], 0], STD[[LK, RK]] = .6, [.5, .05, .05]
LO, HI = MEAN - 2.5 * STD, MEAN + 2.5 * STD
LO[[LK, RK], 0] = 0

z = rng.normal(size=(N, 6))
raw = z @ rng.normal(size=(6, J * 3)) + .7 * rng.normal(size=(N, J * 3))
raw /= raw.std(0) + 1e-9
ANG = np.clip(MEAN + STD * raw.reshape(N, J, 3), LO, HI)

# ---- single split: train on standard skeleton; test poses on standard AND changed skeletons
itr, ite = train_test_split(np.arange(N), test_size=.25, random_state=SEED)
off_tr, off_std, off_chg = make_skel(len(itr), False), make_skel(len(ite), False), make_skel(len(ite), True)
y_tr, _ = metrics(ANG[itr], off_tr)
y_std, _ = metrics(ANG[ite], off_std)
y_chg, knee_chg = metrics(ANG[ite], off_chg)
print(f"poses train={len(itr)} test={len(ite)}; spread std={y_tr.std():.3f}, corr(spread,knee)="
      f"{np.corrcoef(*metrics(ANG[itr], off_tr))[0, 1]:.2f}")

# ---- models: identical pipeline/CV/budget; only the model + features differ
def mlp(): return MLPRegressor(max_iter=300, early_stopping=True, random_state=SEED)
def gbm(): return lgb.LGBMRegressor(n_jobs=1, verbose=-1, random_state=SEED, subsample_freq=1)
SPACES = {
    "baseline": (mlp, feats_raw, {"m__hidden_layer_sizes": [(8,), (16,), (32,)], "m__alpha": [1e-4, 1e-3, 1e-2],
                                  "m__learning_rate_init": [1e-3, 3e-3, 1e-2]}),
    "improved": (gbm, feats_fk, {"m__n_estimators": [200, 400], "m__learning_rate": [.03, .06, .1],
                                 "m__num_leaves": [15, 31, 63], "m__min_child_samples": [10, 20, 40],
                                 "m__subsample": [.7, .9], "m__colsample_bytree": [.5, .8]}),
}
res, M = {}, {}
for name, (make, ff, space) in SPACES.items():
    Xtr, Xstd, Xchg = ff(ANG[itr], off_tr), ff(ANG[ite], off_std), ff(ANG[ite], off_chg)
    search = RandomizedSearchCV(Pipeline([("s", StandardScaler()), ("m", make())]), space, n_iter=N_ITER,
                                cv=KFold(3, shuffle=True, random_state=SEED), random_state=SEED, n_jobs=-1,
                                scoring="neg_root_mean_squared_error").fit(Xtr, y_tr)  # CV on train only
    print(f"{name}: best CV RMSE={-search.best_score_:.4f} params={search.best_params_}")
    r = {"chg": [], "std": [], "corr": []}
    for s in SEEDS:  # refit with seeds 42-44; test set used once, here
        m = clone(search.best_estimator_).set_params(m__random_state=s).fit(Xtr, y_tr)
        pc, ps = m.predict(Xchg), m.predict(Xstd)
        r["chg"].append(np.sqrt(np.mean((pc - y_chg) ** 2)))
        r["std"].append(np.sqrt(np.mean((ps - y_std) ** 2)))
        r["corr"].append(abs(np.corrcoef(pc - y_chg, knee_chg)[0, 1]))
    res[name] = r
    M.update({f"{name}_score": float(np.mean(r["chg"])), f"{name}_score_std": float(np.std(r["chg"])),
              f"{name}_rmse_standard_skeleton": float(np.mean(r["std"])),
              f"{name}_rmse_standard_skeleton_std": float(np.std(r["std"])),
              f"{name}_abs_corr_error_vs_knee_bend": float(np.mean(r["corr"]))})
    print(f"{name}: RMSE changed={M[name + '_score']:.4f} standard={M[name + '_rmse_standard_skeleton']:.4f}")
M["improvement"] = M["baseline_score"] - M["improved_score"]
if not all(np.isfinite(v) for v in M.values()):
    sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(5, 3.5))
for i, (k, lab) in enumerate([("score", "changed skeletons"), ("rmse_standard_skeleton", "standard skeleton")]):
    ax.bar([i - .2, i + .2], [M[f"baseline_{k}"], M[f"improved_{k}"]], .4, color=["tab:gray", "tab:green"],
           label=["MLP raw rotations", "LightGBM FK-normalised"] if i == 0 else None)
ax.set_xticks([0, 1]); ax.set_xticklabels(["changed skeletons", "standard skeleton"])
ax.set_ylabel("RMSE legs-spread"); ax.legend(); fig.tight_layout(); fig.savefig("figures/rmse.png", dpi=120)

out = {
    "question": "Can a metric regressor on skeleton-normalised forward-kinematics features beat a tiny MLP on raw joint rotations at predicting a legs-spread pose metric when the skeleton proportions change?",
    "method": "Simulated 16-joint humanoid with FK; 20,000 correlated random poses; legs spread = foot-to-foot distance / leg length, knee bend as overlapping metric. Train on standard skeleton (75% of poses); test on held-out poses with the standard skeleton and with randomly changed limb proportions. Both models tuned with 3-fold CV on train only (same random-search budget), refit with seeds 42-44.",
    "dataset": {"name": "Simulated humanoid poses (numpy FK)", "source_url": "generated locally with numpy (no download)",
                "n_rows": int(N), "n_train": int(len(itr)), "n_test": int(len(ite)), "same_as_paper": False,
                "baseline_is_papers_method": False},
    "metric_name": "RMSE of legs-spread on changed-skeleton test poses", "higher_is_better": False,
    "baseline": "Tiny scikit-learn MLPRegressor (standardised raw joint rotations only), CV-tuned over a small grid; stand-in for the paper's per-metric tiny network.",
    "improved": "LightGBM (CV-tuned, same number of random-search iterations) on raw rotations + leg-length-normalised FK joint positions + angles between limb vectors computed with each sample's skeleton.",
    "metrics": M,
    "notes": "Simulation only: results depend on the assumed pose distribution and skeleton-variation ranges. The improved model is given the skeleton (bone lengths) at inference, the baseline is not, as in the planned comparison; the gain therefore mixes model class and extra input information. Target is a deterministic function of FK, so features closely related to it are available to the improved model. Error/knee-bend correlation is the absolute Pearson correlation on changed-skeleton test poses (lower = less interference). Paper reports no numeric result to compare with.",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("wrote results.json")
