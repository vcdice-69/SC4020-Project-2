"""Shared metrics and Task 2 models, reused by Task 2 and Task 3."""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

SEED = 42
MIN_CANDIDATES = 6


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def concordance_index(y_true, y_pred):
    """CI = (tau * sqrt(B / A) + 1) / 2, A and B the pair counts that differ
    in true and in predicted score."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) < 2:
        return np.nan
    n_pairs = len(y_true) * (len(y_true) - 1) / 2

    def comparable(values):
        counts = np.unique(values, return_counts=True)[1]
        return n_pairs - np.sum(counts * (counts - 1) / 2)

    a, b = comparable(y_true), comparable(y_pred)
    if a == 0 or b == 0:
        return np.nan
    tau = kendalltau(y_true, y_pred, variant="b").statistic
    return float((tau * np.sqrt(b / a) + 1) / 2)


def ranking_metrics(groups, y_true, y_pred, min_candidates=MIN_CANDIDATES):
    """Rank of the true best candidate within each group. Pass ligand indices
    to rank a ligand's targets, or target indices to rank a target's ligands."""
    table = pd.DataFrame({"group": groups, "true": y_true, "pred": y_pred})
    ranks, sizes = [], []
    for _, block in table.groupby("group"):
        if len(block) < min_candidates:
            continue
        best = block["true"].to_numpy().argmax()
        order = np.argsort(-block["pred"].to_numpy(), kind="stable")
        ranks.append(int(np.where(order == best)[0][0]) + 1)
        sizes.append(len(block))
    if not ranks:
        return {"Hit@1": np.nan, "Hit@5": np.nan, "MRR": np.nan,
                "Hit@1 random": np.nan, "Hit@5 random": np.nan, "groups": 0}
    ranks, sizes = np.array(ranks), np.array(sizes)
    return {
        "Hit@1": float((ranks <= 1).mean()),
        "Hit@5": float((ranks <= 5).mean()),
        "MRR": float((1 / ranks).mean()),
        "Hit@1 random": float(np.mean(1 / sizes)),
        "Hit@5 random": float(np.mean(np.minimum(5, sizes) / sizes)),
        "groups": len(ranks),
    }


class Frame:
    """Index maps plus the training score matrix for one dataset and split."""

    def __init__(self, df):
        self.ligands = np.array(sorted(set(df["ligand_id"])))
        self.targets = np.array(sorted(set(df["target_id"])))
        self.lig_pos = {v: i for i, v in enumerate(self.ligands)}
        self.tgt_pos = {v: i for i, v in enumerate(self.targets)}
        self.parts = {}
        for part in ["train", "val", "test"]:
            sub = df[df["split"] == part]
            self.parts[part] = (sub["ligand_id"].map(self.lig_pos).to_numpy(),
                                sub["target_id"].map(self.tgt_pos).to_numpy(),
                                sub["score"].to_numpy(float))
        rows, cols, vals = self.parts["train"]
        self.R = np.full((len(self.ligands), len(self.targets)), np.nan)
        self.R[rows, cols] = vals

    def xy(self, part):
        return self.parts[part]


class ConstantModel:
    """global / ligand / target mean."""

    def __init__(self, axis="global"):
        self.axis = axis

    def fit(self, frame):
        self.mu = float(np.nanmean(frame.R))
        observed = np.isfinite(frame.R)
        values = np.where(observed, frame.R, 0.0)
        n_row, n_col = observed.sum(axis=1), observed.sum(axis=0)
        self.row_mean = np.where(n_row > 0, values.sum(axis=1) / np.maximum(n_row, 1), self.mu)
        self.col_mean = np.where(n_col > 0, values.sum(axis=0) / np.maximum(n_col, 1), self.mu)
        return self

    def predict(self, rows, cols):
        if self.axis == "global":
            return np.full(len(rows), self.mu)
        return self.row_mean[rows] if self.axis == "ligand" else self.col_mean[cols]


class BiasModel:
    """mu + b_ligand + b_target by alternating shrunken means."""

    def __init__(self, reg_ligand=5.0, reg_target=5.0, iterations=20):
        self.reg_ligand, self.reg_target, self.iterations = reg_ligand, reg_target, iterations

    def fit(self, frame):
        R = frame.R
        observed = np.isfinite(R)
        self.mu = float(np.nanmean(R))
        self.b_lig = np.zeros(R.shape[0])
        self.b_tgt = np.zeros(R.shape[1])
        n_lig, n_tgt = observed.sum(axis=1), observed.sum(axis=0)
        residual = np.where(observed, R - self.mu, 0.0)
        for _ in range(self.iterations):
            self.b_lig = (residual - observed * self.b_tgt).sum(axis=1) / (n_lig + self.reg_ligand)
            self.b_tgt = ((residual - observed * self.b_lig[:, None]).sum(axis=0)
                          / (n_tgt + self.reg_target))
        return self

    def baseline_matrix(self):
        return self.mu + self.b_lig[:, None] + self.b_tgt[None, :]

    def predict(self, rows, cols):
        return self.mu + self.b_lig[rows] + self.b_tgt[cols]


class NeighbourhoodCF:
    """Task 2 recommender: KNN over the bias baseline."""

    def __init__(self, k=20, shrinkage=25.0, axis="ligand", reg=5.0):
        self.k, self.shrinkage, self.axis, self.reg = k, shrinkage, axis, reg

    def fit(self, frame):
        self.bias = BiasModel(self.reg, self.reg).fit(frame)
        self.baseline = self.bias.baseline_matrix()
        observed = np.isfinite(frame.R)
        self.E = np.where(observed, frame.R - self.baseline, 0.0)
        self.O = observed.astype(float)
        self.similarity = self._similarity()
        return self

    def _oriented(self):
        return (self.E, self.O) if self.axis == "ligand" else (self.E.T, self.O.T)

    def _similarity(self):
        E, O = self._oriented()
        dot = E @ E.T
        norm = np.sqrt(np.einsum("ij,ij->i", E, E))
        denominator = np.outer(norm, norm)
        sim = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
        common = O @ O.T
        total = common + self.shrinkage
        sim = sim * np.divide(common, total, out=np.zeros_like(common), where=total > 0)
        np.fill_diagonal(sim, 0.0)
        return sim

    def predict(self, rows, cols):
        E, O = self._oriented()
        queries, others = (rows, cols) if self.axis == "ligand" else (cols, rows)
        predictions = self.baseline[rows, cols].astype(float)
        self.fallbacks = np.zeros(len(rows), dtype=bool)
        for other in np.unique(others):
            picked = np.flatnonzero(others == other)
            pool = np.flatnonzero(O[:, other] > 0)
            if pool.size == 0:
                self.fallbacks[picked] = True
                continue
            sims = self.similarity[np.ix_(queries[picked], pool)].copy()
            sims[sims <= 0] = 0.0
            if pool.size > self.k:
                cut = np.argpartition(-sims, self.k, axis=1)[:, self.k:]
                np.put_along_axis(sims, cut, 0.0, axis=1)
            weight = sims.sum(axis=1)
            usable = weight > 0
            numerator = sims @ E[pool, other]
            predictions[picked[usable]] += numerator[usable] / weight[usable]
            self.fallbacks[picked[~usable]] = True
        return predictions

    def fallback_rate(self, frame, part):
        rows, cols, _ = frame.xy(part)
        self.predict(rows, cols)
        return float(self.fallbacks.mean())


def evaluate(model, frame, part="test"):
    """MSE, CI and ranking metrics under both grouping questions."""
    rows, cols, y_true = frame.xy(part)
    y_pred = model.predict(rows, cols)
    out = {"MSE": mse(y_true, y_pred), "CI": concordance_index(y_true, y_pred)}
    for label, groups in [("per ligand", rows), ("per target", cols)]:
        for key, value in ranking_metrics(groups, y_true, y_pred).items():
            out[f"{key} ({label})"] = value
    return out
