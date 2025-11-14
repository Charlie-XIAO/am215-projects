"""
main.py

What this script does
---------------------
1) Load the public-health lab influenza CSV and aggregate weekly counts into
   two series for a chosen region/season: strain A and strain B.
2) Fit a TWO-STRAIN SEIR model (with and without cross-immunity) on the
   TRAIN window and evaluate on TEST; plots now SHADE the train window.
3) Fit a ONE-STRAIN SEIR model to TOTAL (A+B) on the SAME train/test window.
4) Save side-by-side plots and JSON reports for two-strain and one-strain fits.

How to run
----------
python main.py

Optional quick edits near the top:
- DATA_PHL: path to CSV (put the file next to this script or inside ./data/)
- USER_REGION / USER_SEASON: force a region/season, or leave as None to auto-select
- TRAIN_FRAC: fraction of weeks for training (rest is test)

Requirements
------------
numpy, pandas, matplotlib, scipy
"""

# =========================
# Imports & basic setup
# =========================
import os
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================
# Config
# =========================
# If the CSV is next to this file, this path works; else move the CSV into ./data/
DATA_PHL = "public-health-laboratory-influenza-respiratory-virus-surveillance-data-by-region-and-influenza-season.csv"
if not os.path.exists(DATA_PHL):
    DATA_PHL = os.path.join("data", "public-health-laboratory-influenza-respiratory-virus-surveillance-data-by-region-and-influenza-season.csv")

SAVE_DIR = "outputs"             # where figures and json will be written
USER_REGION = None               # e.g., "Bay Area" or None to auto-pick region with most records
USER_SEASON = None               # e.g., "2017-2018" or None to auto-pick season with most weeks
TRAIN_FRAC = 0.70                # fraction of weeks used for training
RANDOM_SEED = 123                # only used for any randomness (not strictly necessary)


# =========================
# Small utilities
# =========================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def to_datetime_safe(x: pd.Series) -> pd.Series:
    """Robust datetime conversion (coerce invalid to NaT)."""
    return pd.to_datetime(x, errors="coerce", infer_datetime_format=True)


# =========================
# Data loading & aggregation
# =========================
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find CSV at {path}. "
            "Place your file next to main.py or inside ./data/ with the same filename."
        )
    df = pd.read_csv(path)
    needed = {"season", "weekending", "region", "Influenza_Category", "Count"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    # Clean types
    df["weekending"] = to_datetime_safe(df["weekending"])
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0.0)
    df["Influenza_Category"] = df["Influenza_Category"].astype(str)
    df["region"] = df["region"].astype(str)
    df["season"] = df["season"].astype(str)
    return df


def pick_default_region(df: pd.DataFrame) -> str:
    """Pick the region with the most rows (simple heuristic)."""
    return df["region"].value_counts().idxmax()


def map_category_to_strain(cat: str) -> str | None:
    """
    Map detailed category names to 'A' or 'B'. Return None to ignore.
    """
    s = cat.strip().lower()
    if s.startswith("influenza_a"):
        return "A"
    if s.startswith("influenza b") or s.startswith("influenza_b"):
        return "B"
    # Some datasets use e.g. 'Influenza_B' or 'Influenza_BVIC', etc.
    if "influenza" in s and (" b" in s or s.endswith("_b") or "_b" in s):
        return "B"
    if "influenza" in s and (" a" in s or s.endswith("_a") or "_a" in s):
        return "A"
    return None


def aggregate_two_strains(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    """
    Return a tidy weekly table for a REGION with columns:
      season, weekending, A, B, total
    Sorted by season then weekending.
    """
    d = df[df["region"] == region_name].copy()
    if d.empty:
        raise ValueError(f"No rows found for region '{region_name}'.")

    d["strain"] = d["Influenza_Category"].map(map_category_to_strain)
    d = d[~d["strain"].isna()].copy()

    # Sum counts per (season, weekending, strain)
    g = (
        d.groupby(["season", "weekending", "strain"], as_index=False)["Count"]
        .sum()
        .rename(columns={"Count": "count"})
    )
    # Pivot to columns A, B
    piv = g.pivot_table(
        index=["season", "weekending"],
        columns="strain",
        values="count",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    piv.columns.name = None

    # Ensure columns exist
    for col in ["A", "B"]:
        if col not in piv.columns:
            piv[col] = 0.0

    piv["total"] = piv["A"] + piv["B"]
    piv = piv.sort_values(["season", "weekending"]).reset_index(drop=True)
    return piv


def choose_season_with_most_weeks(piv: pd.DataFrame) -> str:
    cnt = piv.groupby("season")["weekending"].nunique().sort_values(ascending=False)
    return cnt.index[0]


def extract_season_slice(piv: pd.DataFrame, season: str) -> pd.DataFrame:
    s = piv[piv["season"] == season].copy()
    s = s.sort_values("weekending").reset_index(drop=True)
    # Drop weeks where both A and B are missing (shouldn't happen after fill_value)
    return s


def build_week_axis(weekending_col: pd.Series) -> np.ndarray:
    """
    Map weekending dates in a season to a 0..N-1 float grid (weeks).
    """
    order = np.argsort(weekending_col.values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks


def make_train_mask(n_weeks: int, frac: float) -> np.ndarray:
    """Train on first ceil(frac * n_weeks), test on the rest."""
    n_train = max(1, int(math.ceil(frac * n_weeks)))
    mask = np.zeros(n_weeks, dtype=bool)
    mask[:n_train] = True
    return mask


# =========================
# Two-strain SEIR model
# =========================
@dataclass
class TwoStrainParams:
    beta1: float = 0.9
    beta2: float = 0.9
    amplitude: float = 0.2     # seasonal forcing amplitude
    phi_weeks: float = 2.0     # seasonal phase (weeks)
    sigma: float = 0.20        # cross-immunity reduction (0=no cross immunity)
    k: float = 1 / 2.0         # E -> I rate per week
    gamma: float = 1 / 3.0     # I -> R rate per week
    S0_frac: float = 0.95
    R10_frac: float = 0.02
    R20_frac: float = 0.02
    seed1: float = 1e-6
    seed2: float = 1e-6
    rho1: float = 50.0         # observation scale strain A
    rho2: float = 50.0         # observation scale strain B


class TwoStrainSEIR:
    """
    Two-strain SEIR with partial cross-immunity.
    Compartments:
      S, R1, R2, R12,
      E1S, I1S, E1R2, I1R2,
      E2S, I2S, E2R1, I2R1
    """

    def __init__(self, seasonal: bool = True):
        self.seasonal = seasonal

    @staticmethod
    def _beta_t(t, beta, amplitude, phi_weeks):
        # Cosine seasonality with ~52.18-week period
        return beta * (1.0 + amplitude * np.cos(2.0 * np.pi * (t - phi_weeks) / 52.18))

    def rhs(self, t, y, p: TwoStrainParams):
        (S, R1, R2, R12,
         E1S, I1S, E1R2, I1R2,
         E2S, I2S, E2R1, I2R1) = y

        b1 = self._beta_t(t, p.beta1, p.amplitude, p.phi_weeks) if self.seasonal else p.beta1
        b2 = self._beta_t(t, p.beta2, p.amplitude, p.phi_weeks) if self.seasonal else p.beta2

        # Force of infection for each strain
        lambda1 = b1 * (I1S + I1R2)
        lambda2 = b2 * (I2S + I2R1)

        # Reduced susceptibility terms (cross-immunity)
        s12 = max(0.0, min(1.0, 1.0 - p.sigma))  # susceptibility to 1 if recovered from 2
        s21 = max(0.0, min(1.0, 1.0 - p.sigma))  # susceptibility to 2 if recovered from 1

        # ODEs
        dS   = -(lambda1 + lambda2) * S
        dR1  = p.gamma * (I1S + I1R2) - s21 * lambda2 * R1
        dR2  = p.gamma * (I2S + I2R1) - s12 * lambda1 * R2
        dR12 = p.gamma * (I2R1 + I1R2)

        dE1S   = lambda1 * S            - p.k * E1S
        dI1S   = p.k * E1S              - p.gamma * I1S
        dE1R2  = s12 * lambda1 * R2     - p.k * E1R2
        dI1R2  = p.k * E1R2             - p.gamma * I1R2

        dE2S   = lambda2 * S            - p.k * E2S
        dI2S   = p.k * E2S              - p.gamma * I2S
        dE2R1  = s21 * lambda2 * R1     - p.k * E2R1
        dI2R1  = p.k * E2R1             - p.gamma * I2R1

        return np.array([dS, dR1, dR2, dR12,
                         dE1S, dI1S, dE1R2, dI1R2,
                         dE2S, dI2S, dE2R1, dI2R1])

    def simulate(self, t_grid: np.ndarray, params: TwoStrainParams):
        # Initial conditions
        S0  = max(1e-8, min(1.0, params.S0_frac))
        R10 = max(0.0, params.R10_frac)
        R20 = max(0.0, params.R20_frac)
        rem = max(0.0, 1.0 - (S0 + R10 + R20))
        y0 = np.array([
            S0, R10, R20, 0.0,          # S, R1, R2, R12
            params.seed1, 0.0,          # E1S, I1S
            0.0, 0.0,                   # E1R2, I1R2
            params.seed2, 0.0,          # E2S, I2S
            0.0, 0.0                    # E2R1, I2R1
        ], dtype=float)

        sol = solve_ivp(
            fun=lambda t, y: self.rhs(t, y, params),
            t_span=(t_grid[0], t_grid[-1]),
            y0=y0, t_eval=t_grid,
            rtol=1e-6, atol=1e-9, vectorized=False
        )
        if not sol.success:
            raise RuntimeError(f"TwoStrainSEIR integration failed: {sol.message}")
        return sol.t, sol.y.T

    @staticmethod
    def weekly_incidence(t, Y, k):
        """
        Approximate weekly incidence per strain using integral of k*E
        on each weekly interval via trapezoidal rule.
        Returns 2 arrays (inc1, inc2) each of length len(t)-1.
        """
        E1 = Y[:, 4] + Y[:, 6]   # E1S + E1R2
        E2 = Y[:, 8] + Y[:, 10]  # E2S + E2R1
        inc1, inc2 = [], []
        for i in range(len(t) - 1):
            h = t[i+1] - t[i]
            inc1.append(0.5 * h * k * (E1[i] + E1[i+1]))
            inc2.append(0.5 * h * k * (E2[i] + E2[i+1]))
        return np.array(inc1), np.array(inc2)


# =========================
# One-strain SEIR (total A+B)
# =========================
class OneStrainSEIR:
    """Minimal SEIR with seasonal forcing for TOTAL (A+B)."""
    def __init__(self, k=1/2.0, gamma=1/3.0, seasonal=True):
        self.k = k
        self.gamma = gamma
        self.seasonal = seasonal

    def _beta_t(self, t, beta, amplitude, phi_weeks):
        if not self.seasonal:
            return beta
        return beta * (1.0 + amplitude * np.cos(2.0 * np.pi * (t - phi_weeks) / 52.18))

    def rhs(self, t, y, beta, amplitude, phi_weeks):
        S, E, I, R = y
        bt = self._beta_t(t, beta, amplitude, phi_weeks)
        dS = -bt * S * I
        dE = bt * S * I - self.k * E
        dI = self.k * E - self.gamma * I
        dR = self.gamma * I
        return np.array([dS, dE, dI, dR])

    def simulate(self, t_grid, y0, beta, amplitude, phi_weeks):
        sol = solve_ivp(
            fun=lambda t, y: self.rhs(t, y, beta, amplitude, phi_weeks),
            t_span=(t_grid[0], t_grid[-1]),
            y0=y0, t_eval=t_grid,
            rtol=1e-6, atol=1e-9, vectorized=False
        )
        if not sol.success:
            raise RuntimeError(f"OneStrainSEIR integration failed: {sol.message}")
        return sol.t, sol.y.T

    def weekly_incidence(self, t, Y):
        E = Y[:, 1]
        out = []
        for i in range(len(t) - 1):
            h = t[i+1] - t[i]
            out.append(0.5 * h * self.k * (E[i] + E[i+1]))
        return np.array(out)


# =========================
# Fitting & Evaluation
# =========================
def evaluate_single_series(obs, pred, mask):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.asarray(mask, dtype=bool)

    def mae(a, b):
        return float(np.mean(np.abs(a - b))) if a.size else float("nan")

    def r2(a, b):
        if a.size == 0:
            return float("nan")
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "train": {"MAE": mae(obs[m], pred[m]), "R2": r2(obs[m], pred[m])},
        "test":  {"MAE": mae(obs[~m], pred[~m]), "R2": r2(obs[~m], pred[~m])},
    }


def evaluate_predictions(weeks_aln, obs_a_aln, obs_b_aln, pred_a_aln, pred_b_aln, mask_aln):
    """Evaluation used for TWO-STRAIN (on aligned arrays of length N-1)."""
    m = np.asarray(mask_aln, dtype=bool)

    def mae(y, yhat):
        return float(np.mean(np.abs(y - yhat))) if y.size else float("nan")

    def r2(y, yhat):
        if y.size == 0:
            return float("nan")
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    metrics = {
        "strain_A": {
            "train": {"MAE": mae(obs_a_aln[m], pred_a_aln[m]), "R2": r2(obs_a_aln[m], pred_a_aln[m])},
            "test":  {"MAE": mae(obs_a_aln[~m], pred_a_aln[~m]), "R2": r2(obs_a_aln[~m], pred_a_aln[~m])},
        },
        "strain_B": {
            "train": {"MAE": mae(obs_b_aln[m], pred_b_aln[m]), "R2": r2(obs_b_aln[m], pred_b_aln[m])},
            "test":  {"MAE": mae(obs_b_aln[~m], pred_b_aln[~m]), "R2": r2(obs_b_aln[~m], pred_b_aln[~m])},
        },
    }
    return metrics


def _peak(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.size == 0 or np.all(~np.isfinite(y)):
        return {"t": float("nan"), "value": float("nan")}
    i = int(np.nanargmax(y))
    return {"t": float(x[i]), "value": float(y[i])}


def fit_two_strain_on_window(weeks, obs_a, obs_b, train_mask, allow_cross_immunity=True):
    """
    Fit the two-strain model. We compute weekly incidence and align predictions
    to weeks[1:], so we also align observations to obs[1:].
    """
    weeks = np.asarray(weeks, dtype=float)
    obs_a = np.asarray(obs_a, dtype=float)
    obs_b = np.asarray(obs_b, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)

    # Align (drop the first week to match interval incidence)
    weeks_aln = weeks[1:]
    obs_a_aln = obs_a[1:]
    obs_b_aln = obs_b[1:]
    mask_aln = train_mask[1:]

    model = TwoStrainSEIR(seasonal=True)
    p = TwoStrainParams()
    if not allow_cross_immunity:
        p.sigma = 0.0

    # Parameter vector: [beta1, beta2, amplitude, phi, sigma, S0, R10, R20, seed1, seed2, rho1, rho2]
    # If sigma is disabled, we freeze it inside p and exclude from theta.
    def pack_params(theta):
        if allow_cross_immunity:
            (p.beta1, p.beta2, p.amplitude, p.phi_weeks, p.sigma,
             p.S0_frac, p.R10_frac, p.R20_frac, p.seed1, p.seed2, p.rho1, p.rho2) = theta
        else:
            (p.beta1, p.beta2, p.amplitude, p.phi_weeks,
             p.S0_frac, p.R10_frac, p.R20_frac, p.seed1, p.seed2, p.rho1, p.rho2) = theta

    def simulate(theta):
        pack_params(theta)
        t, Y = model.simulate(weeks, p)
        inc1, inc2 = model.weekly_incidence(t, Y, p.k)  # len N-1
        pred_a = p.rho1 * inc1
        pred_b = p.rho2 * inc2
        return pred_a, pred_b

    # Initial guess / bounds
    if allow_cross_immunity:
        theta0 = np.array([0.9, 0.9, 0.2, 2.0, 0.2, 0.95, 0.02, 0.02, 1e-6, 1e-6,
                           max(1.0, np.nanmax(obs_a_aln) * 0.5),
                           max(1.0, np.nanmax(obs_b_aln) * 0.5)])
        lower  = np.array([1e-3, 1e-3, 0.0, -52.0, 0.0, 0.50, 0.0, 0.0, 1e-10, 1e-10, 1e-6, 1e-6])
        upper  = np.array([5.0,   5.0,   0.9, 104.0, 1.0, 0.999, 0.5, 0.5, 1e-2,  1e-2,  1e9,  1e9])
    else:
        theta0 = np.array([0.9, 0.9, 0.2, 2.0, 0.95, 0.02, 0.02, 1e-6, 1e-6,
                           max(1.0, np.nanmax(obs_a_aln) * 0.5),
                           max(1.0, np.nanmax(obs_b_aln) * 0.5)])
        lower  = np.array([1e-3, 1e-3, 0.0, -52.0, 0.50, 0.0, 0.0, 1e-10, 1e-10, 1e-6, 1e-6])
        upper  = np.array([5.0,   5.0, 0.9, 104.0, 0.999, 0.5, 0.5, 1e-2,  1e-2,  1e9,  1e9])

    def residuals(theta):
        pred_a, pred_b = simulate(theta)
        res = np.zeros(obs_a_aln.size + obs_b_aln.size, dtype=float)
        m = mask_aln
        res_a = np.zeros_like(obs_a_aln)
        res_b = np.zeros_like(obs_b_aln)
        res_a[m] = pred_a[m] - obs_a_aln[m]
        res_b[m] = pred_b[m] - obs_b_aln[m]
        res[:obs_a_aln.size] = res_a
        res[obs_a_aln.size:] = res_b
        return res

    sol = least_squares(
        residuals, theta0, bounds=(lower, upper),
        method="trf", xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=300
    )

    pred_a_aln, pred_b_aln = simulate(sol.x)

    # Build params dict for JSON
    if allow_cross_immunity:
        keys = ["beta1","beta2","amplitude","phi_weeks","sigma","S0_frac","R10_frac","R20_frac",
                "seed1","seed2","rho1","rho2"]
    else:
        keys = ["beta1","beta2","amplitude","phi_weeks","S0_frac","R10_frac","R20_frac",
                "seed1","seed2","rho1","rho2"]
    params_dict = {k: float(v) for k, v in zip(keys, sol.x)}

    return {
        "weeks_aligned": weeks_aln,
        "obs_a_aligned": obs_a_aln,
        "obs_b_aligned": obs_b_aln,
        "pred_a_aligned": pred_a_aln,
        "pred_b_aligned": pred_b_aln,
        "train_mask_aligned": mask_aln,
        "params": params_dict,
    }


def fit_one_strain_total(weeks, obs_total, train_mask, seasonal=True):
    weeks = np.asarray(weeks, dtype=float)
    obs_total = np.asarray(obs_total, dtype=float)
    mask = np.asarray(train_mask, dtype=bool)

    weeks_aln = weeks[1:]
    obs_aln = obs_total[1:]
    mask_aln = mask[1:]

    model = OneStrainSEIR(seasonal=seasonal)

    def simulate(theta):
        # theta: [beta, amplitude, phi, S0, seedE0, rho]
        beta, amp, phi, S0, seedE0, rho = theta
        S0 = max(1e-8, min(1.0, S0))
        E0 = max(1e-10, seedE0)
        I0, R0 = 0.0, max(0.0, 1.0 - S0 - E0)
        y0 = np.array([S0, E0, I0, R0], dtype=float)
        t, Y = model.simulate(weeks, y0, beta, amp, phi)
        inc = model.weekly_incidence(t, Y)
        return rho * inc

    theta0 = np.array([0.9, 0.2, 2.0, 0.95, 1e-6, max(1.0, np.nanmax(obs_aln) * 0.5)])
    lower  = np.array([1e-3, 0.0, -52.0, 0.50, 1e-10, 1e-6])
    upper  = np.array([5.0, 0.9, 104.0, 0.999, 1e-2, 1e9])

    def residuals(theta):
        pred = simulate(theta)
        res = np.zeros_like(obs_aln)
        res[mask_aln] = pred[mask_aln] - obs_aln[mask_aln]
        return res

    sol = least_squares(
        residuals, theta0, bounds=(lower, upper),
        method="trf", xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=200
    )
    pred = simulate(sol.x)
    params = {k: float(v) for k, v in zip(
        ["beta","amplitude","phi_weeks","S0_frac","seed_E0","rho"], sol.x
    )}
    params["k"] = float(model.k)
    params["gamma"] = float(model.gamma)

    metrics = evaluate_single_series(obs_aln, pred, mask_aln)

    return {
        "weeks_aligned": weeks_aln,
        "obs_aligned": obs_aln,
        "pred_total": pred,
        "train_mask_aligned": mask_aln,
        "params": params,
        "metrics": metrics,
    }


# =========================
# Plotting (with TRAIN shading)
# =========================
def shade_train(ax, x_aligned, mask_aligned, label="Train window"):
    """Shade the continuous region that covers the training weeks (aligned axis)."""
    if mask_aligned.size == 0:
        return
    if not np.any(mask_aligned):
        return
    xs = x_aligned[mask_aligned]
    if xs.size == 0:
        return
    ax.axvspan(xs.min(), xs.max(), color="k", alpha=0.07, label=label)


def plot_two_strain(
    save_dir, season_label, weeks, obs_a, obs_b,
    fit_with, fit_without=None
):
    """
    Create TWO plots:
      1) With cross-immunity
      2) Without cross-immunity (if provided)

    Each plot shades the training window (requested refinement).
    All series are plotted on ALIGNED axis weeks[1:].
    """
    ensure_dir(save_dir)

    def _plot_block(tag, fit):
        fig, ax = plt.subplots(figsize=(11, 5), dpi=140)

        # Aligned axis and series
        x  = fit["weeks_aligned"]
        oa = fit["obs_a_aligned"]
        ob = fit["obs_b_aligned"]
        pa = fit["pred_a_aligned"]
        pb = fit["pred_b_aligned"]
        m  = fit["train_mask_aligned"]

        # Observed
        ax.plot(x, oa, lw=2, label="Observed A")
        ax.plot(x, ob, lw=2, label="Observed B")

        # Predicted
        ax.plot(x, pa, lw=2, ls="--", label="Predicted A")
        ax.plot(x, pb, lw=2, ls="--", label="Predicted B")

        # Shade train
        shade_train(ax, x, m, label="Train window")

        ax.set_title(f"Two-Strain SEIR ({tag}) — {season_label}")
        ax.set_xlabel("Week (aligned: weeks[1:])")
        ax.set_ylabel("Weekly count")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        out = os.path.join(save_dir, f"two_strain_{tag.replace(' ','_')}_{season_label.replace(' ','_')}.png")
        fig.savefig(out)
        plt.close(fig)
        return out

    paths = []
    paths.append(_plot_block("with cross-immunity", fit_with))
    if fit_without is not None:
        paths.append(_plot_block("no cross-immunity", fit_without))
    return paths


def plot_one_strain_vs_total(save_dir, season_label, one_fit, total_obs_full_weeks):
    """
    Plot ONE-STRAIN prediction vs OBSERVED TOTAL (A+B).
    We plot on aligned axis weeks[1:], shading the train window.
    """
    ensure_dir(save_dir)
    x  = one_fit["weeks_aligned"]
    y  = one_fit["obs_aligned"]         # total observed on aligned grid
    p  = one_fit["pred_total"]
    m  = one_fit["train_mask_aligned"]

    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    ax.plot(x, y, lw=2, label="Observed total (A+B)")
    ax.plot(x, p, lw=2, ls="--", label="One-strain SEIR (pred)")

    shade_train(ax, x, m, label="Train window")

    ax.set_title(f"One-Strain SEIR vs Total (A+B) — {season_label}")
    ax.set_xlabel("Week (aligned: weeks[1:])")
    ax.set_ylabel("Weekly count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = os.path.join(save_dir, f"one_strain_total_{season_label.replace(' ','_')}.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# =========================
# JSON writers (compact)
# =========================
def write_two_strain_json(save_dir, season, region, fit_with, metrics_with, fit_without=None, metrics_without=None):
    ensure_dir(save_dir)
    payload = {
        "season": season,
        "region": region,
        "metrics_with_cross_immunity": metrics_with,
        "params_with_cross_immunity": fit_with["params"],
        "peaks_with_cross_immunity": {
            "obs_A": _peak(fit_with["obs_a_aligned"], fit_with["weeks_aligned"]),
            "obs_B": _peak(fit_with["obs_b_aligned"], fit_with["weeks_aligned"]),
            "pred_A": _peak(fit_with["pred_a_aligned"], fit_with["weeks_aligned"]),
            "pred_B": _peak(fit_with["pred_b_aligned"], fit_with["weeks_aligned"]),
        },
    }
    if fit_without is not None and metrics_without is not None:
        payload["metrics_no_cross_immunity"] = metrics_without
        payload["params_no_cross_immunity"] = fit_without["params"]
        payload["peaks_no_cross_immunity"] = {
            "obs_A": _peak(fit_without["obs_a_aligned"], fit_without["weeks_aligned"]),
            "obs_B": _peak(fit_without["obs_b_aligned"], fit_without["weeks_aligned"]),
            "pred_A": _peak(fit_without["pred_a_aligned"], fit_without["weeks_aligned"]),
            "pred_B": _peak(fit_without["pred_b_aligned"], fit_without["weeks_aligned"]),
        }

    out = os.path.join(save_dir, "two_strain_metrics.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"Saved TWO-STRAIN JSON -> {out}")


def write_one_strain_json(save_dir, season, region, one_fit):
    ensure_dir(save_dir)
    payload = {
        "season": season,
        "region": region,
        "params": one_fit["params"],
        "metrics_total": one_fit["metrics"],
        "peaks_total": {
            "observed": _peak(one_fit["obs_aligned"], one_fit["weeks_aligned"]),
            "predicted": _peak(one_fit["pred_total"], one_fit["weeks_aligned"]),
        },
    }
    out = os.path.join(save_dir, "one_strain_metrics.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"Saved ONE-STRAIN JSON -> {out}")


# =========================
# Main orchestration
# =========================
def run(
    csv_path: str = DATA_PHL,
    region: str | None = USER_REGION,
    season: str | None = USER_SEASON,
    save_dir: str = SAVE_DIR,
    train_frac: float = TRAIN_FRAC,
):
    # ---- Load & aggregate
    log("Loading CSV...")
    df = load_csv(csv_path)
    if region is None:
        region = pick_default_region(df)
    log(f"Using region: {region}")

    log("Aggregating to A/B weekly counts...")
    piv = aggregate_two_strains(df, region)

    if season is None:
        season = choose_season_with_most_weeks(piv)
    log(f"Using season: {season}")

    s = extract_season_slice(piv, season)
    if s.empty:
        raise ValueError(f"No data for region={region} season={season}")

    # ---- Week axis and series
    weeks = build_week_axis(s["weekending"])         # 0..N-1
    obs_a = s["A"].to_numpy(dtype=float)
    obs_b = s["B"].to_numpy(dtype=float)
    total = s["total"].to_numpy(dtype=float)

    # Train/test mask on FULL (then aligned by dropping first element)
    train_mask_full = make_train_mask(len(weeks), train_frac)
    log(f"Train weeks: {train_mask_full.sum()} | Test weeks: {(~train_mask_full).sum()}")

    # ---- Two-strain fit (with cross-immunity)
    log("Fitting TWO-STRAIN (with cross-immunity)...")
    fit_with = fit_two_strain_on_window(weeks, obs_a, obs_b, train_mask_full, allow_cross_immunity=True)
    metrics_with = evaluate_predictions(
        fit_with["weeks_aligned"],
        fit_with["obs_a_aligned"], fit_with["obs_b_aligned"],
        fit_with["pred_a_aligned"], fit_with["pred_b_aligned"],
        fit_with["train_mask_aligned"]
    )

    # ---- Two-strain fit (no cross-immunity)
    log("Fitting TWO-STRAIN (no cross-immunity)...")
    fit_without = fit_two_strain_on_window(weeks, obs_a, obs_b, train_mask_full, allow_cross_immunity=False)
    metrics_without = evaluate_predictions(
        fit_without["weeks_aligned"],
        fit_without["obs_a_aligned"], fit_without["obs_b_aligned"],
        fit_without["pred_a_aligned"], fit_without["pred_b_aligned"],
        fit_without["train_mask_aligned"]
    )

    # ---- One-strain fit on TOTAL
    log("Fitting ONE-STRAIN on total (A+B)...")
    one_fit = fit_one_strain_total(weeks, total, train_mask_full, seasonal=True)

    # ---- Plots (with TRAIN shading for two-strain)
    log("Saving plots...")
    ensure_dir(save_dir)
    two_paths = plot_two_strain(save_dir, season, weeks, obs_a, obs_b, fit_with, fit_without)
    one_path  = plot_one_strain_vs_total(save_dir, season, one_fit, total)

    # ---- JSON outputs
    log("Writing JSON...")
    write_two_strain_json(save_dir, season, region, fit_with, metrics_with, fit_without, metrics_without)
    write_one_strain_json(save_dir, season, region, one_fit)

    log("Done.")


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    run()
