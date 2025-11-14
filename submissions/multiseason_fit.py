"""
multiseason_fit.py

Train on MULTIPLE seasons and evaluate on HELD-OUT seasons
for both:
  (1) Two-strain SEIR (A and B) with/without cross-immunity
  (2) One-strain SEIR on TOTAL (A+B)

Usage
-----
python multiseason_fit.py

Quick edits near top:
- DATA_PHL: path to CSV
- USER_REGION: region to use (None => auto-pick most populated)
- TRAIN_SEASONS / TEST_SEASONS: lists of season strings, e.g. ["2015-2016", "2016-2017"]
- SAVE_DIR: where figures and JSON will be written

Requires: numpy, pandas, matplotlib, scipy
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
DATA_PHL = "public-health-laboratory-influenza-respiratory-virus-surveillance-data-by-region-and-influenza-season.csv"
if not os.path.exists(DATA_PHL):
    DATA_PHL = os.path.join("data", "public-health-laboratory-influenza-respiratory-virus-surveillance-data-by-region-and-influenza-season.csv")

SAVE_DIR = "outputs_multiseason"
USER_REGION = None  # e.g., "Bay Area" (None => auto-pick most rows)

# Choose seasons explicitly; if empty, script will auto-pick by sorting unique seasons and split first half/second half
TRAIN_SEASONS: list[str] = []  # e.g., ["2014-2015", "2015-2016", "2016-2017"]
TEST_SEASONS:  list[str] = []  # e.g., ["2017-2018", "2018-2019"]

# =========================
# Small utilities
# =========================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def to_datetime_safe(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce", infer_datetime_format=True)

# =========================
# Data loading & aggregation
# =========================
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find CSV at {path}. "
            "Place the file next to this script or inside ./data/ with the same filename."
        )
    df = pd.read_csv(path)
    needed = {"season", "weekending", "region", "Influenza_Category", "Count"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    df["weekending"] = to_datetime_safe(df["weekending"])
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0.0)
    df["Influenza_Category"] = df["Influenza_Category"].astype(str)
    df["region"] = df["region"].astype(str)
    df["season"] = df["season"].astype(str)
    return df

def pick_default_region(df: pd.DataFrame) -> str:
    return df["region"].value_counts().idxmax()

def map_category_to_strain(cat: str) -> str | None:
    s = cat.strip().lower()
    if s.startswith("influenza_a") or ("influenza" in s and (" a" in s or "_a" in s)):
        return "A"
    if s.startswith("influenza b") or s.startswith("influenza_b") or ("influenza" in s and (" b" in s or "_b" in s)):
        return "B"
    return None

def aggregate_two_strains(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    d = df[df["region"] == region_name].copy()
    if d.empty: raise ValueError(f"No rows for region '{region_name}'.")
    d["strain"] = d["Influenza_Category"].map(map_category_to_strain)
    d = d[~d["strain"].isna()].copy()
    g = (
        d.groupby(["season", "weekending", "strain"], as_index=False)["Count"]
        .sum()
        .rename(columns={"Count": "count"})
    )
    piv = g.pivot_table(
        index=["season", "weekending"], columns="strain", values="count",
        aggfunc="sum", fill_value=0.0
    ).reset_index()
    piv.columns.name = None
    for col in ["A", "B"]:
        if col not in piv.columns: piv[col] = 0.0
    piv["total"] = piv["A"] + piv["B"]
    piv = piv.sort_values(["season", "weekending"]).reset_index(drop=True)
    return piv

def build_week_axis(weekending_col: pd.Series) -> np.ndarray:
    order = np.argsort(weekending_col.values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks

def prepare_season_series(piv: pd.DataFrame) -> dict:
    """
    Returns:
      seasons_dict = {
        season: {
          "weeks": np.ndarray (N,),
          "obs_a": np.ndarray (N,),
          "obs_b": np.ndarray (N,),
          "total": np.ndarray (N,),
        }, ...
      }
    """
    seasons_dict = {}
    for s, df_s in piv.groupby("season"):
        df_s = df_s.sort_values("weekending").reset_index(drop=True)
        weeks = build_week_axis(df_s["weekending"])
        seasons_dict[s] = {
            "weeks": weeks.astype(float),
            "obs_a": df_s["A"].to_numpy(dtype=float),
            "obs_b": df_s["B"].to_numpy(dtype=float),
            "total": df_s["total"].to_numpy(dtype=float),
            "dates": df_s["weekending"].to_numpy(),  # for labeling if needed
        }
    return seasons_dict

# =========================
# SEIR models
# =========================
@dataclass
class TwoStrainParams:
    beta1: float = 0.9
    beta2: float = 0.9
    amplitude: float = 0.2
    phi_weeks: float = 2.0
    sigma: float = 0.20
    k: float = 1 / 2.0
    gamma: float = 1 / 3.0
    S0_frac: float = 0.95
    R10_frac: float = 0.02
    R20_frac: float = 0.02
    seed1: float = 1e-6
    seed2: float = 1e-6
    rho1: float = 50.0
    rho2: float = 50.0

class TwoStrainSEIR:
    def __init__(self, seasonal=True): self.seasonal = seasonal
    @staticmethod
    def _beta_t(t, beta, amplitude, phi_weeks):
        return beta * (1.0 + amplitude * np.cos(2.0 * np.pi * (t - phi_weeks) / 52.18))
    def rhs(self, t, y, p: TwoStrainParams):
        (S, R1, R2, R12, E1S, I1S, E1R2, I1R2, E2S, I2S, E2R1, I2R1) = y
        b1 = self._beta_t(t, p.beta1, p.amplitude, p.phi_weeks) if self.seasonal else p.beta1
        b2 = self._beta_t(t, p.beta2, p.amplitude, p.phi_weeks) if self.seasonal else p.beta2
        lam1 = b1 * (I1S + I1R2)
        lam2 = b2 * (I2S + I2R1)
        s12 = max(0.0, min(1.0, 1.0 - p.sigma))
        s21 = max(0.0, min(1.0, 1.0 - p.sigma))
        dS   = -(lam1 + lam2) * S
        dR1  = p.gamma * (I1S + I1R2) - s21 * lam2 * R1
        dR2  = p.gamma * (I2S + I2R1) - s12 * lam1 * R2
        dR12 = p.gamma * (I2R1 + I1R2)
        dE1S   = lam1 * S        - p.k * E1S
        dI1S   = p.k * E1S       - p.gamma * I1S
        dE1R2  = s12 * lam1 * R2 - p.k * E1R2
        dI1R2  = p.k * E1R2      - p.gamma * I1R2
        dE2S   = lam2 * S        - p.k * E2S
        dI2S   = p.k * E2S       - p.gamma * I2S
        dE2R1  = s21 * lam2 * R1 - p.k * E2R1
        dI2R1  = p.k * E2R1      - p.gamma * I2R1
        return np.array([dS,dR1,dR2,dR12,dE1S,dI1S,dE1R2,dI1R2,dE2S,dI2S,dE2R1,dI2R1])
    def simulate(self, t_grid: np.ndarray, p: TwoStrainParams):
        S0 = max(1e-8, min(1.0, p.S0_frac))
        y0 = np.array([S0, p.R10_frac, p.R20_frac, 0.0, p.seed1, 0.0, 0.0, 0.0, p.seed2, 0.0, 0.0, 0.0], dtype=float)
        sol = solve_ivp(lambda t,y: self.rhs(t,y,p), (t_grid[0], t_grid[-1]), y0, t_eval=t_grid, rtol=1e-6, atol=1e-9)
        if not sol.success: raise RuntimeError(f"TwoStrainSEIR integration failed: {sol.message}")
        return sol.t, sol.y.T
    @staticmethod
    def weekly_incidence(t, Y, k):
        E1 = Y[:,4] + Y[:,6]
        E2 = Y[:,8] + Y[:,10]
        inc1, inc2 = [], []
        for i in range(len(t)-1):
            h = t[i+1]-t[i]
            inc1.append(0.5*h*k*(E1[i]+E1[i+1]))
            inc2.append(0.5*h*k*(E2[i]+E2[i+1]))
        return np.array(inc1), np.array(inc2)

class OneStrainSEIR:
    def __init__(self, k=1/2.0, gamma=1/3.0, seasonal=True):
        self.k, self.gamma, self.seasonal = k, gamma, seasonal
    def _beta_t(self, t, beta, amplitude, phi_weeks):
        return beta * (1.0 + amplitude * np.cos(2.0 * np.pi * (t - phi_weeks) / 52.18))
    def rhs(self, t, y, beta, amplitude, phi_weeks):
        S,E,I,R = y
        bt = self._beta_t(t, beta, amplitude, phi_weeks)
        dS = -bt * S * I
        dE = bt * S * I - self.k * E
        dI = self.k * E - self.gamma * I
        dR = self.gamma * I
        return np.array([dS,dE,dI,dR])
    def simulate(self, t_grid, y0, beta, amplitude, phi_weeks):
        sol = solve_ivp(lambda t,y: self.rhs(t,y,beta,amplitude,phi_weeks), (t_grid[0], t_grid[-1]), y0, t_eval=t_grid, rtol=1e-6, atol=1e-9)
        if not sol.success: raise RuntimeError(f"OneStrainSEIR integration failed: {sol.message}")
        return sol.t, sol.y.T
    def weekly_incidence(self, t, Y):
        E = Y[:,1]
        out=[]
        for i in range(len(t)-1):
            h = t[i+1]-t[i]
            out.append(0.5*h*self.k*(E[i]+E[i+1]))
        return np.array(out)

# =========================
# Metrics & helpers
# =========================
def mae(a, b): 
    a,b = np.asarray(a), np.asarray(b)
    return float(np.mean(np.abs(a-b))) if a.size else float("nan")

def r2(a, b):
    a,b = np.asarray(a), np.asarray(b)
    if a.size==0: return float("nan")
    ss_res = np.sum((a-b)**2); ss_tot = np.sum((a-np.mean(a))**2)
    return float(1.0 - ss_res/ss_tot) if ss_tot>0 else float("nan")

def _peak(y, x):
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    if y.size==0 or np.all(~np.isfinite(y)): return {"t": float("nan"), "value": float("nan")}
    i = int(np.nanargmax(y)); return {"t": float(x[i]), "value": float(y[i])}

# =========================
# Multi-season fitting: TWO-STRAIN
# =========================
def fit_two_strain_multiseason(seasons_dict: dict, train_seasons: list[str], allow_cross_immunity=True):
    """
    Fit SHARED parameters across all training seasons.
    Returns: fit object with per-season predictions (for ALL seasons).
    """
    model = TwoStrainSEIR(seasonal=True)
    p = TwoStrainParams()
    # parameter vector (sigma included/excluded)
    if allow_cross_immunity:
        theta0 = np.array([0.9,0.9,0.2,2.0,0.2, 0.95,0.02,0.02, 1e-6,1e-6, 50.0,50.0])
        lower  = np.array([1e-3,1e-3,0.0,-52.0,0.0, 0.50,0.0,0.0, 1e-10,1e-10, 1e-6,1e-6])
        upper  = np.array([5.0, 5.0, 0.9,104.0,1.0, 0.999,0.5,0.5, 1e-2, 1e-2, 1e9,  1e9])
    else:
        theta0 = np.array([0.9,0.9,0.2,2.0,       0.95,0.02,0.02, 1e-6,1e-6, 50.0,50.0])
        lower  = np.array([1e-3,1e-3,0.0,-52.0,   0.50,0.0,0.0,   1e-10,1e-10,1e-6,1e-6])
        upper  = np.array([5.0, 5.0, 0.9,104.0,   0.999,0.5,0.5,  1e-2, 1e-2, 1e9, 1e9])

    def pack(theta):
        if allow_cross_immunity:
            (p.beta1,p.beta2,p.amplitude,p.phi_weeks,p.sigma,
             p.S0_frac,p.R10_frac,p.R20_frac,p.seed1,p.seed2,p.rho1,p.rho2) = theta
        else:
            (p.beta1,p.beta2,p.amplitude,p.phi_weeks,
             p.S0_frac,p.R10_frac,p.R20_frac,p.seed1,p.seed2,p.rho1,p.rho2) = theta

    def simulate_one_season(theta, weeks, obs_a, obs_b):
        pack(theta)
        t, Y = model.simulate(weeks, p)
        inc1, inc2 = model.weekly_incidence(t, Y, p.k)  # len N-1
        pred_a = p.rho1 * inc1
        pred_b = p.rho2 * inc2
        return weeks[1:], obs_a[1:], obs_b[1:], pred_a, pred_b

    # Residuals: concatenate all training seasons (A then B for each season)
    def residuals(theta):
        res_blocks = []
        for s in train_seasons:
            d = seasons_dict[s]
            x, oa, ob, pa, pb = simulate_one_season(theta, d["weeks"], d["obs_a"], d["obs_b"])
            res_blocks.append(pa - oa)
            res_blocks.append(pb - ob)
        return np.concatenate(res_blocks) if res_blocks else np.array([0.0])

    sol = least_squares(residuals, theta0, bounds=(lower, upper), method="trf",
                        xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=400)

    # Build per-season predictions (for all seasons)
    per_season = {}
    for s, d in seasons_dict.items():
        x, oa, ob, pa, pb = simulate_one_season(sol.x, d["weeks"], d["obs_a"], d["obs_b"])
        per_season[s] = {
            "weeks_aligned": x, "obs_a_aligned": oa, "obs_b_aligned": ob,
            "pred_a_aligned": pa, "pred_b_aligned": pb
        }

    # Metrics per season
    metrics = {}
    for s in seasons_dict:
        d = per_season[s]
        metrics[s] = {
            "A": {"MAE": mae(d["obs_a_aligned"], d["pred_a_aligned"]),
                  "R2":  r2 (d["obs_a_aligned"], d["pred_a_aligned"])},
            "B": {"MAE": mae(d["obs_b_aligned"], d["pred_b_aligned"]),
                  "R2":  r2 (d["obs_b_aligned"], d["pred_b_aligned"])},
        }

    # Params dict
    if allow_cross_immunity:
        keys = ["beta1","beta2","amplitude","phi_weeks","sigma","S0_frac","R10_frac","R20_frac","seed1","seed2","rho1","rho2"]
    else:
        keys = ["beta1","beta2","amplitude","phi_weeks","S0_frac","R10_frac","R20_frac","seed1","seed2","rho1","rho2"]
    params_dict = {k: float(v) for k, v in zip(keys, sol.x)}

    return {"params": params_dict, "per_season": per_season, "metrics_per_season": metrics}

# =========================
# Multi-season fitting: ONE-STRAIN
# =========================
def fit_one_strain_multiseason(seasons_dict: dict, train_seasons: list[str], seasonal=True):
    model = OneStrainSEIR(seasonal=seasonal)

    # theta: [beta, amplitude, phi, S0, seedE0, rho]
    theta0 = np.array([0.9,0.2,2.0,0.95,1e-6, 50.0])
    lower  = np.array([1e-3,0.0,-52.0,0.50,1e-10,1e-6])
    upper  = np.array([5.0, 0.9,104.0,0.999,1e-2, 1e9])

    def simulate_season(theta, weeks, total):
        beta, amp, phi, S0, seedE0, rho = theta
        S0 = max(1e-8, min(1.0, S0))
        E0 = max(1e-10, seedE0)
        y0 = np.array([S0, E0, 0.0, max(0.0, 1.0 - S0 - E0)], dtype=float)
        t, Y = model.simulate(weeks, y0, beta, amp, phi)
        inc = model.weekly_incidence(t, Y)             # len N-1
        pred = rho * inc
        return weeks[1:], total[1:], pred

    def residuals(theta):
        res_blocks=[]
        for s in train_seasons:
            d = seasons_dict[s]
            x, obs, pred = simulate_season(theta, d["weeks"], d["total"])
            res_blocks.append(pred - obs)
        return np.concatenate(res_blocks) if res_blocks else np.array([0.0])

    sol = least_squares(residuals, theta0, bounds=(lower, upper), method="trf",
                        xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=300)

    # Per-season predictions
    per_season = {}
    for s, d in seasons_dict.items():
        x, obs, pred = simulate_season(sol.x, d["weeks"], d["total"])
        per_season[s] = {"weeks_aligned": x, "obs_aligned": obs, "pred_total": pred}

    # Metrics per season
    metrics = {}
    for s in seasons_dict:
        d = per_season[s]
        metrics[s] = {"MAE": mae(d["obs_aligned"], d["pred_total"]),
                      "R2":  r2 (d["obs_aligned"], d["pred_total"])}

    params = {k: float(v) for k, v in zip(["beta","amplitude","phi_weeks","S0_frac","seed_E0","rho"], sol.x)}
    params["k"] = float(model.k); params["gamma"] = float(model.gamma)

    return {"params": params, "per_season": per_season, "metrics_per_season": metrics}

# =========================
# Plotting (per season)
# =========================
def shade_train_if(ax, x, is_train, label="Train season"):
    if is_train and x.size:
        ax.axvspan(x.min(), x.max(), color="k", alpha=0.07, label=label)

def plot_two_strain_per_season(save_dir, region, with_fit, without_fit, train_seasons: list[str]):
    ensure_dir(save_dir)
    paths = []
    for s, d in with_fit["per_season"].items():
        fig, ax = plt.subplots(figsize=(11,5), dpi=140)
        x = d["weeks_aligned"]
        oa, ob = d["obs_a_aligned"], d["obs_b_aligned"]
        pa, pb = d["pred_a_aligned"], d["pred_b_aligned"]
        ax.plot(x, oa, lw=2, label="Obs A"); ax.plot(x, ob, lw=2, label="Obs B")
        ax.plot(x, pa, lw=2, ls="--", label="Pred A (with)"); ax.plot(x, pb, lw=2, ls="--", label="Pred B (with)")

        if without_fit is not None and s in without_fit["per_season"]:
            dw = without_fit["per_season"][s]
            ax.plot(x, dw["pred_a_aligned"], lw=1.5, ls=":", label="Pred A (no x-im)")
            ax.plot(x, dw["pred_b_aligned"], lw=1.5, ls=":", label="Pred B (no x-im)")

        shade_train_if(ax, x, s in train_seasons, label="Train season window")
        ax.set_title(f"Two-Strain — {region} — {s}")
        ax.set_xlabel("Week (aligned: weeks[1:])"); ax.set_ylabel("Weekly count")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")
        fig.tight_layout()
        out = os.path.join(save_dir, f"two_strain_{s.replace(' ','_')}.png"); fig.savefig(out); plt.close(fig)
        paths.append(out)
    return paths

def plot_one_strain_per_season(save_dir, region, one_fit, train_seasons: list[str]):
    ensure_dir(save_dir)
    paths=[]
    for s, d in one_fit["per_season"].items():
        x, obs, pred = d["weeks_aligned"], d["obs_aligned"], d["pred_total"]
        fig, ax = plt.subplots(figsize=(11,5), dpi=140)
        ax.plot(x, obs, lw=2, label="Obs total (A+B)")
        ax.plot(x, pred, lw=2, ls="--", label="Pred total (one-strain)")
        shade_train_if(ax, x, s in train_seasons, label="Train season window")
        ax.set_title(f"One-Strain — {region} — {s}")
        ax.set_xlabel("Week (aligned: weeks[1:])"); ax.set_ylabel("Weekly count")
        ax.grid(True, alpha=0.3); ax.legend(loc="best")
        fig.tight_layout()
        out = os.path.join(save_dir, f"one_strain_{s.replace(' ','_')}.png"); fig.savefig(out); plt.close(fig)
        paths.append(out)
    return paths

# =========================
# JSON writers
# =========================
def write_multiseason_json(save_dir, region, train_seasons, test_seasons, two_with, two_without, one_fit):
    ensure_dir(save_dir)
    payload = {
        "meta": {
            "region": region,
            "train_seasons": list(train_seasons),
            "test_seasons": list(test_seasons),
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        "two_strain": {
            "with_cross_immunity": {
                "params": two_with["params"],
                "metrics_per_season": two_with["metrics_per_season"],
            }
        },
        "one_strain": {
            "params": one_fit["params"],
            "metrics_per_season": one_fit["metrics_per_season"],
        }
    }
    if two_without is not None:
        payload["two_strain"]["no_cross_immunity"] = {
            "params": two_without["params"],
            "metrics_per_season": two_without["metrics_per_season"],
        }
    out = os.path.join(save_dir, "multiseason_report.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"Saved JSON -> {out}")

# =========================
# Orchestration
# =========================
def run_multiseason(
    csv_path: str = DATA_PHL,
    region: str | None = USER_REGION,
    train_seasons: list[str] = TRAIN_SEASONS,
    test_seasons: list[str]  = TEST_SEASONS,
    save_dir: str = SAVE_DIR,
):
    log("Loading CSV...")
    df = load_csv(csv_path)
    if region is None:
        region = pick_default_region(df)
    log(f"Region: {region}")

    log("Aggregating weekly A/B by season...")
    piv = aggregate_two_strains(df, region)
    seasons_avail = sorted(piv["season"].unique().tolist())
    log(f"Seasons available ({len(seasons_avail)}): {', '.join(seasons_avail)}")

    # Auto split if not provided: first half train, second half test
    if not train_seasons and not test_seasons:
        n = len(seasons_avail)
        k = max(1, n//2)
        train_seasons = seasons_avail[:k]
        test_seasons  = seasons_avail[k:]
        log(f"Auto split -> Train: {train_seasons} | Test: {test_seasons}")

    seasons_dict = prepare_season_series(piv)

    # Fit TWO-STRAIN (with & without cross-immunity) on TRAIN seasons (shared params), evaluate ALL seasons
    log("Fitting TWO-STRAIN (with cross-immunity) across training seasons...")
    two_with = fit_two_strain_multiseason(seasons_dict, train_seasons, allow_cross_immunity=True)

    log("Fitting TWO-STRAIN (no cross-immunity) across training seasons...")
    two_without = fit_two_strain_multiseason(seasons_dict, train_seasons, allow_cross_immunity=False)

    # Fit ONE-STRAIN on TOTAL (A+B) across TRAIN seasons (shared params), evaluate ALL
    log("Fitting ONE-STRAIN across training seasons...")
    one_fit = fit_one_strain_multiseason(seasons_dict, train_seasons, seasonal=True)

    # Plots per season (train seasons shaded)
    log("Saving plots per season...")
    ensure_dir(save_dir)
    plot_two_strain_per_season(save_dir, region, two_with, two_without, train_seasons)
    plot_one_strain_per_season(save_dir, region, one_fit, train_seasons)

    # JSON summary
    log("Writing JSON summary...")
    write_multiseason_json(save_dir, region, train_seasons, test_seasons, two_with, two_without, one_fit)

    # Console summary of overall metrics (averaged)
    def avg_metrics_two(ms):
        A_mae = []; A_r2 = []; B_mae = []; B_r2 = []
        for s in test_seasons:
            if s in ms:
                A_mae.append(ms[s]["A"]["MAE"]); A_r2.append(ms[s]["A"]["R2"])
                B_mae.append(ms[s]["B"]["MAE"]); B_r2.append(ms[s]["B"]["R2"])
        return (np.nanmean(A_mae), np.nanmean(A_r2), np.nanmean(B_mae), np.nanmean(B_r2)) if A_mae else (np.nan,)*4

    def avg_metrics_one(ms):
        vals = [ms[s] for s in test_seasons if s in ms]
        if not vals: return (np.nan, np.nan)
        return (np.nanmean([v["MAE"] for v in vals]), np.nanmean([v["R2"] for v in vals]))

    A_mae_w, A_r2_w, B_mae_w, B_r2_w = avg_metrics_two(two_with["metrics_per_season"])
    A_mae_wo, A_r2_wo, B_mae_wo, B_r2_wo = avg_metrics_two(two_without["metrics_per_season"])
    one_mae, one_r2 = avg_metrics_one(one_fit["metrics_per_season"])

    log("=== Held-out (TEST seasons) average metrics ===")
    log(f"Two-strain WITH x-immunity :  A(MAE={A_mae_w:.2f}, R2={A_r2_w:.2f})  |  B(MAE={B_mae_w:.2f}, R2={B_r2_w:.2f})")
    log(f"Two-strain NO  x-immunity :  A(MAE={A_mae_wo:.2f}, R2={A_r2_wo:.2f}) |  B(MAE={B_mae_wo:.2f}, R2={B_r2_wo:.2f})")
    log(f"One-strain TOTAL (A+B)     :  MAE={one_mae:.2f}, R2={one_r2:.2f}")
    log("Done.")

# =========================
# Entry
# =========================
if __name__ == "__main__":
    run_multiseason()
