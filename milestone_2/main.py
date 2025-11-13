import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PHL = "data/public-health-laboratory-influenza-respiratory-virus-surveillance-data-by-region-and-influenza-season.csv"
USER_REGION = None  # e.g., "California", "Northern", "Bay Area", etc.
USER_SEASON = None  # e.g., "2017-2018"


def log(msg):
    print(f"[LOG] {msg}")


def parse_week(df, date_col="weekending"):
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        for c in df.columns:
            if "week" in c.lower():
                df[c] = pd.to_datetime(df[c], errors="coerce")
                date_col = c
                break
    return df, date_col


def choose_region(df):
    regions = sorted(df["region"].dropna().unique().tolist())
    for name in ["Statewide", "California", "CA", "All Regions"]:
        if name in regions:
            return name
    # Fallback: region with most records
    cnt = df.groupby("region").size().sort_values(ascending=False)
    return cnt.index[0]


def aggregate_phl_two_strains(df, region_name):
    df = df[df["region"] == region_name].copy()

    def to_strain(cat):
        if isinstance(cat, str):
            s = cat.strip().lower()
            s = s.replace(" ", "").replace("-", "").replace("/", "").replace("_", "")
            if s.startswith("influenzaa") or s.startswith("a(") or s == "a":
                return "A"
            if s.startswith("influenzab") or s.startswith("b(") or s == "b":
                return "B"
        return None

    df["strain"] = df["Influenza_Category"].apply(to_strain)
    df = df[df["strain"].isin(["A", "B"])].copy()

    df, date_col = parse_week(df, "weekending")
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0).astype(int)

    piv = (
        df.groupby(["season", date_col, "strain"])["Count"]
        .sum()
        .reset_index()
        .pivot(index=["season", date_col], columns="strain", values="Count")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "A" not in piv.columns:
        piv["A"] = 0
    if "B" not in piv.columns:
        piv["B"] = 0
    piv = piv.sort_values(["season", date_col])
    return piv, date_col


def select_best_season(piv, date_col):
    seasons = piv["season"].dropna().unique().tolist()
    best, best_score = None, -1
    for s in seasons:
        tmp = piv[piv["season"] == s]
        weeks = tmp[["A", "B"]].fillna(0).sum(axis=1)
        score = np.count_nonzero(weeks.values)
        if score > best_score:
            best, best_score = s, score
    return best


class TwoStrainSEIR:
    """
    Two-strain SEIR with symmetric partial cross-immunity.

    Compartments (12 total):
      S, R1, R2, R12,
      E1S, I1S, E1R2, I1R2,
      E2S, I2S, E2R1, I2R1

    Force of infection uses total I1 = I1S + I1R2, I2 = I2S + I2R1.
    """

    def __init__(
        self,
        N=1.0,
        k=4.5,
        gamma=2.0,
        sigma12=0.5,
        sigma21=0.5,
        amplitude=0.25,
        phi_weeks=0.0,
        beta1=1.4,
        beta2=1.2,
    ):
        self.N = float(N)
        self.k = float(k)
        self.gamma = float(gamma)
        self.sigma12 = float(sigma12)
        self.sigma21 = float(sigma21)
        self.amplitude = float(amplitude)
        self.phi_weeks = float(phi_weeks)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)

    def betas(self, t):
        season = 2.0 * math.pi * (t - self.phi_weeks) / 52.18
        mod = 1.0 + self.amplitude * math.cos(season)
        b1 = max(0.0, self.beta1 * mod)
        b2 = max(0.0, self.beta2 * mod)
        return b1, b2

    def rhs(self, t, y):
        S, R1, R2, R12, E1S, I1S, E1R2, I1R2, E2S, I2S, E2R1, I2R1 = y

        I1 = I1S + I1R2
        I2 = I2S + I2R1

        b1, b2 = self.betas(t)
        lam1 = b1 * I1 / self.N
        lam2 = b2 * I2 / self.N

        ne1_s = lam1 * S
        ne1_r2 = self.sigma21 * lam1 * R2
        ne2_s = lam2 * S
        ne2_r1 = self.sigma12 * lam2 * R1

        dS = -ne1_s - ne2_s
        dR1 = self.gamma * I1S - ne2_r1
        dR2 = self.gamma * I2S - ne1_r2
        dR12 = self.gamma * I1R2 + self.gamma * I2R1

        dE1S = ne1_s - self.k * E1S
        dI1S = self.k * E1S - self.gamma * I1S

        dE1R2 = ne1_r2 - self.k * E1R2
        dI1R2 = self.k * E1R2 - self.gamma * I1R2

        dE2S = ne2_s - self.k * E2S
        dI2S = self.k * E2S - self.gamma * I2S

        dE2R1 = ne2_r1 - self.k * E2R1
        dI2R1 = self.k * E2R1 - self.gamma * I2R1

        return [dS, dR1, dR2, dR12, dE1S, dI1S, dE1R2, dI1R2, dE2S, dI2S, dE2R1, dI2R1]

    def simulate(self, t_grid, y0):
        sol = solve_ivp(
            fun=self.rhs,
            t_span=(t_grid[0], t_grid[-1]),
            y0=y0,
            t_eval=t_grid,
            method="RK45",
            atol=1e-6,
            rtol=1e-3,
        )
        if not sol.success:
            log(f"ODE solver failed: {sol.message}")
        return sol.t, sol.y

    def weekly_incidence(self, t, y):
        E1S, E1R2 = y[4, :], y[6, :]
        E2S, E2R1 = y[8, :], y[10, :]
        inc1 = self.k * (E1S + E1R2)
        inc2 = self.k * (E2S + E2R1)
        return inc1, inc2


def fit_model(
    weeks,
    obs_a,
    obs_b,
    train_mask,
    init_guess,
    bounds,
    N_pop=1e6,
    fixed_params=None,
    max_nfev=120,
):
    """Fit by least squares on training weeks.

    Parameter vector layout:
      [beta1, beta2, amplitude, phi_weeks, sigma,
       S0_frac, R10_frac, R20_frac, seed1, seed2, rho1, rho2]
    """

    def unpack(p):
        d = {
            "beta1": p[0],
            "beta2": p[1],
            "amplitude": p[2],
            "phi": p[3],
            "sigma": p[4],
            "S0": p[5],
            "R10": p[6],
            "R20": p[7],
            "seed1": p[8],
            "seed2": p[9],
            "rho1": p[10],
            "rho2": p[11],
        }
        if fixed_params:
            d.update(fixed_params)
        return d

    t_eval = weeks - weeks[0]

    def residuals(p):
        pr = unpack(p)
        mdl = TwoStrainSEIR(
            N=N_pop,
            k=4.5,
            gamma=2.0,
            sigma12=pr["sigma"],
            sigma21=pr["sigma"],
            amplitude=pr["amplitude"],
            phi_weeks=pr["phi"],
            beta1=pr["beta1"],
            beta2=pr["beta2"],
        )
        y0 = [
            pr["S0"] * N_pop,
            pr["R10"] * N_pop,
            pr["R20"] * N_pop,
            0.0,
            0.0,
            pr["seed1"],
            0.0,
            0.0,
            0.0,
            pr["seed2"],
            0.0,
            0.0,
        ]
        total0 = sum(y0)
        if total0 > N_pop:
            y0[0] = max(0.0, y0[0] - (total0 - N_pop))
        tt, yy = mdl.simulate(t_eval, y0)
        inc1, inc2 = mdl.weekly_incidence(tt, yy)
        pred_a = pr["rho1"] * inc1
        pred_b = pr["rho2"] * inc2
        res = np.concatenate(
            [(pred_a - obs_a) * train_mask, (pred_b - obs_b) * train_mask]
        )
        return res

    log("Starting least squares fit...")
    result = least_squares(
        residuals,
        x0=np.array(init_guess, dtype=float),
        bounds=(np.array(bounds[0], dtype=float), np.array(bounds[1], dtype=float)),
        max_nfev=max_nfev,
        verbose=0,
    )
    pr = unpack(result.x)

    mdl = TwoStrainSEIR(
        N=N_pop,
        k=4.5,
        gamma=2.0,
        sigma12=pr["sigma"],
        sigma21=pr["sigma"],
        amplitude=pr["amplitude"],
        phi_weeks=pr["phi"],
        beta1=pr["beta1"],
        beta2=pr["beta2"],
    )
    y0 = [
        pr["S0"] * N_pop,
        pr["R10"] * N_pop,
        pr["R20"] * N_pop,
        0.0,
        0.0,
        pr["seed1"],
        0.0,
        0.0,
        0.0,
        pr["seed2"],
        0.0,
        0.0,
    ]
    total0 = sum(y0)
    if total0 > N_pop:
        y0[0] = max(0.0, y0[0] - (total0 - N_pop))

    tt, yy = mdl.simulate(t_eval, y0)
    inc1, inc2 = mdl.weekly_incidence(tt, yy)

    return {
        "params": pr,
        "t": tt,
        "y": yy,
        "inc_a": inc1,
        "inc_b": inc2,
        "pred_a": pr["rho1"] * inc1,
        "pred_b": pr["rho2"] * inc2,
        "train_mask": train_mask.astype(bool),
    }


def evaluate_predictions(weeks, obs_a, obs_b, pred_a, pred_b, train_mask):
    def peak_metrics(obs, pred, idx):
        i_obs = int(np.argmax(obs[idx]))
        i_pred = int(np.argmax(pred[idx]))
        t_obs = float(weeks[idx][i_obs])
        t_pred = float(weeks[idx][i_pred])
        return {
            "PeakErrWeeks": t_pred - t_obs,
            "RelPeakHeight": float(pred[idx][i_pred] / (obs[idx][i_obs] + 1e-9)),
            "ObsPeakWeek": t_obs,
            "PredPeakWeek": t_pred,
        }

    def summary(obs, pred, mask):
        idx = mask
        mae = mean_absolute_error(obs[idx], pred[idx])
        rmse = float(np.sqrt(np.mean((obs[idx] - pred[idx]) ** 2)))
        r2 = float(r2_score(obs[idx], pred[idx]))
        corr = (
            float(np.corrcoef(obs[idx], pred[idx])[0, 1])
            if np.std(pred[idx]) > 0
            else float("nan")
        )
        pk = peak_metrics(obs, pred, idx)
        return {"MAE": mae, "RMSE": rmse, "R2": r2, "Corr": corr, **pk}

    idx_tr = train_mask
    idx_te = ~train_mask

    return {
        "A": {
            "train": summary(obs_a, pred_a, idx_tr),
            "test": summary(obs_a, pred_a, idx_te),
        },
        "B": {
            "train": summary(obs_b, pred_b, idx_tr),
            "test": summary(obs_b, pred_b, idx_te),
        },
    }


def run(region=None, season=None, save_dir=None):
    if not os.path.exists(DATA_PHL):
        raise FileNotFoundError(DATA_PHL)
    phl = pd.read_csv(DATA_PHL)

    if region is None:
        region = choose_region(phl)
    log(f"Region = {region}")

    piv, date_col = aggregate_phl_two_strains(phl, region)
    if season is None:
        season = select_best_season(piv, date_col)
    log(f"Season = {season}")

    df = (
        piv[piv["season"] == season].copy().sort_values(date_col).reset_index(drop=True)
    )
    df["tweek"] = np.arange(len(df), dtype=float)

    obs_a = df["A"].astype(float).values
    obs_b = df["B"].astype(float).values
    weeks = df["tweek"].astype(float).values

    n = len(weeks)
    n_tr = max(10, int(0.7 * n))
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:n_tr] = True
    log(f"Weeks total={n}, training={n_tr}, testing={n - n_tr}")

    init_guess = [1.4, 1.2, 0.2, 5.0, 0.5, 0.85, 0.07, 0.07, 10.0, 10.0, 0.05, 0.05]
    lb = [0.2, 0.2, 0.0, 0.0, 0.0, 0.50, 0.00, 0.00, 0.0, 0.0, 1e-4, 1e-4]
    ub = [3.0, 3.0, 0.7, 52.18, 1.0, 0.98, 0.40, 0.40, 5e4, 5e4, 10.0, 10.0]
    bounds = (lb, ub)

    res_free = fit_model(
        weeks,
        obs_a,
        obs_b,
        train_mask,
        init_guess,
        bounds,
        N_pop=1_000_000.0,
        fixed_params=None,
        max_nfev=120,
    )
    res_nox = fit_model(
        weeks,
        obs_a,
        obs_b,
        train_mask,
        init_guess,
        bounds,
        N_pop=1_000_000.0,
        fixed_params={"sigma": 1.0},
        max_nfev=80,
    )

    metrics_free = evaluate_predictions(
        weeks,
        obs_a,
        obs_b,
        res_free["pred_a"],
        res_free["pred_b"],
        res_free["train_mask"],
    )
    metrics_nox = evaluate_predictions(
        weeks, obs_a, obs_b, res_nox["pred_a"], res_nox["pred_b"], res_nox["train_mask"]
    )

    log("=== Metrics (with cross-immunity) ===")
    print(json.dumps(metrics_free, indent=2))
    log("=== Metrics (no cross-immunity) ===")
    print(json.dumps(metrics_nox, indent=2))

    # Visualizations
    plt.figure(figsize=(10, 5))
    plt.plot(weeks, obs_a, label="Observed A")
    plt.plot(weeks, res_free["pred_a"], label="Pred A (with X)")
    plt.plot(weeks, res_nox["pred_a"], label="Pred A (no X)")
    plt.xlabel("Week index")
    plt.ylabel("Weekly positives (A)")
    plt.title(f"{region} {season} - Strain A")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "strain_A_fit.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(weeks, obs_b, label="Observed B")
    plt.plot(weeks, res_free["pred_b"], label="Pred B (with X)")
    plt.plot(weeks, res_nox["pred_b"], label="Pred B (no X)")
    plt.xlabel("Week index")
    plt.ylabel("Weekly positives (B)")
    plt.title(f"{region} {season} - Strain B")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "strain_B_fit.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(weeks, res_free["inc_a"], label="Model incidence A")
    plt.plot(weeks, res_free["inc_b"], label="Model incidence B")
    plt.xlabel("Week index")
    plt.ylabel("Model incidence")
    plt.title("Model incidence (with cross-immunity)")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "model_incidence.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(5, 5))
    lim = max(obs_a.max(), res_free["pred_a"].max(), 1.0)
    plt.scatter(obs_a, res_free["pred_a"], label="A")
    plt.plot([0, lim], [0, lim])
    plt.xlabel("Observed A")
    plt.ylabel("Predicted A")
    plt.title("Observed vs Predicted (A)")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "scatter_A.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(5, 5))
    lim = max(obs_b.max(), res_free["pred_b"].max(), 1.0)
    plt.scatter(obs_b, res_free["pred_b"], label="B")
    plt.plot([0, lim], [0, lim])
    plt.xlabel("Observed B")
    plt.ylabel("Predicted B")
    plt.title("Observed vs Predicted (B)")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "scatter_B.png"), dpi=150)
    plt.show()

    # Residuals plot
    resA = res_free["pred_a"] - obs_a
    resB = res_free["pred_b"] - obs_b
    plt.figure(figsize=(10, 4))
    plt.plot(weeks, resA, label="Residuals A")
    plt.plot(weeks, resB, label="Residuals B")
    plt.axvspan(weeks[0], weeks[n_tr - 1], alpha=0.15, label="Train window")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Week index")
    plt.ylabel("Prediction - Observation")
    plt.title(f"Residuals (with cross-immunity)\n{region} {season}")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "residuals.png"), dpi=150)
    plt.show()

    # Peak-succession report
    def peak_info(y):
        i = int(np.argmax(y))
        return i, float(y[i])

    a_obs_pk_w, a_obs_pk_v = peak_info(obs_a)
    b_obs_pk_w, b_obs_pk_v = peak_info(obs_b)
    a_pred_pk_w, a_pred_pk_v = peak_info(res_free["pred_a"])
    b_pred_pk_w, b_pred_pk_v = peak_info(res_free["pred_b"])

    peak_report = {
        "region": region,
        "season": season,
        "observed_peaks": {
            "A": {"week_idx": int(a_obs_pk_w), "value": a_obs_pk_v},
            "B": {"week_idx": int(b_obs_pk_w), "value": b_obs_pk_v},
        },
        "predicted_peaks_with_cross_immunity": {
            "A": {"week_idx": int(a_pred_pk_w), "value": a_pred_pk_v},
            "B": {"week_idx": int(b_pred_pk_w), "value": b_pred_pk_v},
        },
        "successive_peaks_observed_weeks": int(b_obs_pk_w - a_obs_pk_w),
        "successive_peaks_predicted_weeks": int(b_pred_pk_w - a_pred_pk_w),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "two_strain_metrics.json"), "w") as f:
            json.dump(
                {
                    "season": season,
                    "region": region,
                    "metrics_with_cross_immunity": metrics_free,
                    "metrics_no_cross_immunity": metrics_nox,
                    "params_with_cross_immunity": res_free["params"],
                    "params_no_cross_immunity": res_nox["params"],
                    "peak_report": peak_report,
                },
                f,
                indent=2,
            )
        log(f"Saved JSON metrics + peaks to {save_dir}/two_strain_metrics.json")

    return {
        "region": region,
        "season": season,
        "metrics_free": metrics_free,
        "metrics_nox": metrics_nox,
        "params_free": res_free["params"],
        "params_nox": res_nox["params"],
        "peak_report": peak_report,
    }


if __name__ == "__main__":
    out = run(region=USER_REGION, season=USER_SEASON, save_dir="out")
    log("Done.")
