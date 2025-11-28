"""
Phase 9 – Sensitivities & Robustness toolkit.

This module stress-tests the Phase 2 parameter set across the key shocks
outlined in the plan: single-factor tornado analysis, ELR × acquisition heatmap,
and coverage-unit diagnostics.  Results are written to `phase9_results/` and
summaries are printed to stdout for quick inspection.

Run from the repo root:

    python phase9.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from phases import (
    Phase7Params,
    build_phase7_summary,
    coverage_unit_robustness,
    run_paa_projection,
)


# ---------------------------------------------------------------------------
# Reinsurance configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReinsuranceConfig:
    qs_share: float
    xol_retention: float
    xol_limit: float
    xol_premium: float = 500_000_000  # Assumed XoL treaty cost for reinstatement base


BASE_REINSURANCE = ReinsuranceConfig(qs_share=0.50, xol_retention=1_500_000_000, xol_limit=5_000_000_000)


# ---------------------------------------------------------------------------
# Base parameters & payouts (Phase 2 locked)
# ---------------------------------------------------------------------------

BASE_PARAMS = Phase7Params(
    gwp=20_000_000_000, # Reverted to Old Plan
    acq_cost=7_000_000_000, # Reverted to Old Plan
    discount_rate_pa=0.0625,
    ra_margin=0.15,
    coverage_units=(1, 1, 1, 1, 2, 3, 3, 2, 2, 4, 4, 1), # Reverted to Seasonal
)

BASE_PAYOUTS = np.array(
    [
        0,
        3_500_000_000,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        4_200_000_000,
        4_300_000_000,
        0,
    ],
    dtype=float,
)

GWP_TOTAL = BASE_PARAMS.gwp
BASE_ELR = BASE_PAYOUTS.sum() / GWP_TOTAL
BASE_ACQ_RATE = BASE_PARAMS.acq_cost / GWP_TOTAL


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def apply_reinsurance(paa_df: pd.DataFrame, config: ReinsuranceConfig) -> pd.DataFrame:
    """Add reinsurance-adjusted columns to the PAA projection."""

    df = paa_df.copy()

    df["ceded_claims_qs"] = config.qs_share * df["expected_payout_gross"]
    df["retained_after_qs"] = df["expected_payout_gross"] - df["ceded_claims_qs"]

    excess = np.maximum(df["retained_after_qs"] - config.xol_retention, 0.0)
    df["recovery_xol"] = np.minimum(excess, config.xol_limit)
    df["retained_claims_net"] = df["retained_after_qs"] - df["recovery_xol"]

    # Reinstatement Premium: Cost to restore the limit after usage.
    # Formula: (Recovery / Limit) * Original Premium
    df["reinstatement_premium"] = (df["recovery_xol"] / config.xol_limit) * config.xol_premium

    df["ISR_gross"] = df["insurance_service_result"]
    df["ISR_net"] = (
        df["revenue_recognized"]
        - df["retained_claims_net"]
        - df["acq_cost_amortized"]
        - df["ra_release"]
        - df["reinstatement_premium"]
    )

    return df


def measure_metrics(df: pd.DataFrame, paa_df: pd.DataFrame) -> Dict[str, float]:
    """Compute summary metrics from the net dataframe (with reinsurance applied)."""

    metrics = {
        "ISR_std_gross": float(paa_df["insurance_service_result"].std(ddof=0)),
        "ISR_std_net": float(df["ISR_net"].std(ddof=0)),
        "peak_claim_gross": float(df["expected_payout_gross"].max()),
        "peak_claim_net": float(df["retained_claims_net"].max()),
        "annual_isr_gross": float(paa_df["insurance_service_result"].sum()),
        "annual_isr_net": float(df["ISR_net"].sum()),
    }
    return metrics


def run_case(
    params: Phase7Params,
    payouts: Sequence[float],
    re_config: ReinsuranceConfig = BASE_REINSURANCE,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Run the PAA engine + reinsurance overlay and collect metrics."""

    paa = run_paa_projection(params, payouts)
    net = apply_reinsurance(paa, re_config)
    metrics = measure_metrics(net, paa)
    metrics["onerous_flag"] = bool(paa.attrs["onerous_at_inception"])
    return paa, net, metrics


def clone_params(params: Phase7Params, **changes) -> Phase7Params:
    """Create a modified copy of Phase7Params."""

    return Phase7Params(
        gwp=changes.get("gwp", params.gwp),
        acq_cost=changes.get("acq_cost", params.acq_cost),
        discount_rate_pa=changes.get("discount_rate_pa", params.discount_rate_pa),
        ra_margin=changes.get("ra_margin", params.ra_margin),
        coverage_units=changes.get("coverage_units", params.coverage_units),
    )


def ensure_array(payouts: Sequence[float]) -> np.ndarray:
    return np.asarray(payouts, dtype=float).copy()


# ---------------------------------------------------------------------------
# Shock definitions
# ---------------------------------------------------------------------------

Mutator = Callable[
    [Phase7Params, np.ndarray, ReinsuranceConfig],
    Tuple[Phase7Params, np.ndarray, ReinsuranceConfig],
]


def mutator_elr(delta_pp: float) -> Mutator:
    target_elr = max(0.0, BASE_ELR + delta_pp)
    scale = target_elr / BASE_ELR if BASE_ELR else 1.0

    def mutate(params: Phase7Params, payouts: np.ndarray, rei: ReinsuranceConfig):
        return params, payouts * scale, rei

    return mutate


def mutator_acq(delta_pp: float) -> Mutator:
    new_rate = max(0.0, BASE_ACQ_RATE + delta_pp)
    new_acq = new_rate * BASE_PARAMS.gwp

    def mutate(params: Phase7Params, payouts: np.ndarray, rei: ReinsuranceConfig):
        return clone_params(params, acq_cost=new_acq), payouts, rei

    return mutate


def mutator_frequency(scale: float) -> Mutator:
    """Add/remove events by shifting payouts into zero months."""

    def mutate(params: Phase7Params, payouts: np.ndarray, rei: ReinsuranceConfig):
        payouts = ensure_array(payouts)
        positive_idx = np.where(payouts > 0)[0]
        zero_idx = np.where(payouts == 0)[0]
        if scale > 1.0 and zero_idx.size > 0:
            add_events = max(1, int(math.ceil((scale - 1.0) * positive_idx.size)))
            avg_loss = payouts[positive_idx].mean() if positive_idx.size else 0.0
            for idx in zero_idx[:add_events]:       
                payouts[idx] = avg_loss * 0.6  # lighter events
        elif scale < 1.0 and positive_idx.size > 0:
            remove_events = max(1, int(math.ceil((1.0 - scale) * positive_idx.size)))
            smallest = positive_idx[np.argsort(payouts[positive_idx])[:remove_events]]
            payouts[smallest] = 0.0
        return params, payouts, rei

    return mutate


def mutator_severity_mu(scale: float) -> Mutator:
    """Scale existing losses (all positive months) by factor."""

    def mutate(params: Phase7Params, payouts: np.ndarray, rei: ReinsuranceConfig):
        payouts = ensure_array(payouts)
        payouts[payouts > 0] *= scale
        return params, payouts, rei

    return mutate


def mutator_severity_sigma(scale: float) -> Mutator:
    """Change tail thickness: boost top half, trim bottom half, keep total."""

    def mutate(params: Phase7Params, payouts: np.ndarray, rei: ReinsuranceConfig):
        payouts = ensure_array(payouts)
        pos_idx = np.where(payouts > 0)[0]
        if pos_idx.size <= 1:
            return params, payouts, rei
        sorted_idx = pos_idx[np.argsort(payouts[pos_idx])]
        half = len(sorted_idx) // 2
        bottom = sorted_idx[:half]
        top = sorted_idx[half:]
        if bottom.size:
            payouts[bottom] *= max(0.1, 1.0 - 0.2 * scale)
        payouts[top] *= 1.0 + 0.2 * scale
        desired_sum = BASE_PAYOUTS.sum() * (payouts.sum() / BASE_PAYOUTS.sum())
        if payouts.sum() > 0:
            payouts *= desired_sum / payouts.sum()
        return params, payouts, rei

    return mutate


def mutator_discount(delta_bps: int) -> Mutator:
    new_rate = max(0.0, BASE_PARAMS.discount_rate_pa + delta_bps / 10_000)

    def mutate(params: Phase7Params, payouts: np.ndarray, rei: ReinsuranceConfig):
        return clone_params(params, discount_rate_pa=new_rate), payouts, rei

    return mutate


def mutator_retention(delta: float) -> Mutator:
    new_retention = max(0.0, BASE_REINSURANCE.xol_retention + delta)

    def mutate(params: Phase7Params, payouts: np.ndarray, rei: ReinsuranceConfig):
        return params, payouts, ReinsuranceConfig(
            qs_share=rei.qs_share,
            xol_retention=new_retention,
            xol_limit=rei.xol_limit,
        )

    return mutate


def mutator_flat_units() -> Mutator:
    def mutate(params: Phase7Params, payouts: np.ndarray, rei: ReinsuranceConfig):
        return clone_params(params, coverage_units=(1,) * 12), payouts, rei

    return mutate


# Shock registry: assumption -> (label, mutate_up, mutate_down)
SHOCKS: List[Dict[str, object]] = [
    {
        "key": "ELR +/-10pp",
        "label": "ELR +/-10pp",
        "mutate_up": mutator_elr(+0.10),
        "mutate_down": mutator_elr(-0.10),
    },
    {
        "key": "Acquisition +/-5pp",
        "label": "Acquisition cost +/-5pp",
        "mutate_up": mutator_acq(+0.05),
        "mutate_down": mutator_acq(-0.05),
    },
    {
        "key": "Frequency +/-20%",
        "label": "Frequency lambda +/-20%",
        "mutate_up": mutator_frequency(1.20),
        "mutate_down": mutator_frequency(0.80),
    },
    {
        "key": "Severity mu +/-20%",
        "label": "Severity mu +/-20%",
        "mutate_up": mutator_severity_mu(1.20),
        "mutate_down": mutator_severity_mu(0.80),
    },
    {
        "key": "Severity sigma +/-20%",
        "label": "Severity sigma +/-20%",
        "mutate_up": mutator_severity_sigma(1.20),
        "mutate_down": mutator_severity_sigma(0.80),
    },
    {
        "key": "Discount +/-100bps",
        "label": "Discount rate +/-100 bps",
        "mutate_up": mutator_discount(+100),
        "mutate_down": mutator_discount(-100),
    },
    {
        "key": "Retention +/-0.5B",
        "label": "XoL retention +/-0.5B",
        "mutate_up": mutator_retention(+500_000_000),
        "mutate_down": mutator_retention(-500_000_000),
    },
    {
        "key": "Coverage units flat",
        "label": "Coverage units -> flat",
        "mutate_up": mutator_flat_units(),
        "mutate_down": mutator_flat_units(),
    },
]


# ---------------------------------------------------------------------------
# Sensitivity sweeps
# ---------------------------------------------------------------------------

def tornado_analysis(
    params: Phase7Params,
    payouts: np.ndarray,
    rei: ReinsuranceConfig,
) -> pd.DataFrame:
    """Compute single-factor sensitivity deltas for ISR std and peak retained."""

    _, _, base_metrics = run_case(params, payouts, rei)

    rows = []
    for entry in SHOCKS:
        mutate_up: Mutator = entry["mutate_up"]  # type: ignore[assignment]
        mutate_down: Mutator = entry["mutate_down"]  # type: ignore[assignment]

        p_up, c_up, r_up = mutate_up(params, payouts, rei)
        _, _, metrics_up = run_case(p_up, c_up, r_up)

        p_dn, c_dn, r_dn = mutate_down(params, payouts, rei)
        _, _, metrics_dn = run_case(p_dn, c_dn, r_dn)

        rows.append(
            {
                "assumption": entry["label"],
                "delta_ISR_std_up": metrics_up["ISR_std_net"] - base_metrics["ISR_std_net"],
                "delta_ISR_std_down": metrics_dn["ISR_std_net"] - base_metrics["ISR_std_net"],
                "delta_peak_up": metrics_up["peak_claim_net"] - base_metrics["peak_claim_net"],
                "delta_peak_down": metrics_dn["peak_claim_net"] - base_metrics["peak_claim_net"],
            }
        )

    df = pd.DataFrame(rows)
    df["max_abs_ISR_std"] = df[["delta_ISR_std_up", "delta_ISR_std_down"]].abs().max(axis=1)
    df["max_abs_peak"] = df[["delta_peak_up", "delta_peak_down"]].abs().max(axis=1)
    return df.sort_values("max_abs_ISR_std", ascending=False).reset_index(drop=True)


def heatmap_elr_acq(
    params: Phase7Params,
    payouts: np.ndarray,
    rei: ReinsuranceConfig,
    elr_range: Iterable[float],
    acq_range: Iterable[float],
) -> pd.DataFrame:
    """Build a grid over ELR × acquisition rate, capturing key metrics."""

    records = []
    for elr in elr_range:
        scale = elr / BASE_ELR if BASE_ELR else 1.0
        scaled_payouts = payouts * scale
        for acq in acq_range:
            new_params = clone_params(params, acq_cost=acq * params.gwp)
            _, _, metrics = run_case(new_params, scaled_payouts, rei)
            records.append(
                {
                    "ELR": elr,
                    "acq_rate": acq,
                    "ISR_std_net": metrics["ISR_std_net"],
                    "peak_claim_net": metrics["peak_claim_net"],
                    "annual_isr_net": metrics["annual_isr_net"],
                    "onerous": metrics["onerous_flag"],
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def format_idr(value: float) -> str:
    return f"{value:,.0f}"


def format_billion(value: float) -> str:
    return f"{value / 1e9:,.3f}"


def print_baseline(metrics: Mapping[str, float]) -> None:
    print("1. BASELINE METRICS (NET OF REINSURANCE)")
    print("----------------------------------------")
    print(f"{'Metric':<25} | {'Value':>15}")
    print("-" * 43)
    print(f"{'Onerous Flag':<25} | {'Yes' if metrics['onerous_flag'] else 'No':>15}")
    print(f"{'Annual ISR (Net)':<25} | {format_idr(metrics['annual_isr_net']):>15} IDR")
    print(f"{'Annual ISR (Gross)':<25} | {format_idr(metrics['annual_isr_gross']):>15} IDR")
    print(f"{'ISR Std Dev (Net)':<25} | {format_idr(metrics['ISR_std_net']):>15} IDR")
    print(f"{'Peak Retained Claim':<25} | {format_idr(metrics['peak_claim_net']):>15} IDR")
    print("-" * 43)
    print()


def print_tornado(df: pd.DataFrame) -> None:
    view = df.copy()
    view["dISR_std_up"] = view["delta_ISR_std_up"].map(format_idr)
    view["dISR_std_dn"] = view["delta_ISR_std_down"].map(format_idr)
    
    print("2. SENSITIVITY ANALYSIS (TORNADO)")
    print("---------------------------------")
    # Simplified columns for clean terminal output
    cols = ["assumption", "dISR_std_up", "dISR_std_dn"]
    print(view[cols].to_string(index=False, justify="left", col_space=15))
    print()


def print_heatmap_summary(df: pd.DataFrame) -> None:
    pivot_onerous = df.pivot_table(
        values="onerous",
        index="acq_rate",
        columns="ELR",
        aggfunc="max",
    )
    print("3. ONEROUS BOUNDARY MAP (1 = ONEROUS)")
    print("-------------------------------------")
    print(pivot_onerous.astype(int).to_string())
    print()

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path("phase9_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline
    baseline_paa, baseline_net, baseline_metrics = run_case(BASE_PARAMS, BASE_PAYOUTS, BASE_REINSURANCE)
    print_baseline(baseline_metrics)

    # Tornado
    tornado_df = tornado_analysis(BASE_PARAMS, BASE_PAYOUTS, BASE_REINSURANCE)
    tornado_df.to_csv(out_dir / "tornado.csv", index=False)
    print_tornado(tornado_df)

    # Heatmap (ELR 50% - 110%, Acquisition 0.30 - 0.45)
    elr_levels = np.linspace(0.50, 1.10, 7)
    acq_levels = np.linspace(0.30, 0.45, 7)
    heatmap_df = heatmap_elr_acq(BASE_PARAMS, BASE_PAYOUTS, BASE_REINSURANCE, elr_levels, acq_levels)
    heatmap_df.to_csv(out_dir / "heatmap.csv", index=False)
    print_heatmap_summary(heatmap_df)

    # Coverage-unit robustness (reuse Phase 7 helper)
    robust_stats, mean_abs = coverage_unit_robustness(BASE_PARAMS, BASE_PAYOUTS)
    print("4. COVERAGE UNIT DIAGNOSTICS")
    print("----------------------------")
    print(f"Comparison: Flat (1/12) vs Seasonal Units")
    print(f"Max ISR Diff:  {format_idr(robust_stats['max_abs_diff']):>15} IDR")
    print(f"Mean ISR Diff: {format_idr(mean_abs):>15} IDR")
    print(f"Correlation:   {robust_stats['corr']:>15.4f}")
    print("----------------------------")
    print()

    # Save baseline monthly table for reference
    baseline_net.to_csv(out_dir / "baseline_monthly_net.csv", index=False)
    baseline_paa.to_csv(out_dir / "baseline_monthly_gross.csv", index=False)


if __name__ == "__main__":
    main()
