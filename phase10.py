"""
Phase 10 – Financial Summary & Capital View.

Generates gross vs net management summaries, profit bridges, and volatility
metrics for the Base, Clash, and High-Acquisition scenarios. Outputs are saved
under `phase10_results/` and printed in an executive-friendly format.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from phases import (
    Phase7Params,
    run_paa_projection,
    SCENARIO_SPECS,
)


# ---------------------------------------------------------------------------
# Reinsurance configuration (same as Phase 4/9)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReinsuranceConfig:
    qs_share: float
    xol_retention: float
    xol_limit: float
    xol_premium: float = 500_000_000  # Assumed XoL treaty cost for reinstatement base


BASE_REINSURANCE = ReinsuranceConfig(qs_share=0.50, xol_retention=1_500_000_000, xol_limit=5_000_000_000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_reinsurance(paa_df: pd.DataFrame, config: ReinsuranceConfig) -> pd.DataFrame:
    """Augment the PAA dataframe with reinsurance-adjusted cashflows."""

    df = paa_df.copy()

    df["ceded_claims_qs"] = config.qs_share * df["expected_payout_gross"]
    df["retained_after_qs"] = df["expected_payout_gross"] - df["ceded_claims_qs"]

    excess = np.maximum(df["retained_after_qs"] - config.xol_retention, 0.0)
    df["recovery_xol"] = np.minimum(excess, config.xol_limit)
    df["retained_claims_net"] = df["retained_after_qs"] - df["recovery_xol"]

    # Reinstatement Premium: Cost to restore the limit after usage.
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


def scenario_params(name: str) -> Tuple[Phase7Params, np.ndarray]:
    params, payouts = SCENARIO_SPECS[name]
    return params, np.asarray(payouts, dtype=float)


def acq_rate(params: Phase7Params) -> float:
    return params.acq_cost / params.gwp if params.gwp else 0.0


def format_idr(value: float) -> str:
    return f"{value:,.0f}"


def format_pct(value: float) -> str:
    return f"{value:.1f}%" if np.isfinite(value) else "n/a"


def profit_bridge(ep_net: float, acq_total: float, net_claims: float, net_isr: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"step": "Net Earned Premium", "amount": ep_net},
            {"step": "Acquisition Cost", "amount": -acq_total},
            {"step": "Net Claims", "amount": -net_claims},
            {"step": "Net ISR", "amount": net_isr},
        ]
    )


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def summarize_financials(
    scenario_name: str,
    params: Phase7Params,
    payouts: np.ndarray,
    reins: ReinsuranceConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Return gross PAA df, net df, and headline metrics."""

    paa_df = run_paa_projection(params, payouts)
    net_df = apply_reinsurance(paa_df, reins)

    gwp = params.gwp
    ceded_premium = gwp * reins.qs_share
    nwp = gwp - ceded_premium

    ep_gross = paa_df["revenue_recognized"].sum()
    ep_net = ep_gross * (1 - reins.qs_share)

    acq_total = params.acq_cost

    gross_claims = paa_df["expected_payout_gross"].sum()
    net_claims = net_df["retained_claims_net"].sum()

    ra_release_total = paa_df["ra_release"].sum()

    isr_gross_sum = paa_df["insurance_service_result"].sum()
    isr_net_sum = net_df["ISR_net"].sum()

    isr_gross_std = paa_df["insurance_service_result"].std(ddof=0)
    isr_net_std = net_df["ISR_net"].std(ddof=0)

    peak_claim_gross = paa_df["expected_payout_gross"].max()
    peak_claim_net = net_df["retained_claims_net"].max()

    worst_month_net = net_df.loc[net_df["ISR_net"].idxmin(), ["month", "ISR_net"]]

    vol_reduction = 1 - (isr_net_std / isr_gross_std) if isr_gross_std > 0 else np.nan
    peak_reduction = 1 - (peak_claim_net / peak_claim_gross) if peak_claim_gross > 0 else np.nan

    metrics = {
        "scenario": scenario_name.title(),
        "GWP": gwp,
        "Ceded_Premium": ceded_premium,
        "NWP": nwp,
        "EP_gross": ep_gross,
        "EP_net": ep_net,
        "Acq_total": acq_total,
        "Claims_gross": gross_claims,
        "Claims_net": net_claims,
        "RA_release": ra_release_total,
        "ISR_gross_sum": isr_gross_sum,
        "ISR_net_sum": isr_net_sum,
        "ISR_gross_std": isr_gross_std,
        "ISR_net_std": isr_net_std,
        "Peak_claim_gross": peak_claim_gross,
        "Peak_claim_net": peak_claim_net,
        "Volatility_reduction_pct": vol_reduction * 100,
        "Peak_claim_reduction_pct": peak_reduction * 100,
        "Worst_month": int(worst_month_net["month"]),
        "Worst_month_ISR_net": float(worst_month_net["ISR_net"]),
    }

    return paa_df, net_df, metrics


def management_table(metrics: Mapping[str, float]) -> pd.DataFrame:
    """Construct the management table for a single scenario."""

    rows = [
        ("Written Premium", metrics["GWP"], metrics["NWP"], "50% quota share"),
        ("Earned Premium", metrics["EP_gross"], metrics["EP_net"], "PAA coverage units"),
        ("Acquisition Cost (total)", metrics["Acq_total"], metrics["Acq_total"], "Assume not ceded"),
        ("Claims (total)", metrics["Claims_gross"], metrics["Claims_net"], "After QS + Cat-XoL"),
        ("Risk Adjustment release", metrics["RA_release"], metrics["RA_release"], ""),
        ("Insurance Service Result", metrics["ISR_gross_sum"], metrics["ISR_net_sum"], "Headline"),
        ("ISR std (monthly)", metrics["ISR_gross_std"], metrics["ISR_net_std"], "Volatility"),
        ("Peak monthly claim", metrics["Peak_claim_gross"], metrics["Peak_claim_net"], "Catastrophe cap"),
        ("Volatility reduction", np.nan, metrics["Volatility_reduction_pct"], "% vs gross"),
        ("Peak claim reduction", np.nan, metrics["Peak_claim_reduction_pct"], "% vs gross"),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Gross_IDR", "Net_IDR", "Notes"])
    return df


# ---------------------------------------------------------------------------
# Output utilities
# ---------------------------------------------------------------------------

def write_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def print_table(df: pd.DataFrame) -> None:
    view = df.copy()
    formatted_rows = []
    for _, row in view.iterrows():
        metric = row["Metric"]
        gross = row["Gross_IDR"]
        net = row["Net_IDR"]
        if np.isfinite(gross):
            gross_str = format_idr(gross)
        else:
            gross_str = "n/a"
        if "reduction" in metric.lower() and np.isfinite(net):
            net_str = format_pct(net)
        else:
            net_str = format_idr(net) if np.isfinite(net) else "n/a"
        formatted_rows.append((metric, gross_str, net_str, row["Notes"]))
    display_df = pd.DataFrame(formatted_rows, columns=view.columns)
    # Clean output
    print(display_df.to_string(index=False, justify="left", col_space=15))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path("phase10_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ["base", "clash", "high_acq"]
    summary_rows: List[Dict[str, float]] = []

    print("PHASE 10: FINANCIAL SUMMARY & CAPITAL VIEW")
    print("==========================================")
    print()

    for i, name in enumerate(scenarios, 1):
        params, payouts = scenario_params(name)
        paa_df, net_df, metrics = summarize_financials(name, params, payouts, BASE_REINSURANCE)

        summary_rows.append(metrics)

        table = management_table(metrics)
        bridge = profit_bridge(metrics["EP_net"], metrics["Acq_total"], metrics["Claims_net"], metrics["ISR_net_sum"])

        paa_df.to_csv(out_dir / f"{name}_monthly_gross.csv", index=False)
        net_df.to_csv(out_dir / f"{name}_monthly_net.csv", index=False)
        table.to_csv(out_dir / f"{name}_management_table.csv", index=False)
        bridge.to_csv(out_dir / f"{name}_profit_bridge.csv", index=False)

        print(f"{i}. FINANCIAL SUMMARY: {metrics['scenario'].upper()}")
        print("-" * 40)
        print_table(table)
        
        print(f"   Profit Bridge (Net):")
        print(bridge.to_string(index=False, justify="left"))
        print()
        print(
            f"   Worst Net ISR Month: Month {metrics['Worst_month']} "
            f"({format_idr(metrics['Worst_month_ISR_net'])} IDR)"
        )
        print()
        print("-" * 72)
        print()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "scenario_summary.csv", index=False)

    print("4. SCENARIO COMPARISON SUMMARY")
    print("------------------------------")
    display_cols = [
        "scenario",
        "ISR_net_sum",
        "ISR_net_std",
        "Peak_claim_net",
        "Volatility_reduction_pct",
        "Peak_claim_reduction_pct",
    ]
    view = summary_df[display_cols].copy()
    view["ISR_net_sum"] = view["ISR_net_sum"].map(format_idr)
    view["ISR_net_std"] = view["ISR_net_std"].map(format_idr)
    view["Peak_claim_net"] = view["Peak_claim_net"].map(format_idr)
    view["Volatility_reduction_pct"] = view["Volatility_reduction_pct"].map(format_pct)
    view["Peak_claim_reduction_pct"] = view["Peak_claim_reduction_pct"].map(format_pct)
    print(view.to_string(index=False, justify="left", col_space=12))
    print()


if __name__ == "__main__":
    main()
