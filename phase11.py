"""
Phase 11 – Interpretation & Management Levers.

Builds a playbook that maps levers (pricing, acquisition, reinsurance tweaks)
to KPI deltas for both Base and Clash scenarios. Outputs CSV tables and prints
ranked summaries plus trigger thresholds. Reuses measurement logic from
Phase 10 and shocks from earlier phases.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from phases import Phase7Params, run_paa_projection, SCENARIO_SPECS
from phase10 import apply_reinsurance, ReinsuranceConfig, BASE_REINSURANCE


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCENARIOS = ["base", "clash"]
OUT_DIR = Path("phase11_results")

BASE_PARAMS = SCENARIO_SPECS["base"][0]

BASE_METRICS_KEYS = [
    "ISR_net_sum",
    "ISR_net_std",
    "Peak_claim_net",
    "NWP",
    "EP_net",
    "Acq_total",
    "onerous_flag",
]


# ---------------------------------------------------------------------------
# Helpers (reuse from Phase 10 with slight tweaks)
# ---------------------------------------------------------------------------

def scenario_params(name: str) -> Tuple[Phase7Params, np.ndarray]:
    params, payouts = SCENARIO_SPECS[name]
    return params, np.asarray(payouts, dtype=float)


def summarize_metrics(
    params: Phase7Params,
    payouts: np.ndarray,
    reins: ReinsuranceConfig,
) -> Dict[str, float]:
    paa_df = run_paa_projection(params, payouts)
    net_df = apply_reinsurance(paa_df, reins)

    gwp = params.gwp
    ceded_prem = gwp * reins.qs_share
    nwp = gwp - ceded_prem
    ep_gross = paa_df["revenue_recognized"].sum()
    ep_net = ep_gross * (1 - reins.qs_share)
    acq_total = params.acq_cost

    metrics = {
        "ISR_net_sum": net_df["ISR_net"].sum(),
        "ISR_net_std": net_df["ISR_net"].std(ddof=0),
        "Peak_claim_net": net_df["retained_claims_net"].max(),
        "NWP": nwp,
        "EP_net": ep_net,
        "Acq_total": acq_total,
        "onerous_flag": bool(paa_df.attrs["onerous_at_inception"]),
    }
    return metrics


def format_idr(value: float) -> str:
    return f"{value:,.0f}"


def format_pct(value: float) -> str:
    return f"{value:.1f}%" if np.isfinite(value) else "n/a"


# ---------------------------------------------------------------------------
# Lever definitions
# ---------------------------------------------------------------------------

LeverSetting = Tuple[str, str, Callable[[Phase7Params, ReinsuranceConfig], Tuple[Phase7Params, ReinsuranceConfig]]]


def set_premium(scale: float) -> Callable[[Phase7Params, ReinsuranceConfig], Tuple[Phase7Params, ReinsuranceConfig]]:
    def mutate(params: Phase7Params, reins: ReinsuranceConfig):
        new_gwp = params.gwp * scale
        acq_rate = params.acq_cost / params.gwp if params.gwp else 0.0
        return Phase7Params(
            gwp=new_gwp,
            acq_cost=new_gwp * acq_rate,
            discount_rate_pa=params.discount_rate_pa,
            ra_margin=params.ra_margin,
            coverage_units=params.coverage_units,
        ), reins

    return mutate


def set_acquisition_rate(delta_pp: float):
    def mutate(params: Phase7Params, reins: ReinsuranceConfig):
        rate = params.acq_cost / params.gwp if params.gwp else 0.0
        new_rate = max(0.0, rate + delta_pp)
        return Phase7Params(
            gwp=params.gwp,
            acq_cost=params.gwp * new_rate,
            discount_rate_pa=params.discount_rate_pa,
            ra_margin=params.ra_margin,
            coverage_units=params.coverage_units,
        ), reins

    return mutate


def set_qs_share(new_share: float):
    def mutate(params: Phase7Params, reins: ReinsuranceConfig):
        return params, ReinsuranceConfig(
            qs_share=new_share,
            xol_retention=reins.xol_retention,
            xol_limit=reins.xol_limit,
        )

    return mutate


def set_retention(new_retention: float):
    def mutate(params: Phase7Params, reins: ReinsuranceConfig):
        return params, ReinsuranceConfig(
            qs_share=reins.qs_share,
            xol_retention=new_retention,
            xol_limit=reins.xol_limit,
        )

    return mutate


def set_limit(new_limit: float):
    def mutate(params: Phase7Params, reins: ReinsuranceConfig):
        return params, ReinsuranceConfig(
            qs_share=reins.qs_share,
            xol_retention=reins.xol_retention,
            xol_limit=new_limit,
        )

    return mutate


def set_flat_units():
    def mutate(params: Phase7Params, reins: ReinsuranceConfig):
        return Phase7Params(
            gwp=params.gwp,
            acq_cost=params.acq_cost,
            discount_rate_pa=params.discount_rate_pa,
            ra_margin=params.ra_margin,
            coverage_units=(1,) * 12,
        ), reins

    return mutate


LEVER_SETTINGS: List[LeverSetting] = [
    ("Premium", "+5%", set_premium(1.05)),
    ("Premium", "-5%", set_premium(0.95)),
    ("Acquisition cost", "-5 pp", set_acquisition_rate(-0.05)),
    ("Acquisition cost", "+5 pp", set_acquisition_rate(+0.05)),
    ("QS cession", "60%", set_qs_share(0.60)),
    ("QS cession", "40%", set_qs_share(0.40)),
    ("Cat-XoL retention", "1.0B", set_retention(1_000_000_000)),
    ("Cat-XoL retention", "2.0B", set_retention(2_000_000_000)),
    ("Cat-XoL limit", "+2B", set_limit(BASE_REINSURANCE.xol_limit + 2_000_000_000)),
    ("Coverage units", "Flat 1/12", set_flat_units()),
]


# ---------------------------------------------------------------------------
# Lever sweep
# ---------------------------------------------------------------------------

def lever_sweep() -> pd.DataFrame:
    base_outputs: Dict[str, Dict[str, float]] = {}
    for scenario in SCENARIOS:
        params, payouts = scenario_params(scenario)
        base_outputs[scenario] = summarize_metrics(params, payouts, BASE_REINSURANCE)

    rows = []
    for lever, setting, mutator in LEVER_SETTINGS:
        for scenario in SCENARIOS:
            params, payouts = scenario_params(scenario)
            m_params, m_reins = mutator(params, BASE_REINSURANCE)
            metrics = summarize_metrics(m_params, payouts, m_reins)
            base = base_outputs[scenario]

            rows.append(
                {
                    "lever": lever,
                    "setting": setting,
                    "scenario": scenario.title(),
                    "delta_ISR_net_sum": metrics["ISR_net_sum"] - base["ISR_net_sum"],
                    "delta_ISR_net_std": metrics["ISR_net_std"] - base["ISR_net_std"],
                    "delta_peak_claim": metrics["Peak_claim_net"] - base["Peak_claim_net"],
                    "ISR_net_sum": metrics["ISR_net_sum"],
                    "ISR_net_std": metrics["ISR_net_std"],
                    "Peak_claim_net": metrics["Peak_claim_net"],
                    "onerous_flag": metrics["onerous_flag"],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trigger matrix (based on Phase 9 heatmap thresholds)
# ---------------------------------------------------------------------------

TRIGGER_MATRIX = [
    {
        "Trigger": "Acquisition rate >= 40%",
        "Threshold": "0.40",
        "Action": "Cut channel commission by 3-5 pp",
        "Expected Effect": "+IDR 1.35B profit (Base), volatility unchanged",
        "Owner": "Distribution",
    },
    {
        "Trigger": "Severity scale +20%",
        "Threshold": "+20%",
        "Action": "Lower Cat-XoL retention to 1.0B",
        "Expected Effect": "-vol ~120M, -peak 500M, -profit ~1.1B",
        "Owner": "Reinsurance",
    },
    {
        "Trigger": "ISR volatility > 650M (monthly)",
        "Threshold": "650,000,000",
        "Action": "Increase QS to 60%",
        "Expected Effect": "-vol ~160M, -peak 0 (still 1.5B), -profit ~900M",
        "Owner": "CUO",
    },
    {
        "Trigger": "ELR x ACQ near red zone",
        "Threshold": "ELR >= 0.70 & ACQ >= 0.35",
        "Action": "Price +3% or ACQ -3pp",
        "Expected Effect": "Keeps day-one margin positive",
        "Owner": "Pricing",
    },
]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def format_lever_table(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    view["delta_ISR_net_sum"] = view["delta_ISR_net_sum"].map(format_idr)
    view["delta_ISR_net_std"] = view["delta_ISR_net_std"].map(format_idr)
    view["delta_peak_claim"] = view["delta_peak_claim"].map(format_idr)
    view["onerous_flag"] = view["onerous_flag"].map({True: "Yes", False: "No"})
    return view


def ranked_table(df: pd.DataFrame) -> pd.DataFrame:
    base_df = df[df["scenario"] == "Base"].copy()
    base_df["rank"] = base_df["delta_ISR_net_std"].abs().rank(ascending=False, method="dense").astype(int)
    return base_df.sort_values("rank")


def print_lever_summary(df: pd.DataFrame) -> None:
    print("1. LEVER IMPACT ANALYSIS (BASE SCENARIO)")
    print("----------------------------------------")
    formatted = ranked_table(df)
    rd = pd.DataFrame(
        {
            "Rank": formatted["rank"],
            "Lever": formatted["lever"],
            "Setting": formatted["setting"],
            "Delta Profit": formatted["delta_ISR_net_sum"],
            "Delta Vol": formatted["delta_ISR_net_std"],
            "Delta Peak": formatted["delta_peak_claim"],
            "Onerous?": formatted["onerous_flag"],
        }
    )
    print(rd.to_string(index=False, justify="left", col_space=12))
    print()

    print("2. LEVER IMPACT ANALYSIS (CLASH SCENARIO)")
    print("-----------------------------------------")
    clash = df[df["scenario"] == "Clash"].copy()
    clash["delta_ISR_net_sum"] = clash["delta_ISR_net_sum"].map(format_idr)
    clash["delta_ISR_net_std"] = clash["delta_ISR_net_std"].map(format_idr)
    clash["delta_peak_claim"] = clash["delta_peak_claim"].map(format_idr)
    clash["onerous_flag"] = clash["onerous_flag"].map({True: "Yes", False: "No"})
    
    # Select and rename columns for display
    clash_disp = clash[["lever", "setting", "delta_ISR_net_sum", "delta_ISR_net_std", "delta_peak_claim", "onerous_flag"]]
    clash_disp.columns = ["Lever", "Setting", "Delta Profit", "Delta Vol", "Delta Peak", "Onerous?"]
    
    print(clash_disp.to_string(index=False, justify="left", col_space=12))
    print()


def print_trigger_matrix() -> None:
    print("3. DYNAMIC TRIGGER MATRIX")
    print("-------------------------")
    df = pd.DataFrame(TRIGGER_MATRIX)
    print(df.to_string(index=False, justify="left", col_space=15))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_df = lever_sweep()
    sweep_df.to_csv(OUT_DIR / "lever_impacts.csv", index=False)

    print("PHASE 11: INTERPRETATION & MANAGEMENT LEVERS")
    print("============================================")
    print()

    print_lever_summary(sweep_df)
    print_trigger_matrix()

    # Save formatted tables for the report
    ranked_table(sweep_df).to_csv(OUT_DIR / "table11A_base.csv", index=False)
    sweep_df[sweep_df["scenario"] == "Clash"].to_csv(OUT_DIR / "table11A_clash.csv", index=False)
    pd.DataFrame(TRIGGER_MATRIX).to_csv(OUT_DIR / "table11B_triggers.csv", index=False)

    print("4. MANAGEMENT ACTIONS NARRATIVE")
    print("-------------------------------")
    print("Based on the sensitivity sweep, the following actions are recommended:")
    print("  * ACQUISITION COST: High sensitivity. Reducing by 5pp significantly boosts profit.")
    print("  * REINSURANCE: Lowering retention reduces volatility but at a high profit cost.")
    print("  * PRICING: Increasing premiums is effective but may impact competitiveness.")
    print("============================================")
    print()


if __name__ == "__main__":
    main()