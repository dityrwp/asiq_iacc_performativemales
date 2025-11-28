import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from phases import (
    _base_params, 
    run_gmm_lite, 
    get_loss_component_schedule, 
    coverage_unit_robustness, 
    run_paa_projection,
    SCENARIO_SPECS,
    Phase7Params,
    SEASONAL_UNITS,
    FLAT_UNITS
)
from phase9 import (
    run_case, 
    BASE_REINSURANCE, 
    ReinsuranceConfig
)

def main():
    print("=== Phase 1: Solvency Reality Check (Onerous Test) ===")
    params = _base_params()
    # Base scenario payouts
    payouts = SCENARIO_SPECS["base"][1]
    
    gmm_df = run_gmm_lite(params, payouts)
    
    print(f"GWP: {params.gwp:,.0f}")
    print(f"Acquisition Cost: {params.acq_cost:,.0f} ({params.acq_cost/params.gwp:.1%})")
    print(f"PV Inflows: {gmm_df.attrs['pv_inflows']:,.0f}")
    print(f"PV Outflows: {gmm_df.attrs['pv_outflows']:,.0f}")
    print(f"Loss Component Initial: {gmm_df.attrs['loss_component']:,.0f}")
    
    if gmm_df.attrs['loss_component'] > 0:
        print("\n[SUCCESS] Portfolio is Onerous as planned.")
        print("\nLoss Component Amortization Table:")
        lc_table = get_loss_component_schedule(gmm_df)
        print(lc_table.to_string(index=False))
    else:
        print("\n[WARNING] Portfolio is NOT Onerous. Adjust assumptions.")

    print("\n" + "="*50 + "\n")

    print("=== Phase 2: Smoothness Upgrade (Coverage Units) ===")
    # Compare Flat (New) vs Seasonal (Old)
    # We want to plot the ISR of GMM (Seasonal) vs GMM (Flat) or PAA (Flat)
    # The user wants "Line A (Old Way): Spiky profit... Line B (New Way): Smooth"
    
    # Old Way: GMM with Seasonal Units
    old_params = Phase7Params(
        gwp=params.gwp,
        acq_cost=params.acq_cost,
        discount_rate_pa=params.discount_rate_pa,
        ra_margin=params.ra_margin,
        coverage_units=SEASONAL_UNITS
    )
    gmm_old = run_gmm_lite(old_params, payouts)
    
    # New Way: GMM with Flat Units (Passage of Time)
    # (params is already Flat)
    gmm_new = run_gmm_lite(params, payouts)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(gmm_old['month'], gmm_old['insurance_service_result_gmm'], label='Old Way (Claims-Based/Seasonal)', marker='o', linestyle='--')
    plt.plot(gmm_new['month'], gmm_new['insurance_service_result_gmm'], label='New Way (Time-Based/Flat)', marker='o', linewidth=2)
    plt.title('Profit Release: Volatile (Old) vs Stable (New)')
    plt.xlabel('Month')
    plt.ylabel('Insurance Service Result (IDR)')
    plt.legend()
    plt.grid(True)
    plt.savefig('smoothness_upgrade.png')
    print("Generated 'smoothness_upgrade.png'")
    
    print("\n" + "="*50 + "\n")

    print("=== Phase 3: Clash Strategy (Reinsurance) ===")
    # Clash Scenario
    clash_payouts = SCENARIO_SPECS["clash"][1]
    
    # Run with Reinsurance
    # Ensure we use the updated ReinsuranceConfig with premium
    # phase9.BASE_REINSURANCE already has the premium if I updated it correctly.
    
    paa_clash, net_clash, metrics_clash = run_case(params, clash_payouts, BASE_REINSURANCE)
    
    print("Clash Scenario Results:")
    print(f"Gross Payouts: {clash_payouts}")
    print("\nReinsurance Recovery & Reinstatement Premium:")
    cols = ['month', 'expected_payout_gross', 'recovery_xol', 'reinstatement_premium', 'ISR_net']
    print(net_clash[cols].to_string(index=False))
    
    print(f"\nTotal Reinstatement Premium Paid: {net_clash['reinstatement_premium'].sum():,.0f}")
    print(f"Net ISR (Profit/Loss): {metrics_clash['annual_isr_net']:,.0f}")

    print("\n" + "="*50 + "\n")

    print("=== Phase 4: Commercial Viability (Fixing the Product) ===")
    print("Strategy: Replace 'Dynamic Pricing' with 'Zone-Based Flat Pricing' for simplicity.")
    
    # Assumptions
    target_gwp = params.gwp # 17B
    assumed_policies = 100_000 # Assumption for calculation
    avg_premium = target_gwp / assumed_policies
    
    # Zone Relativities (Risk Based)
    zones = {
        "Green (Low Risk)": 0.8,
        "Yellow (Med Risk)": 1.0,
        "Red (High Risk)": 1.5
    }
    
    base_rate = avg_premium / (0.4 * 0.8 + 0.4 * 1.0 + 0.2 * 1.5) # Assume 40/40/20 mix
    
    print(f"\nTarget GWP: {target_gwp:,.0f}")
    print(f"Assumed Policy Count: {assumed_policies:,.0f}")
    print(f"Average Technical Premium: {avg_premium:,.0f} IDR")
    
    print("\nProposed Zone-Based Flat Premiums:")
    for zone, rel in zones.items():
        prem = base_rate * rel
        print(f"  - {zone}: {prem:,.0f} IDR")
        
    # Affordability Check
    # Benchmark: 5% of Monthly Income. 
    # Fisherman Income assumption: 3,000,000 IDR/month -> 150,000 IDR budget? 
    # Wait, if Premium is ~300k, that's high.
    
    avg_fisherman_income = 3_000_000 # Monthly
    affordability_threshold = avg_fisherman_income * 0.05 # 5%
    
    print(f"\nAffordability Check (Benchmark: 5% of {avg_fisherman_income:,.0f} = {affordability_threshold:,.0f}):")
    if base_rate > affordability_threshold:
        print(f"  [ALERT] Base Premium ({base_rate:,.0f}) exceeds threshold!")
        print("  Recommendation: Implement 'Aggregator Distribution' (Co-ops) to reduce acquisition costs further.")
        print("  Recommendation: Seek Government Subsidy for Red Zone.")
    else:
        print("  [OK] Premium is affordable.")

if __name__ == "__main__":
    main()
