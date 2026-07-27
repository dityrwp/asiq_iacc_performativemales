# Taming Volatility: A PSAK 117 Framework for a Parametric Catastrophe Cover

**3rd Place — International Actuarial Case Competition (IACC), Actuarial Science Quest (ASiQ), Universitas Gadjah Mada, 2025**
Team: performative males — Muhammad Aditya Rahmansyah Baskoro, Adzka Bagus Juniarta

An actuarial projection engine and governance framework built from scratch to assess whether a new parametric catastrophe insurance product is financially viable under PSAK 117 (Indonesia's adoption of IFRS 17), and to design the reinsurance and accounting policy needed to make it survive a bad year.

## The case

Tugu Andalas Insurance is launching a 12-month parametric catastrophe cover for MSMEs — a fixed lump-sum payout triggered by climate-related events, sold through high-cost digital and bancassurance channels. Two things put the product at risk: acquisition costs above 35% of gross written premium, and the possibility of a "clash season" — multiple severe events in the same year. Layered on top is the mandatory adoption of PSAK 117, which changes how insurance liabilities and profit are recognized and could expose the product's earnings volatility in a way management hadn't planned for.

The brief asked for five things: validate whether the simplified Premium Allocation Approach (PAA) is a legitimate stand-in for the full General Measurement Model (GMM); build a 12-month liability projection engine; run a day-one onerous-contract test; quantify what reinsurance actually buys the company; and turn all of it into accounting policy recommendations.

## Approach

The codebase is organized as a sequence of phases, each a standalone, runnable Python module built on shared dataclasses (`Phase7Params`, `ReinsuranceConfig`) rather than notebook cells — a deliberate choice so the model could be rerun and audited independently of any single analyst's environment.

**Measurement model validation (`phases.py`)** — implements both a PAA roll-forward and a simplified GMM projection (Liability for Remaining Coverage, Risk Adjustment, CSM/Loss Component) over three scenarios: a 60% loss-ratio Base case, a 100% loss-ratio Clash case with two major events in months 10–11, and a High-Acquisition-Cost case (45% instead of 35%). The two engines' insurance service results are compared directly — correlation ≈1.0 — which is the quantitative proof that PAA is a faithful simplification for this contract.

A secondary but consequential finding lives in the same module: the choice of "coverage units" (the revenue recognition pattern) matters. A seasonal weighting `[1,1,1,1,2,3,3,2,2,4,4,1]`, matched to when risk actually occurs, versus a naive flat 1/12 pattern, causes results to diverge by over 860M IDR in peak months — the seasonal pattern is what keeps revenue recognition matched to risk instead of just to time.

**Reinsurance impact (`phase9.py`, `phase10.py`)** — layers a 50% Quota Share plus a Catastrophe Excess-of-Loss treaty (1.5B IDR retention, 5B IDR limit, with reinstatement premium modeled explicitly) on top of the PAA engine, and quantifies gross-vs-net P&L, volatility, and peak claims across all three scenarios. `phase9.py` additionally runs a full tornado sensitivity analysis (event frequency, severity, discount rate, acquisition cost, retention) and an ELR × acquisition-rate heatmap to map the onerous boundary.

**Management levers (`phase11.py`)** — sweeps ten concrete levers (premium, acquisition rate, QS cession, Cat-XoL retention/limit, coverage-unit pattern) against both scenarios and ranks them by impact on profit, volatility, and peak claims, producing the dynamic trigger matrix below.

**Product redesign (`rescue_plan_output.py`)** — a follow-on exploration of a simplified zone-based flat-pricing structure as an alternative to dynamic pricing, including a basic affordability check against target customer income.

## Results

**The core finding**: reinsurance turns a catastrophic year into a profitable one.

| Scenario | Gross ISR (B IDR) | Net ISR (B IDR) | Volatility ↓ | Peak claim ↓ | Onerous? |
|---|---|---|---|---|---|
| Base | −0.73 | 6.77 | 60.6% | 65.1% | No |
| Clash | −9.86 | 4.89 | 76.3% | 82.4% | No |
| High Acquisition Cost | — | 4.77 | 62.7% | 65.1% | No |

In the Clash scenario specifically — two major catastrophe events in the same year — the 50% Quota Share plus Cat-XoL program turns a −9.86B IDR gross loss into a +4.89B IDR net profit, caps peak monthly claims at 1.5B IDR (an 82.4% reduction), and cuts P&L volatility by 76.3%. None of the three scenarios triggers the onerous-contract flag once reinsurance is applied.

**PAA vs GMM**: correlation of ~1.0 between the two measurement approaches across all scenarios — PAA is quantitatively validated as an appropriate simplification for this 12-month contract, avoiding the operational overhead of running the full General Model.

**Governance framework**: the sensitivity analysis was translated into a trigger matrix — pre-approved management actions keyed to specific thresholds, so the response to deteriorating metrics is decided in advance rather than improvised mid-crisis.

| Trigger | Threshold | Action | Expected effect | Owner |
|---|---|---|---|---|
| Acquisition rate ≥ 40% | 0.40 | Cut channel commission 3–5pp | +1.35B profit, volatility ~unchanged | Distribution |
| Severity scale +20% | +20% | Lower Cat-XoL retention to 1.0B | −120M σ, −500M peak, −1.1B profit | Reinsurance |
| ISR volatility > 650M/month | 650M IDR | Raise QS to 60% | −160M σ, flat peak, −0.9B profit | CUO |
| ELR × Acquisition near red zone | ELR ≥0.70 & Acq ≥0.35 | Price +3% or Acq −3pp | Keeps day-one margin positive | Pricing |

## What I'd do differently

The GMM implementation here is intentionally "lite" — CSM/Loss Component amortization is simplified for a single 12-month cohort. A production model would need to handle multi-cohort aggregation, more granular risk adjustment run-off, and OCI vs P&L presentation choices for finance effects, none of which mattered for a single-contract-group case study but would for a real in-force book.

## Files

- `phases.py` — PAA and GMM projection engines, scenario definitions, PAA/GMM equivalence proof, coverage-unit robustness check.
- `phase9.py` — reinsurance overlay, tornado sensitivity analysis, ELR × acquisition heatmap.
- `phase10.py` — gross-vs-net financial summary, profit bridges, management tables per scenario.
- `phase11.py` — management lever sweep and dynamic trigger matrix.
- `rescue_plan_output.py` — exploratory zone-based flat-pricing redesign and affordability check.
- `report.pdf` — full written report submitted to the competition.
- `case_brief.pdf` — the original case materials issued by ASiQ.

## Running it

```bash
pip install -r requirements.txt
python phases.py phase7      # PAA/GMM equivalence proof
python phase9.py             # reinsurance overlay + sensitivity analysis
python phase10.py            # financial summary across scenarios
python phase11.py            # management levers + trigger matrix
python rescue_plan_output.py # product redesign exploration
```
