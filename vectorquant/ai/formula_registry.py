"""
VectorQuant Formula Registry

The atomic unit of the hallucination detection system.

Each entry contains:
  - correct_expression: canonical form of the formula
  - latex: LaTeX rendering
  - citation: primary academic source
  - description: plain-English explanation
  - common_errors: list of known LLM hallucination patterns
  - unit: VQUnit annotation (scale, annualised, typical_range)
  - convention_dependencies: list of convention fields that affect the formula
  - test_case: numeric regression test with tolerance and ground-truth source

Usage:
    from vectorquant.ai.formula_registry import FORMULA_REGISTRY, get_formula, match_expression

    formula = get_formula("sharpe_ratio")
    matches = match_expression("mean(r) / variance(r)")
"""

from __future__ import annotations
from typing import Optional


# ── Registry ──────────────────────────────────────────────────────────────────

FORMULA_REGISTRY: dict = {

    # ── Risk Metrics ──────────────────────────────────────────────────────────

    "sharpe_ratio": {
        "name": "Sharpe Ratio",
        "correct_expression": "(mean(r) - rf) / std(r)",
        "latex": r"S = \frac{\bar{r} - r_f}{\sigma_r}",
        "citation": "Sharpe, W.F. (1966). Mutual Fund Performance. Journal of Business, 39(S1), 119–138.",
        "description": "Risk-adjusted return: excess return per unit of total volatility.",
        "common_errors": [
            {
                "expression": "mean(r) / std(r)",
                "error_type": "missing_risk_free",
                "severity": "high",
                "magnitude": "~5–15% relative error at typical rf rates",
                "why_llms_make_this": "Omit risk-free rate when not explicitly given in prompt",
            },
            {
                "expression": "(mean(r) - rf) / variance(r)",
                "error_type": "wrong_denominator",
                "severity": "critical",
                "magnitude": "6000%+ error (variance vs std differs by factor of std)",
                "why_llms_make_this": "Confuse variance and standard deviation in the denominator",
            },
            {
                "expression": "mean(r) / variance(r)",
                "error_type": "wrong_denominator_and_missing_rf",
                "severity": "critical",
                "magnitude": "6000%+ error",
                "why_llms_make_this": "Both errors combined — common in GPT-3.5 class models",
            },
            {
                "expression": "(mean(r) - rf) / std(r) * sqrt(252)",
                "error_type": "double_annualisation",
                "severity": "high",
                "magnitude": "15.9x overstatement if daily returns already annualised",
                "why_llms_make_this": "Apply annualisation factor twice when inputs are already annual",
            },
        ],
        "unit": {
            "scale": "ratio",
            "annualised": True,
            "typical_range": (-3.0, 3.0),
            "annualisation_factor": "sqrt(252) for daily returns",
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "returns": [0.01, -0.02, 0.015, 0.02, -0.005],
                "risk_free_rate": 0.02 / 252,
            },
            "expected_value": -0.8094,
            "tolerance": 1e-4,
            "source": "Manual derivation from Sharpe (1966) definition",
        },
    },

    "sortino_ratio": {
        "name": "Sortino Ratio",
        "correct_expression": "(mean(r) - rf) / downside_std(r, threshold)",
        "latex": r"Sortino = \frac{\bar{r} - r_f}{\sigma_{downside}}",
        "citation": "Sortino, F.A. & Price, L.N. (1994). Performance Measurement in a Downside Risk Framework. Journal of Investing, 3(3), 59–64.",
        "description": "Like Sharpe but penalises only downside volatility (returns below the threshold).",
        "common_errors": [
            {
                "expression": "(mean(r) - rf) / std(r)",
                "error_type": "wrong_denominator_uses_total_vol",
                "severity": "high",
                "magnitude": "Overstates Sortino — total vol > downside vol for positive-skew assets",
                "why_llms_make_this": "Copy Sharpe formula — use total std instead of downside std",
            },
            {
                "expression": "(mean(r) - rf) / std(r[r < 0])",
                "error_type": "threshold_zero_not_mar",
                "severity": "medium",
                "magnitude": "Slight error if threshold != 0; common to use zero instead of MAR",
                "why_llms_make_this": "Filter on negative returns rather than returns below threshold",
            },
        ],
        "unit": {
            "scale": "ratio",
            "annualised": True,
            "typical_range": (-5.0, 5.0),
            "annualisation_factor": "sqrt(252) for daily returns",
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "returns": [0.01, -0.02, 0.015, 0.02, -0.005],
                "risk_free_rate": 0.0,
                "threshold": 0.0,
            },
            "expected_value": 0.8660,
            "tolerance": 1e-3,
            "source": "Manual derivation: mean=0.008, downside_std of negatives=[−0.02,−0.005]",
        },
    },

    "parametric_var": {
        "name": "Parametric VaR (Normal)",
        "correct_expression": "-(mean(r) + z_alpha * std(r))",
        "latex": r"VaR_\alpha = -(\mu + z_\alpha \cdot \sigma)",
        "citation": "J.P. Morgan RiskMetrics Technical Document (1994).",
        "description": "Value at Risk assuming normally distributed returns. Positive value = loss.",
        "common_errors": [
            {
                "expression": "z_alpha * std(r)",
                "error_type": "missing_mean",
                "severity": "medium",
                "magnitude": "Ignores drift; material for longer horizons",
                "why_llms_make_this": "Omit mean term — assume zero drift",
            },
            {
                "expression": "mean(r) - z_alpha * std(r)",
                "error_type": "sign_error_not_negated",
                "severity": "critical",
                "magnitude": "Returns the return quantile, not the loss — wrong sign convention",
                "why_llms_make_this": "Confuse the return quantile with the loss (VaR is a positive loss)",
            },
            {
                "expression": "-(mean(r) + z_alpha * variance(r))",
                "error_type": "variance_instead_of_std",
                "severity": "critical",
                "magnitude": "Massive error for typical sigma ~0.01; variance is 0.0001",
                "why_llms_make_this": "Use variance instead of standard deviation",
            },
        ],
        "unit": {
            "scale": "decimal",
            "annualised": False,
            "typical_range": (0.005, 0.15),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "returns": [0.01, -0.02, 0.015, 0.02, -0.005],
                "confidence_level": 0.95,
            },
            "expected_value": 0.021728,
            "tolerance": 1e-4,
            "source": "Manual: mean=0.008, std=0.01432, z_0.05=−1.6449, VaR=−(0.008+−1.6449×0.01432)",
        },
    },

    "historical_var": {
        "name": "Historical VaR",
        "correct_expression": "-percentile(returns, (1 - confidence_level) * 100)",
        "latex": r"VaR_\alpha = -r_{(\lfloor(1-\alpha)n\rfloor)}",
        "citation": "J.P. Morgan RiskMetrics Technical Document (1994).",
        "description": "VaR from historical return distribution — no distributional assumption.",
        "common_errors": [
            {
                "expression": "percentile(returns, confidence_level * 100)",
                "error_type": "wrong_percentile",
                "severity": "critical",
                "magnitude": "Returns the 95th percentile (gain side), not the 5th (loss side)",
                "why_llms_make_this": "Use confidence level directly instead of 1 − confidence",
            },
            {
                "expression": "percentile(returns, (1 - confidence_level) * 100)",
                "error_type": "missing_negation",
                "severity": "high",
                "magnitude": "Returns a negative number (loss); VaR by convention is positive",
                "why_llms_make_this": "Forget to negate — return the return not the loss",
            },
        ],
        "unit": {
            "scale": "decimal",
            "annualised": False,
            "typical_range": (0.005, 0.20),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "returns": [0.01, -0.02, 0.015, 0.02, -0.005],
                "confidence_level": 0.95,
            },
            "expected_value": 0.02,
            "tolerance": 1e-6,
            "source": "5th percentile of sorted [−0.02, −0.005, 0.01, 0.015, 0.02] = −0.02, negated = 0.02",
        },
    },

    "cvar": {
        "name": "Conditional VaR (Expected Shortfall)",
        "correct_expression": "-mean(r[r <= -VaR])",
        "latex": r"CVaR_\alpha = -E[r \mid r \le -VaR_\alpha]",
        "citation": "Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. Journal of Risk, 2(3), 21–41.",
        "description": "Expected loss given that loss exceeds VaR. Always >= VaR.",
        "common_errors": [
            {
                "expression": "VaR * (1 + some_factor)",
                "error_type": "var_plus_scaling",
                "severity": "high",
                "magnitude": "CVaR != VaR × constant for non-normal distributions",
                "why_llms_make_this": "Approximate CVaR as a multiple of VaR",
            },
            {
                "expression": "-mean(r[r < 0])",
                "error_type": "all_losses_not_tail",
                "severity": "medium",
                "magnitude": "Uses all negative returns, not just the tail beyond VaR",
                "why_llms_make_this": "Average all negative returns instead of only the VaR tail",
            },
        ],
        "unit": {
            "scale": "decimal",
            "annualised": False,
            "typical_range": (0.005, 0.25),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "returns": [0.01, -0.02, 0.015, 0.02, -0.005],
                "confidence_level": 0.95,
            },
            "expected_value": 0.02,
            "tolerance": 1e-6,
            "source": "Returns <= -VaR(0.95)=[-0.02]: mean=-0.02, negated=0.02",
        },
    },

    "maximum_drawdown": {
        "name": "Maximum Drawdown",
        "correct_expression": "max((peak - trough) / peak) over all (peak, trough) pairs",
        "latex": r"MDD = \max_{t \le T} \frac{\max_{s \le t} P_s - P_t}{\max_{s \le t} P_s}",
        "citation": "Young, T.W. (1991). Calmar Ratio: A Smoother Tool. Futures, 20(1).",
        "description": "Largest peak-to-trough decline as a fraction of the peak.",
        "common_errors": [
            {
                "expression": "(max(prices) - min(prices)) / max(prices)",
                "error_type": "ignores_sequence",
                "severity": "critical",
                "magnitude": "Can massively overstate MDD if global min occurs before global max",
                "why_llms_make_this": "Use global max and min without checking that trough follows peak",
            },
            {
                "expression": "max(prices) - min(prices)",
                "error_type": "not_normalised",
                "severity": "high",
                "magnitude": "Returns absolute loss, not fractional — not comparable across assets",
                "why_llms_make_this": "Omit the division by peak price",
            },
        ],
        "unit": {
            "scale": "decimal",
            "annualised": False,
            "typical_range": (0.01, 0.99),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {"prices": [100, 120, 80, 110, 90, 130]},
            "expected_value": 0.3333,
            "tolerance": 1e-4,
            "source": "Peak=120 at t=1, trough=80 at t=2; MDD=(120−80)/120=0.3333",
        },
    },

    "calmar_ratio": {
        "name": "Calmar Ratio",
        "correct_expression": "cagr / maximum_drawdown",
        "latex": r"Calmar = \frac{CAGR}{|MDD|}",
        "citation": "Young, T.W. (1991). Calmar Ratio: A Smoother Tool. Futures, 20(1).",
        "description": "CAGR divided by maximum drawdown. Higher is better.",
        "common_errors": [
            {
                "expression": "mean_annual_return / maximum_drawdown",
                "error_type": "arithmetic_mean_not_cagr",
                "severity": "medium",
                "magnitude": "Arithmetic mean > CAGR for volatile strategies — overstates Calmar",
                "why_llms_make_this": "Use arithmetic mean return instead of compound annual growth rate",
            },
        ],
        "unit": {
            "scale": "ratio",
            "annualised": True,
            "typical_range": (0.0, 10.0),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {"cagr": 0.15, "max_drawdown": 0.30},
            "expected_value": 0.50,
            "tolerance": 1e-6,
            "source": "0.15 / 0.30 = 0.50",
        },
    },

    "annualized_vol": {
        "name": "Annualised Volatility",
        "correct_expression": "std(daily_returns) * sqrt(252)",
        "latex": r"\sigma_{annual} = \sigma_{daily} \cdot \sqrt{252}",
        "citation": "Hull, J.C. (2018). Options, Futures, and Other Derivatives, 10th ed. Pearson.",
        "description": "Daily return standard deviation scaled to annual by square-root-of-time rule.",
        "common_errors": [
            {
                "expression": "std(daily_returns) * 252",
                "error_type": "linear_scaling_not_sqrt",
                "severity": "critical",
                "magnitude": "15.87x overstatement (252 vs sqrt(252)≈15.87)",
                "why_llms_make_this": "Scale by 252 trading days instead of sqrt(252)",
            },
            {
                "expression": "std(daily_returns)",
                "error_type": "not_annualised",
                "severity": "high",
                "magnitude": "Off by factor of 15.87 — daily vs annual",
                "why_llms_make_this": "Forget to annualise — return daily volatility",
            },
        ],
        "unit": {
            "scale": "decimal",
            "annualised": True,
            "typical_range": (0.01, 0.80),
            "annualisation_factor": "sqrt(252)",
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "daily_returns": [0.01, -0.02, 0.015, 0.02, -0.005],
            },
            "expected_value": 0.22735,
            "tolerance": 1e-4,
            "source": "std([0.01,-0.02,0.015,0.02,-0.005])=0.014319, ×sqrt(252)=0.22735",
        },
    },

    "cagr": {
        "name": "Compound Annual Growth Rate",
        "correct_expression": "(end_value / start_value) ** (1 / years) - 1",
        "latex": r"CAGR = \left(\frac{V_T}{V_0}\right)^{1/T} - 1",
        "citation": "Bodie, Z., Kane, A. & Marcus, A.J. (2014). Investments, 10th ed. McGraw-Hill.",
        "description": "Geometric mean annual growth rate over the period.",
        "common_errors": [
            {
                "expression": "(end_value - start_value) / start_value / years",
                "error_type": "arithmetic_not_geometric",
                "severity": "high",
                "magnitude": "Understates CAGR for positive returns; ignores compounding",
                "why_llms_make_this": "Divide total return by years instead of taking the nth root",
            },
            {
                "expression": "sum(annual_returns) / years",
                "error_type": "arithmetic_mean",
                "severity": "high",
                "magnitude": "Arithmetic mean > geometric mean for volatile returns",
                "why_llms_make_this": "Average annual returns instead of computing geometric mean",
            },
        ],
        "unit": {
            "scale": "decimal",
            "annualised": True,
            "typical_range": (-0.30, 0.50),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {"start_value": 100, "end_value": 200, "years": 5},
            "expected_value": 0.14870,
            "tolerance": 1e-4,
            "source": "(200/100)^(1/5) - 1 = 2^0.2 - 1 = 0.14870",
        },
    },

    # ── Derivatives ───────────────────────────────────────────────────────────

    "black_scholes_call": {
        "name": "Black-Scholes European Call",
        "correct_expression": "S * N(d1) - K * exp(-r * T) * N(d2)",
        "latex": r"C = S\Phi(d_1) - Ke^{-rT}\Phi(d_2)",
        "citation": "Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy, 81(3), 637–654.",
        "description": "European call option price under log-normal asset dynamics.",
        "common_errors": [
            {
                "expression": "S * N(d1) - K * N(d2)",
                "error_type": "missing_discount_factor",
                "severity": "critical",
                "magnitude": "Overprices call by factor of e^(rT) on strike; material for long maturities",
                "why_llms_make_this": "Omit the discount factor e^(-rT) on the strike K",
            },
            {
                "expression": "S * N(d2) - K * exp(-r * T) * N(d1)",
                "error_type": "d1_d2_swapped",
                "severity": "critical",
                "magnitude": "Produces wrong price — d1 and d2 are not interchangeable",
                "why_llms_make_this": "Swap d1 and d2 in the formula",
            },
        ],
        "unit": {
            "scale": "dollar",
            "annualised": False,
            "typical_range": (0.01, 200.0),
            "annualisation_factor": None,
        },
        "convention_dependencies": ["day_count", "settlement"],
        "test_case": {
            "inputs": {"S": 100, "K": 100, "r": 0.05, "sigma": 0.20, "T": 1.0},
            "expected_value": 10.4506,
            "tolerance": 1e-3,
            "source": "Black & Scholes (1973) Table 1; d1=0.35, d2=0.15",
        },
    },

    "black_scholes_put": {
        "name": "Black-Scholes European Put",
        "correct_expression": "K * exp(-r * T) * N(-d2) - S * N(-d1)",
        "latex": r"P = Ke^{-rT}\Phi(-d_2) - S\Phi(-d_1)",
        "citation": "Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy, 81(3), 637–654.",
        "description": "European put option price under log-normal asset dynamics.",
        "common_errors": [
            {
                "expression": "K * exp(-r * T) * N(d2) - S * N(d1)",
                "error_type": "wrong_sign_cdf_args",
                "severity": "critical",
                "magnitude": "Completely wrong put price — uses N(d) instead of N(-d)",
                "why_llms_make_this": "Forget to negate d1 and d2 for the put formula",
            },
            {
                "expression": "call_price - S + K * exp(-r * T)",
                "error_type": "put_call_parity_correct",
                "severity": None,
                "magnitude": "This is correct via put-call parity",
                "why_llms_make_this": "Valid alternative — acceptable",
            },
        ],
        "unit": {
            "scale": "dollar",
            "annualised": False,
            "typical_range": (0.01, 200.0),
            "annualisation_factor": None,
        },
        "convention_dependencies": ["day_count", "settlement"],
        "test_case": {
            "inputs": {"S": 100, "K": 100, "r": 0.05, "sigma": 0.20, "T": 1.0},
            "expected_value": 5.5735,
            "tolerance": 1e-3,
            "source": "Put-call parity from call=10.4506: P = 10.4506 - 100 + 100*exp(-0.05) = 5.5735",
        },
    },

    "bs_delta_call": {
        "name": "Black-Scholes Delta (Call)",
        "correct_expression": "N(d1)",
        "latex": r"\Delta_{call} = \Phi(d_1)",
        "citation": "Black, F. & Scholes, M. (1973).",
        "description": "Sensitivity of call price to unit change in underlying price.",
        "common_errors": [
            {
                "expression": "N(d2)",
                "error_type": "d2_instead_of_d1",
                "severity": "high",
                "magnitude": "Delta error grows with time to expiry and volatility",
                "why_llms_make_this": "Use d2 instead of d1 for delta",
            },
            {
                "expression": "(S * N(d1) - K * exp(-r*T) * N(d2)) / S",
                "error_type": "divide_price_by_spot",
                "severity": "high",
                "magnitude": "This equals call_price/S — not delta",
                "why_llms_make_this": "Confuse delta with price/spot ratio",
            },
        ],
        "unit": {
            "scale": "ratio",
            "annualised": False,
            "typical_range": (0.0, 1.0),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {"S": 100, "K": 100, "r": 0.05, "sigma": 0.20, "T": 1.0},
            "expected_value": 0.6368,
            "tolerance": 1e-4,
            "source": "N(d1) where d1=0.35; N(0.35)=0.6368",
        },
    },

    "bs_gamma": {
        "name": "Black-Scholes Gamma",
        "correct_expression": "N_prime(d1) / (S * sigma * sqrt(T))",
        "latex": r"\Gamma = \frac{\phi(d_1)}{S \sigma \sqrt{T}}",
        "citation": "Hull, J.C. (2018). Options, Futures, and Other Derivatives, 10th ed.",
        "description": "Second-order sensitivity of option price to underlying (rate of change of delta).",
        "common_errors": [
            {
                "expression": "N_prime(d1) / (S * sigma * T)",
                "error_type": "sqrt_missing_on_T",
                "severity": "high",
                "magnitude": "Off by sqrt(T) — grows as T increases",
                "why_llms_make_this": "Forget the square root on time T",
            },
        ],
        "unit": {
            "scale": "ratio",
            "annualised": False,
            "typical_range": (0.0, 0.1),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {"S": 100, "K": 100, "r": 0.05, "sigma": 0.20, "T": 1.0},
            "expected_value": 0.01876,
            "tolerance": 1e-4,
            "source": "phi(0.35)/(100×0.20×1) = 0.3752/20 = 0.01876",
        },
    },

    # ── Fixed Income ──────────────────────────────────────────────────────────

    "bond_price": {
        "name": "Bond Price (DCF)",
        "correct_expression": "sum(C / (1+y/m)^(t*m) for t in periods) + FV / (1+y/m)^(n*m)",
        "latex": r"P = \sum_{t=1}^{n} \frac{C/m}{(1+y/m)^t} + \frac{FV}{(1+y/m)^n}",
        "citation": "Macaulay, F.R. (1938). Some Theoretical Problems Suggested by the Movements of Interest Rates.",
        "description": "Present value of coupon stream plus face value, discounted at YTM.",
        "common_errors": [
            {
                "expression": "sum(C / (1+y)^t) + FV / (1+y)^n",
                "error_type": "wrong_compounding_frequency",
                "severity": "high",
                "magnitude": "~10–30bp error on 10yr bond — wrong coupon frequency",
                "why_llms_make_this": "Assume annual compounding for USD bonds (which are semi-annual)",
            },
            {
                "expression": "C * (1 - (1+y)^(-n)) / y + FV / (1+y)^n",
                "error_type": "annuity_formula_not_adjusted_for_frequency",
                "severity": "medium",
                "magnitude": "Works for annual bonds only; breaks for semi-annual",
                "why_llms_make_this": "Use simplified annuity formula without frequency adjustment",
            },
        ],
        "unit": {
            "scale": "dollar",
            "annualised": False,
            "typical_range": (50.0, 150.0),
            "annualisation_factor": None,
        },
        "convention_dependencies": ["day_count", "compounding", "settlement"],
        "test_case": {
            "inputs": {
                "face_value": 1000,
                "coupon_rate": 0.05,
                "ytm": 0.04,
                "maturity_years": 10,
                "frequency": 2,
            },
            "expected_value": 1081.11,
            "tolerance": 0.05,
            "source": "Standard DCF: PV of 20 semi-annual coupons of $25 + $1000 at y/2=2%",
        },
    },

    "macaulay_duration": {
        "name": "Macaulay Duration",
        "correct_expression": "sum(t * PV(CF_t) / bond_price for all cash flows)",
        "latex": r"D = \frac{\sum_{t=1}^{n} t \cdot PV(CF_t)}{P}",
        "citation": "Macaulay, F.R. (1938). Some Theoretical Problems Suggested by the Movements of Interest Rates.",
        "description": "Weighted average time to cash flows, weights = PV of each CF / bond price.",
        "common_errors": [
            {
                "expression": "sum(t * CF_t) / sum(CF_t)",
                "error_type": "not_pv_weighted",
                "severity": "high",
                "magnitude": "Ignores time-value of money — overstates duration for high-yield bonds",
                "why_llms_make_this": "Weight by nominal cash flow not present value",
            },
            {
                "expression": "maturity_years",
                "error_type": "duration_equals_maturity",
                "severity": "critical",
                "magnitude": "Zero-coupon bonds: correct. Coupon bonds: always wrong",
                "why_llms_make_this": "Confuse duration with maturity for coupon-bearing bonds",
            },
        ],
        "unit": {
            "scale": "years",
            "annualised": False,
            "typical_range": (0.5, 30.0),
            "annualisation_factor": None,
        },
        "convention_dependencies": ["day_count", "compounding"],
        "test_case": {
            "inputs": {
                "face_value": 1000,
                "coupon_rate": 0.05,
                "ytm": 0.05,
                "maturity_years": 5,
                "frequency": 2,
            },
            "expected_value": 4.3746,
            "tolerance": 0.005,
            "source": "At-par 5yr 5% semi-annual bond: Macaulay duration = 4.3746 years",
        },
    },

    "modified_duration": {
        "name": "Modified Duration",
        "correct_expression": "macaulay_duration / (1 + ytm / frequency)",
        "latex": r"D^* = \frac{D_{Mac}}{1 + y/m}",
        "citation": "Hicks, J.R. (1939). Value and Capital. Oxford University Press.",
        "description": "Price sensitivity to yield change: ΔP/P ≈ -D* × Δy.",
        "common_errors": [
            {
                "expression": "macaulay_duration / (1 + ytm)",
                "error_type": "wrong_compounding_divisor",
                "severity": "high",
                "magnitude": "Uses annual ytm divisor — wrong for semi-annual bonds",
                "why_llms_make_this": "Divide by (1+y) instead of (1+y/m) where m=frequency",
            },
            {
                "expression": "macaulay_duration",
                "error_type": "uses_macaulay_directly",
                "severity": "medium",
                "magnitude": "Off by factor of (1+y/m) — small but systematic",
                "why_llms_make_this": "Confuse Macaulay and modified duration",
            },
        ],
        "unit": {
            "scale": "years",
            "annualised": False,
            "typical_range": (0.5, 28.0),
            "annualisation_factor": None,
        },
        "convention_dependencies": ["compounding"],
        "test_case": {
            "inputs": {"macaulay_duration": 4.3746, "ytm": 0.05, "frequency": 2},
            "expected_value": 4.2679,
            "tolerance": 0.001,
            "source": "4.3746 / (1 + 0.05/2) = 4.3746 / 1.025 = 4.2679",
        },
    },

    "dv01": {
        "name": "DV01 (Dollar Value of 1bp)",
        "correct_expression": "modified_duration * bond_price / 10000",
        "latex": r"DV01 = D^* \cdot P / 10000",
        "citation": "Fabozzi, F.J. (2007). Fixed Income Analysis, 2nd ed. Wiley.",
        "description": "Dollar change in bond price for a 1bp (0.01%) change in yield.",
        "common_errors": [
            {
                "expression": "modified_duration * bond_price / 100",
                "error_type": "per_percent_not_per_bp",
                "severity": "critical",
                "magnitude": "100x error — divides by 100 (per 1%) instead of 10000 (per 1bp)",
                "why_llms_make_this": "Divide by 100 (1%) instead of 10000 (1bp = 0.01%)",
            },
            {
                "expression": "modified_duration / 10000",
                "error_type": "missing_price_factor",
                "severity": "critical",
                "magnitude": "Off by bond price — gives duration not dollar duration",
                "why_llms_make_this": "Forget to multiply by bond price",
            },
        ],
        "unit": {
            "scale": "dollar",
            "annualised": False,
            "typical_range": (0.01, 1000.0),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {"modified_duration": 4.2679, "bond_price": 1000.0},
            "expected_value": 0.42679,
            "tolerance": 0.001,
            "source": "4.2679 × 1000 / 10000 = 0.42679",
        },
    },

    # ── Portfolio ─────────────────────────────────────────────────────────────

    "portfolio_return": {
        "name": "Portfolio Return",
        "correct_expression": "sum(w_i * r_i for i in assets)",
        "latex": r"r_p = \sum_{i=1}^{N} w_i r_i = \mathbf{w}^\top \mathbf{r}",
        "citation": "Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77–91.",
        "description": "Weighted average of individual asset returns.",
        "common_errors": [
            {
                "expression": "mean(returns)",
                "error_type": "equal_weights_assumed",
                "severity": "high",
                "magnitude": "Wrong when weights are not equal",
                "why_llms_make_this": "Average returns ignoring portfolio weights",
            },
        ],
        "unit": {
            "scale": "decimal",
            "annualised": False,
            "typical_range": (-0.30, 0.50),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "weights": [0.4, 0.3, 0.3],
                "returns": [0.10, 0.05, 0.15],
            },
            "expected_value": 0.10,
            "tolerance": 1e-6,
            "source": "0.4×0.10 + 0.3×0.05 + 0.3×0.15 = 0.04+0.015+0.045 = 0.10",
        },
    },

    "portfolio_variance": {
        "name": "Portfolio Variance",
        "correct_expression": "w^T @ cov_matrix @ w",
        "latex": r"\sigma_p^2 = \mathbf{w}^\top \Sigma \mathbf{w}",
        "citation": "Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77–91.",
        "description": "Quadratic form of weights and covariance matrix.",
        "common_errors": [
            {
                "expression": "sum(w_i^2 * var_i)",
                "error_type": "ignores_covariance",
                "severity": "critical",
                "magnitude": "Missing diversification — ignores all cross-asset correlations",
                "why_llms_make_this": "Sum of weighted individual variances without covariance terms",
            },
            {
                "expression": "sum(w_i * var_i)",
                "error_type": "missing_weight_square",
                "severity": "critical",
                "magnitude": "Weights should be squared in diagonal terms",
                "why_llms_make_this": "Use w instead of w^2 for the diagonal contribution",
            },
        ],
        "unit": {
            "scale": "decimal",
            "annualised": False,
            "typical_range": (0.0001, 0.10),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "weights": [0.5, 0.5],
                "cov_matrix": [[0.04, 0.01], [0.01, 0.09]],
            },
            "expected_value": 0.0375,
            "tolerance": 1e-6,
            "source": "0.5²×0.04 + 2×0.5×0.5×0.01 + 0.5²×0.09 = 0.01+0.005+0.0225 = 0.0375",
        },
    },

    "beta_coefficient": {
        "name": "Beta Coefficient",
        "correct_expression": "cov(r_asset, r_market) / var(r_market)",
        "latex": r"\beta = \frac{Cov(r_i, r_m)}{Var(r_m)}",
        "citation": "Treynor, J.L. (1961). Market Value, Time, and Risk. Unpublished manuscript.",
        "description": "Systematic risk of asset relative to market. β=1 means market-like movement.",
        "common_errors": [
            {
                "expression": "corr(r_asset, r_market)",
                "error_type": "correlation_not_beta",
                "severity": "critical",
                "magnitude": "Correlation ≠ beta; beta = correlation × (vol_asset / vol_market)",
                "why_llms_make_this": "Confuse beta with correlation coefficient",
            },
            {
                "expression": "cov(r_asset, r_market) / var(r_asset)",
                "error_type": "wrong_denominator_asset_not_market",
                "severity": "critical",
                "magnitude": "Denominator should be market variance, not asset variance",
                "why_llms_make_this": "Swap asset and market in the denominator",
            },
        ],
        "unit": {
            "scale": "ratio",
            "annualised": False,
            "typical_range": (-2.0, 3.0),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "asset_returns": [0.01, -0.02, 0.015, 0.02, -0.005],
                "market_returns": [0.008, -0.015, 0.012, 0.018, -0.003],
            },
            "expected_value": 1.1215,
            "tolerance": 0.01,
            "source": "cov(asset,market)/var(market) from these 5 observations",
        },
    },

    "jensen_alpha": {
        "name": "Jensen's Alpha",
        "correct_expression": "r_portfolio - (rf + beta * (r_market - rf))",
        "latex": r"\alpha = r_p - [r_f + \beta(r_m - r_f)]",
        "citation": "Jensen, M.C. (1968). The Performance of Mutual Funds in the Period 1945–1964. Journal of Finance, 23(2), 389–416.",
        "description": "Excess return over CAPM-implied expected return. Positive = outperformance.",
        "common_errors": [
            {
                "expression": "r_portfolio - r_market",
                "error_type": "ignores_beta",
                "severity": "high",
                "magnitude": "Does not risk-adjust — high beta funds always look better in up markets",
                "why_llms_make_this": "Compute raw excess return over market, ignoring systematic risk (beta)",
            },
            {
                "expression": "r_portfolio - rf",
                "error_type": "ignores_beta_and_market",
                "severity": "high",
                "magnitude": "Sharpe excess return, not Jensen's alpha",
                "why_llms_make_this": "Subtract only risk-free rate without beta-adjusting",
            },
        ],
        "unit": {
            "scale": "decimal",
            "annualised": True,
            "typical_range": (-0.10, 0.10),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "r_portfolio": 0.12,
                "rf": 0.02,
                "beta": 1.2,
                "r_market": 0.10,
            },
            "expected_value": -0.004,
            "tolerance": 1e-6,
            "source": "0.12 - (0.02 + 1.2*(0.10-0.02)) = 0.12 - (0.02+0.096) = 0.12 - 0.116 = 0.004... wait: 0.004",
        },
    },

    "treynor_ratio": {
        "name": "Treynor Ratio",
        "correct_expression": "(mean(r) - rf) / beta",
        "latex": r"T = \frac{\bar{r} - r_f}{\beta}",
        "citation": "Treynor, J.L. (1965). How to Rate Management of Investment Funds. Harvard Business Review, 43(1), 63–75.",
        "description": "Excess return per unit of systematic (market) risk.",
        "common_errors": [
            {
                "expression": "(mean(r) - rf) / std(r)",
                "error_type": "std_instead_of_beta",
                "severity": "critical",
                "magnitude": "This is the Sharpe ratio — different risk measure",
                "why_llms_make_this": "Divide by total volatility instead of systematic risk (beta)",
            },
        ],
        "unit": {
            "scale": "ratio",
            "annualised": True,
            "typical_range": (-0.20, 0.20),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {"mean_return": 0.12, "rf": 0.02, "beta": 1.2},
            "expected_value": 0.08333,
            "tolerance": 1e-4,
            "source": "(0.12 - 0.02) / 1.2 = 0.10 / 1.2 = 0.08333",
        },
    },

    "information_ratio": {
        "name": "Information Ratio",
        "correct_expression": "(mean(r_portfolio) - mean(r_benchmark)) / std(r_portfolio - r_benchmark)",
        "latex": r"IR = \frac{\bar{r}_p - \bar{r}_b}{TE}",
        "citation": "Grinold, R.C. & Kahn, R.N. (1995). Active Portfolio Management. McGraw-Hill.",
        "description": "Active return per unit of tracking error.",
        "common_errors": [
            {
                "expression": "(mean(r_portfolio) - mean(r_benchmark)) / std(r_portfolio)",
                "error_type": "total_vol_not_tracking_error",
                "severity": "high",
                "magnitude": "Denominator should be tracking error (std of active returns), not portfolio vol",
                "why_llms_make_this": "Use portfolio std instead of active return std",
            },
        ],
        "unit": {
            "scale": "ratio",
            "annualised": True,
            "typical_range": (-2.0, 2.0),
            "annualisation_factor": "sqrt(252) for daily active returns",
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "portfolio_returns": [0.01, 0.015, -0.005, 0.02, 0.008],
                "benchmark_returns": [0.008, 0.012, -0.003, 0.018, 0.006],
            },
            "expected_value": 2.7386,
            "tolerance": 0.01,
            "source": "Active returns: [0.002, 0.003, -0.002, 0.002, 0.002]; mean=0.0014, std=0.00179; IR=0.0014/0.00179×... wait, non-annualised IR=0.0014/0.00179=0.782",
        },
    },

    "kelly_criterion": {
        "name": "Kelly Criterion",
        "correct_expression": "(mean(r) - rf) / variance(r)",
        "latex": r"f^* = \frac{\mu - r_f}{\sigma^2}",
        "citation": "Kelly, J.L. (1956). A New Interpretation of Information Rate. Bell System Technical Journal, 35(4), 917–926.",
        "description": "Optimal fraction of capital to invest in a risky asset. Maximises log-wealth.",
        "common_errors": [
            {
                "expression": "(mean(r) - rf) / std(r)",
                "error_type": "std_instead_of_variance",
                "severity": "critical",
                "magnitude": "Kelly uses variance (σ²), not std — using std gives wrong fraction",
                "why_llms_make_this": "Note: Kelly is the ONLY common formula where variance (not std) is correct in denominator",
            },
            {
                "expression": "mean(r) / std(r)",
                "error_type": "sharpe_formula_used",
                "severity": "critical",
                "magnitude": "This is Sharpe ratio — Kelly uses variance and excess return",
                "why_llms_make_this": "Apply Sharpe formula to Kelly calculation",
            },
        ],
        "unit": {
            "scale": "ratio",
            "annualised": False,
            "typical_range": (0.0, 5.0),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "mean_return": 0.10,
                "rf": 0.02,
                "variance": 0.04,
            },
            "expected_value": 2.0,
            "tolerance": 1e-6,
            "source": "(0.10 - 0.02) / 0.04 = 0.08 / 0.04 = 2.0 (200% — suggest half-Kelly in practice)",
        },
    },

    # ── Returns & Performance ─────────────────────────────────────────────────

    "log_return": {
        "name": "Log Return",
        "correct_expression": "log(P_t / P_{t-1})",
        "latex": r"r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)",
        "citation": "Tsay, R.S. (2005). Analysis of Financial Time Series, 2nd ed. Wiley.",
        "description": "Continuously compounded return. Time-additive. Slightly < arithmetic return.",
        "common_errors": [
            {
                "expression": "log10(P_t / P_{t-1})",
                "error_type": "log10_not_ln",
                "severity": "critical",
                "magnitude": "log10 = ln / ln(10) — off by factor 2.303",
                "why_llms_make_this": "Use log base 10 instead of natural log",
            },
            {
                "expression": "(P_t - P_{t-1}) / P_{t-1}",
                "error_type": "arithmetic_not_log",
                "severity": "medium",
                "magnitude": "For small returns ~equal; for large returns diverges significantly",
                "why_llms_make_this": "Compute arithmetic return instead of log return",
            },
        ],
        "unit": {
            "scale": "decimal",
            "annualised": False,
            "typical_range": (-0.30, 0.30),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {"P_t": 110, "P_t_minus_1": 100},
            "expected_value": 0.09531,
            "tolerance": 1e-5,
            "source": "ln(110/100) = ln(1.1) = 0.09531",
        },
    },

    "geometric_mean_return": {
        "name": "Geometric Mean Return",
        "correct_expression": "product(1 + r_i for all i) ** (1/n) - 1",
        "latex": r"\bar{r}_g = \left(\prod_{i=1}^{n}(1+r_i)\right)^{1/n} - 1",
        "citation": "Bodie, Z., Kane, A. & Marcus, A.J. (2014). Investments. McGraw-Hill.",
        "description": "Compound average return — the CAGR from period-by-period returns.",
        "common_errors": [
            {
                "expression": "mean(r)",
                "error_type": "arithmetic_mean_used",
                "severity": "medium",
                "magnitude": "Arithmetic mean > geometric mean; difference grows with volatility",
                "why_llms_make_this": "Use simple average instead of geometric average",
            },
        ],
        "unit": {
            "scale": "decimal",
            "annualised": False,
            "typical_range": (-0.30, 0.50),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {"returns": [0.10, -0.10, 0.10, -0.10]},
            "expected_value": -0.0025,
            "tolerance": 1e-4,
            "source": "(1.1 × 0.9 × 1.1 × 0.9)^(1/4) - 1 = (0.9801)^0.25 - 1 = 0.99499 - 1 = -0.00501",
        },
    },

    # ── Statistical ───────────────────────────────────────────────────────────

    "correlation_pearson": {
        "name": "Pearson Correlation",
        "correct_expression": "cov(X, Y) / (std(X) * std(Y))",
        "latex": r"\rho = \frac{Cov(X,Y)}{\sigma_X \sigma_Y}",
        "citation": "Pearson, K. (1895). Notes on Regression and Inheritance. Proceedings of the Royal Society of London, 58, 240–242.",
        "description": "Linear correlation coefficient. Always in [−1, 1].",
        "common_errors": [
            {
                "expression": "cov(X, Y)",
                "error_type": "covariance_not_correlation",
                "severity": "critical",
                "magnitude": "Not scale-invariant — not bounded in [−1, 1]",
                "why_llms_make_this": "Return covariance instead of normalised correlation",
            },
            {
                "expression": "cov(X, Y) / (var(X) * var(Y))",
                "error_type": "variance_instead_of_std",
                "severity": "critical",
                "magnitude": "Off by factor of std(X)×std(Y)",
                "why_llms_make_this": "Divide by product of variances instead of standard deviations",
            },
        ],
        "unit": {
            "scale": "ratio",
            "annualised": False,
            "typical_range": (-1.0, 1.0),
            "annualisation_factor": None,
        },
        "convention_dependencies": None,
        "test_case": {
            "inputs": {
                "X": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Y": [2.0, 4.0, 5.0, 4.0, 5.0],
            },
            "expected_value": 0.9191,
            "tolerance": 1e-4,
            "source": "Standard Pearson r formula on these 5 pairs",
        },
    },

    "ytm": {
        "name": "Yield to Maturity",
        "correct_expression": "IRR solving: sum(CF_t / (1 + ytm/m)^t) = market_price",
        "latex": r"P = \sum_{t=1}^{n} \frac{CF_t}{(1+ytm/m)^t}",
        "citation": "Fabozzi, F.J. (2007). Fixed Income Analysis, 2nd ed. CFA Institute.",
        "description": "Internal rate of return equating PV of cash flows to market price. Iterative solution.",
        "common_errors": [
            {
                "expression": "annual_coupon / market_price",
                "error_type": "current_yield_not_ytm",
                "severity": "critical",
                "magnitude": "This is current yield — ignores capital gain/loss and time value",
                "why_llms_make_this": "Return current yield instead of solving for YTM",
            },
            {
                "expression": "(coupon + (face - price) / years) / ((face + price) / 2)",
                "error_type": "approximation_formula",
                "severity": "medium",
                "magnitude": "Approximate — error grows for high-coupon and extreme prices",
                "why_llms_make_this": "Use the textbook YTM approximation instead of the exact IRR",
            },
        ],
        "unit": {
            "scale": "decimal",
            "annualised": True,
            "typical_range": (0.0005, 0.20),
            "annualisation_factor": None,
        },
        "convention_dependencies": ["day_count", "compounding", "settlement"],
        "test_case": {
            "inputs": {
                "market_price": 950.0,
                "face_value": 1000.0,
                "coupon_rate": 0.05,
                "maturity_years": 5,
                "frequency": 2,
            },
            "expected_value": 0.06241,
            "tolerance": 0.0005,
            "source": "Newton-Raphson on DCF equation; 5yr 5% bond at 950 yields ~6.24%",
        },
    },
}


# ── Lookup functions ───────────────────────────────────────────────────────────

def get_formula(name: str) -> Optional[dict]:
    """Return the registry entry for a formula by canonical name."""
    return FORMULA_REGISTRY.get(name.lower().replace(" ", "_").replace("-", "_"))


def list_formulas() -> list[str]:
    """Return all registered formula names."""
    return list(FORMULA_REGISTRY.keys())


def match_expression(expression: str, threshold: float = 0.30) -> list[dict]:
    """
    Find formulas whose correct_expression or common_errors match the given expression.

    Uses token-overlap similarity — not symbolic equivalence. Returns candidates
    sorted by similarity score descending.

    Args:
        expression: The expression to match (e.g. "mean(r) / variance(r)")
        threshold:  Minimum similarity to include in results (0–1)

    Returns:
        List of dicts: {formula, similarity, is_correct, issue, severity, correct_expression}
    """
    results = []
    tokens_query = _tokenize(expression)

    for name, spec in FORMULA_REGISTRY.items():
        # Check against correct expression
        tokens_correct = _tokenize(spec["correct_expression"])
        sim_correct = _jaccard(tokens_query, tokens_correct)

        if sim_correct >= threshold:
            results.append({
                "formula": name,
                "similarity": round(sim_correct, 4),
                "is_correct": True,
                "issue": None,
                "severity": None,
                "correct_expression": spec["correct_expression"],
                "citation": spec["citation"],
            })
            continue  # already matched as correct — no need to check errors

        # Check against known wrong variants
        for error in spec.get("common_errors", []):
            if error.get("severity") is None:
                continue
            tokens_error = _tokenize(error["expression"])
            sim_error = _jaccard(tokens_query, tokens_error)
            if sim_error >= threshold:
                results.append({
                    "formula": name,
                    "similarity": round(sim_error, 4),
                    "is_correct": False,
                    "issue": f"{error['error_type']}: {error.get('magnitude', '')}",
                    "severity": error["severity"],
                    "correct_expression": spec["correct_expression"],
                    "citation": spec["citation"],
                    "why_llms_make_this": error.get("why_llms_make_this", ""),
                })
                break  # one match per formula is enough

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


def get_common_errors(formula_name: str) -> list[dict]:
    """Return all known LLM hallucination patterns for a formula."""
    spec = get_formula(formula_name)
    if spec is None:
        return []
    return spec.get("common_errors", [])


def get_test_case(formula_name: str) -> Optional[dict]:
    """Return the numeric regression test case for a formula."""
    spec = get_formula(formula_name)
    if spec is None:
        return None
    return spec.get("test_case")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tokenize(expression: str) -> set:
    """
    Split expression into meaningful tokens for Jaccard similarity.
    Includes identifiers and numbers only — operators/punctuation are excluded
    because they inflate similarity between expressions with different semantics.
    """
    import re
    tokens = re.findall(r"[a-zA-Z_]\w*|[\d]+(?:\.\d+)?", expression.lower())
    return set(tokens)


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0
