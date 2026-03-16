
import QuantLib as ql
import numpy as np
import math
from vectorquant.core.backend import get_backend
import vectorquant.finance.derivatives as vq_deriv
import vectorquant.finance.risk_models as vq_risk
from benchmarks.bench_utils import BenchmarkRunner

backend = get_backend()
runner = BenchmarkRunner("Quant Finance")

def run_quant_benchmarks():
    # 1. Black-Scholes Put Pricing
    S, K, r, sigma, T = 100.0, 105.0, 0.05, 0.2, 1.0
    
    def ql_put():
        today = ql.Date(13, 3, 2026)
        ql.Settings.instance().evaluationDate = today
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
        exercise = ql.EuropeanExercise(today + int(T*365))
        option = ql.VanillaOption(payoff, exercise)
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
        rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
        vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed()))
        process = ql.BlackScholesProcess(spot_handle, rate_handle, vol_handle)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return option.NPV()

    runner.run(
        "BlackScholes_Put",
        vq_deriv.black_scholes_put,
        args=(S, K, r, sigma, T),
        compare_to=lambda *args: ql_put()
    )

    # 2. Value at Risk (Batch Performance)
    returns = np.random.normal(0, 0.01, 1000).tolist()
    runner.run(
        "Parametric_VaR_1k",
        vq_risk.parametric_var,
        args=(returns, 0.95)
    )

if __name__ == "__main__":
    run_quant_benchmarks()
    runner.save()
    runner.print_table()
