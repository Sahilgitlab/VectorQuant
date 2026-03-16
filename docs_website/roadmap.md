# Roadmap

Version history and future direction of VectorQuant.

---

## Current Version: 5.2

**Release Date:** Q1 2025

### What's Included

✅ **Core Statistics**
- Mean, standard deviation, variance
- Skewness, kurtosis
- Correlation and covariance

✅ **Portfolio Optimization**
- Sharpe ratio maximization
- Risk metrics (parametric/historical VaR, CVaR)
- Black-Litterman model
- Efficient frontier computation

✅ **Derivatives Pricing**
- Black-Scholes European options
- Greeks (delta, gamma, vega, theta, rho)
- Monte Carlo valuation

✅ **Stochastic Models**
- Geometric Brownian Motion
- Heston model
- Safe Monte Carlo configuration (prevents PC hanging)

✅ **Optimization**
- Gradient descent
- BFGS algorithm
- Portfolio constraints

✅ **Risk Models**
- Value-at-Risk
- Conditional Value-at-Risk
- Factor models
- Kalman filter

✅ **AI Verification**
- Hallucination detection
- Proof traces
- Confidence scoring

✅ **Three-Layer Architecture**
- Python API (user-friendly)
- Smart dispatch layer (automatic backend selection)
- C engine (50-200x speedup) with Python fallback

### Performance

- C engine: 50-200x faster than pure Python
- Tests: 251/252 passing (99.6%)
- Deterministic across platforms
- ~5,000 lines of C and Python code

---

## Version History

### 5.0 (Q4 2024) — Initial Release

- Core statistics module
- Portfolio optimization basics
- Basic derivatives pricing
- Monte Carlo framework

### 5.1 (Q1 2025) — Stochastic Expansion

- Advanced Monte Carlo configuration
- Heston stochastic volatility model
- Risk model enhancements
- Performance: 50x speedup

### 5.2 (Q1 2025) — AI Verification + Polish

- Hallucination detection pipeline
- Proof trace system
- Improved documentation
- Safe Monte Carlo defaults
- Performance: 200x speedup on selected operations
- Architecture cleanup
- Full test coverage

---

## Planned Features

### Q2 2025 Track (Near-Term)

**🔄 Factor Models Enhancement**
- Multi-factor risk decomposition
- Fama-French integration
- Performance attribution

**📊 Advanced Optimization**
- Quadratic programming
- Inequality constraints
- Integer programming (for asset selection)

**🎯 Real-Time Risk Dashboard**
- Live portfolio Greeks
- Intraday VaR updates
- Stress testing

### Q3 2025 Track (Mid-Term)

**⚡ GPU Acceleration (CUDA)**
- GPU-accelerated Monte Carlo
- Parallel covariance computation
- ~500x speedup on GPU-compatible operations

**🌍 Distributed Computing**
- Multi-machine portfolio optimization
- Parallel risk aggregation
- MPI/Dask integration

### Q4 2025-Q1 2026 Track (Extended-Term)

**🔮 Quantum Computing**
- Variational quantum eigensolver (VQE) for covariance
- Quantum optimization (QAOA)
- Partnership with quantum hardware providers

---

## Experimental Features (High Priority)

### GPU Support

**Timeline:** Q3 2025

**Scope:**
- CUDA kernels for matrix operations
- GPU Monte Carlo path generation
- Automatic CPU↔GPU memory transfer

**Expected Performance:**
- Matrix multiply: 500x speedup
- Monte Carlo: 300x speedup
- Covariance: 250x speedup

**Status:** Design phase

---

### Distributed Optimization

**Timeline:** Q4 2025

**Scope:**
- Multi-node portfolio optimization
- Parallel grid search
- Distributed Monte Carlo

**Partnership:** Considering integration with Ray or Dask

**Status:** Architecture planning

---

### Quantum Computing

**Timeline:** Q2 2026

**Scope:**
- Variational Quantum Eigensolver for eigen-decomposition
- Quantum Approximate Optimization Algorithm (QAOA)
- Hybrid classical-quantum workflows

**Target Platforms:**
- IBM Quantum
- IonQ
- Google Sycamore

**Status:** Research phase (not ready for production)

---

## Maintenance Track

### Continuous Improvements (Every Release)

✅ Performance optimization
✅ Test coverage increase (target: 99%+)
✅ Documentation updates
✅ Bug fixes and stability

---

## Deprecation Policy

### Backward Compatibility

VectorQuant maintains backward compatibility within major versions.

**Example:**
- 5.0 → 5.1 → 5.2: All APIs remain compatible
- 5.x → 6.0: May introduce breaking changes (with deprecation warnings first)

### Deprecation Timeline

Functions marked `@deprecated` in v5.x will be removed in v6.0:

```python
@deprecated("Use new_function() instead", removal_version="6.0")
def old_function():
    ...
```

---

## Getting Features Implemented

### Roadmap is Flexible

Priority driven by:
1. **User demand** (GitHub issues/requests)
2. **Technical foundation** (aligned with architecture)
3. **Industry relevance** (addressing quant workflow needs)

### How to Influence

1. **Open GitHub Issues** for feature requests
2. **Contribute code** (pull requests welcome)
3. **Share use cases** (helps prioritization)

---

## Support Timeline

### LTS (Long-Term Support)

- **v5.x:** Supported through Q4 2025 (critical fixes)
- **v6.0:** Released Q2 2026 (new 18-month support window)

### Breaking Changes

Major versions (5→6, 6→7) may introduce breaking changes, but:
- Always documented
- Migration guide provided
- Deprecation warnings in prior version

---

## Research Directions

### Beyond v6.0

**Machine Learning Integration**
- Automated parameter tuning
- Anomaly detection in portfolios
- Regime detection

**Causal Inference**
- Causal factor models
- Intervention analysis
- Counterfactual risk computation

**Alternative Data**
- Sentiment analysis integration
- News-driven probability updates
- Real-time event risk

---

## Performance Targets

### Current (v5.2)

- C engine: 50-200x faster than Python
- Covariance: 50x
- Monte Carlo: 170x
- Optimization: 170x

### v6.0 Goals

- Maintain 50x+ baseline speedup (C stable)
- Add GPU option: 300x-500x for suitable operations
- Reduce memory footprint by 20%

### v7.0 Goals (Post-Quantum Era)

- Quantum algorithms: 10-100x for specific problems
- Classical-quantum hybrid: Beat pure classical on large portfolios

---

## Known Limitations

### Current Version (5.2)

| Limitation | Impact | Planned Fix |
|-----------|--------|-------------|
| Single-machine only | Can't optimize 10K assets | Distributed computing (Q4 2025) |
| CPU-only | Slow for 1M+ simulations | GPU acceleration (Q3 2025) |
| No neural networks | Limited to traditional quant | ML integration (v6.0 research) |
| No exotic derivatives | Common in structured products | To be evaluated |

### Explicitly Out of Scope (5.2)

- ❌ Real-time trading execution
- ❌ Data ingestion/market feeds
- ❌ Portfolio accounting
- ❌ Compliance reporting

---

## Architecture Evolution

### v5.2 Structure
```
Python API → Smart Dispatch → C Engine + Python Fallback
```

### v6.0 Structure (Planned)
```
Python API → Smart Dispatch → C Engine + Python + GPU + Distributed
```

### v7.0 Structure (Research)
```
Python API → Smart Dispatch → C Engine + Python + GPU + Distributed + Quantum
```

**Key:** Each layer is optional. Python always works as fallback.

---

## Release Cycle

### Schedule

- **Every 2 months:** Bug fixes and small improvements
- **Every 6 months:** Major feature release
- **Annually:** Benchmark review and performance optimization

### Process

1. Feature development on `develop` branch
2. Beta testing (4-week window)
3. Community feedback and fixes
4. Release to production
5. Maintenance phase

---

## Community & Contribution

### How to Help

1. **Report bugs:** GitHub issues with reproducible examples
2. **Request features:** Describe use case and impact
3. **Contribute code:** Focus areas:
   - GPU kernels (CUDA)
   - Distributed framework integration
   - Additional stochastic models
   - Documentation and examples
4. **Benchmark:** Run performance tests and report results
5. **Test:** Try VectorQuant on your problems

### License

MIT License — Use freely, including in production

---

## Contact & Feedback

### Feedback Channels

- **GitHub Issues:** Bug reports and feature requests
- **Email:** Direct technical questions
- **Discussions:** General usage and best practices

### Keeping Updated

- Watch GitHub repository for releases
- Follow changelog for breaking changes
- Subscribe to performance reports

---

## FAQ: Roadmap Questions

**Q: Will VectorQuant support NumPy integration in future?**
- A: No. Zero-dependency policy is core design. Instead, use Python bindings to convert data.

**Q: When will quantum computing be production-ready?**
- A: Estimated 2027 at earliest. Quantum hardware still in early stages. Current research focus.

**Q: Can I request features?**
- A: Yes. Open GitHub issue describing your use case. Priority given to:
  - Addressing real workflow gaps
  - Aligned with architecture
  - Community interest

**Q: How long will v5.2 be supported?**
- A: Through Q4 2025 (critical bugs). Upgrade to v6.0 for new features (Q2 2026).

**Q: Will old APIs still work after major version bump?**
- A: Deprecation warnings provided one major version ahead. Clear migration guide.

---

## Next Steps

### For Users

1. Try VectorQuant with your portfolio data
2. Run benchmarks (compare to NumPy)
3. Share feedback via GitHub

### For Contributors

1. Fork the repository
2. Pick an issue or feature from roadmap
3. Submit pull request
4. Join community

---

**Last Updated:** Q1 2025

**Next Roadmap Review:** Q2 2025
