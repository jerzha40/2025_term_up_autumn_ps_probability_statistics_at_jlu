# Research Topic Recommendations for Probability & Statistics Essay

## Overview
Based on frontier research from 2024-2025, here are **4 concrete, feasible research topics** that build on your current ARMA work with BTC/USDT data.

---

## 🎯 **RECOMMENDED TOPIC #1: Comparative Analysis of Asymmetric GARCH Models**

### Title (Chinese)
**基于非对称GARCH模型族的比特币波动率建模与预测比较研究**

### Title (English)
**Comparative Study of Asymmetric GARCH Models for Bitcoin Volatility Modeling and Forecasting**

### Research Question
**Does asymmetric volatility (leverage effect) exist in Bitcoin, and which GARCH variant (GARCH, EGARCH, GJR-GARCH, TGARCH) provides the best fit and forecast performance?**

### Why This is Novel
- Recent 2024-2025 studies show **conflicting results** about leverage effects in cryptocurrencies
- Some find "inverted leverage effect" (positive news → higher volatility) vs traditional leverage effect
- No consensus yet on which model is best for Bitcoin specifically

### What You'll Do
1. **Estimate** ARMA(p,q)-GARCH(1,1), ARMA(p,q)-EGARCH(1,1), ARMA(p,q)-GJR-GARCH(1,1)
2. **Compare** using AIC, BIC, log-likelihood
3. **Test** for asymmetric response: γ parameter in GJR-GARCH/EGARCH
4. **Evaluate** out-of-sample forecast accuracy (RMSE, MAE, QLIKE)
5. **Interpret** economic meaning: Do negative shocks cause higher volatility?

### Your Contribution
- **Apply multiple GARCH variants** to YOUR specific BTC/USDT dataset
- **Document the asymmetry pattern** in your data period
- **Provide implementable Python code** using `arch` library

### Data & Programming Requirements ✅
- ✅ You already have BTC/USDT data
- ✅ Python: Install `arch` library (specialized for GARCH models)
- ✅ Code complexity: Medium (extends your current ARMA work)
- ✅ Time: 2-3 weeks

### Frontier References (2024-2025)
1. **Volatility dynamics of cryptocurrencies: a comparative analysis using GARCH-family models** (2025)
   - Future Business Journal
   - Found TGARCH best for BTC, EGARCH best for ETH
   - Data: Jan 2019 - Jan 2025

2. **Volatility Forecasting Using GARCH Versus EGARCH Models** (2024)
   - IJBE Journal
   - Data: April 2018 - Sept 2024
   - Found EGARCH generally better for cryptocurrencies

3. **The Cryptocurrency Market Through Volatility Clustering and Leverage Effects** (2023)
   - ATAMS Journal
   - Used ARMA-GARCH, GJR-GARCH, EGARCH
   - Found evidence of leverage effects

### Python Implementation Outline
```python
from arch import arch_model
import numpy as np

# Your existing returns data
r = np.diff(np.log(close))

# GARCH(1,1)
model_garch = arch_model(r, vol='GARCH', p=1, q=1)
res_garch = model_garch.fit()

# EGARCH(1,1) - captures asymmetry
model_egarch = arch_model(r, vol='EGARCH', p=1, o=1, q=1)
res_egarch = model_egarch.fit()

# GJR-GARCH(1,1) - threshold GARCH
model_gjr = arch_model(r, vol='GARCH', p=1, o=1, q=1)
res_gjr = model_gjr.fit()

# Compare AIC/BIC
print(f"AIC: GARCH={res_garch.aic}, EGARCH={res_egarch.aic}, GJR={res_gjr.aic}")
```

---

## 🎯 **RECOMMENDED TOPIC #2: Rolling Window ARMA-GARCH Forecasting**

### Title (Chinese)
**基于滚动窗口的ARMA-GARCH模型动态校准与短期预测**

### Title (English)
**Dynamic Calibration and Short-term Forecasting of ARMA-GARCH Models Using Rolling Windows**

### Research Question
**How does rolling window size affect ARMA-GARCH forecast accuracy for Bitcoin, and what is the optimal re-calibration frequency for cryptocurrency markets?**

### Why This is Novel
- 2024 research shows rolling windows **significantly improve** crypto forecasts
- Optimal window size is still **under-researched** for different cryptocurrencies
- Trade-off: larger windows (more data, stable) vs smaller windows (adapt to regime changes)

### What You'll Do
1. **Implement** rolling window forecasting with windows: 100, 250, 500, 750 days
2. **Re-estimate** ARMA(p,q)-GARCH(1,1) at each step
3. **Generate** 1-step, 5-step, 10-step ahead forecasts
4. **Evaluate** RMSE, MAE, MAPE for each window size
5. **Analyze** forecast stability during high-volatility periods (e.g., 2021-2022 crash)

### Your Contribution
- **Systematic comparison** of window sizes on YOUR data
- **Identify optimal window** for BTC/USDT specifically
- **Document performance** during different market regimes

### Data & Programming Requirements ✅
- ✅ You already have sufficient historical data
- ✅ Python: `statsmodels` ARMA + `arch` GARCH
- ✅ Code complexity: Medium-High (loops + model re-estimation)
- ✅ Time: 2-3 weeks

### Frontier References (2024-2025)
1. **Forecasting Bitcoin returns: Econometric vs. ML** (Berger & Koubová, 2024)
   - Journal of Forecasting (Wiley)
   - RMSE: 0.0402-0.0428 with rolling windows (100, 250, 500)
   - Your goal: Beat or match this benchmark!

2. **Bitcoin Return Dynamics Volatility and Forecasting** (2025)
   - MDPI Journal
   - RMSE: 0.03602-0.03763 with optimized rolling windows
   - 11.27% improvement over benchmarks

3. **Estimating and forecasting bitcoin using ARIMA-GARCH** (2024)
   - Emerald Publishing
   - Recommended rolling windows for high volatility

### Python Implementation Outline
```python
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import numpy as np

window_sizes = [100, 250, 500, 750]
forecasts = []

for window_size in window_sizes:
    for t in range(window_size, len(r)):
        # Rolling window data
        r_window = r[t-window_size:t]

        # Fit ARMA(2,2)-GARCH(1,1)
        am = ARIMA(r_window, order=(2,0,2)).fit()
        residuals = am.resid

        gm = arch_model(residuals, vol='GARCH', p=1, q=1).fit(disp='off')

        # 1-step forecast
        forecast = gm.forecast(horizon=1)
        forecasts.append(forecast.mean.values[-1, 0])

    # Evaluate RMSE
    rmse = np.sqrt(np.mean((r[window_size:] - forecasts)**2))
    print(f"Window {window_size}: RMSE = {rmse}")
```

---

## 🎯 **RECOMMENDED TOPIC #3: ARMA-GARCH vs. Simple Benchmarks**

### Title (Chinese)
**ARMA-GARCH模型对加密货币收益率的预测能力评估：与随机游走及历史均值的比较**

### Title (English)
**Evaluating Predictive Power of ARMA-GARCH for Cryptocurrency Returns: Comparison with Random Walk and Historical Mean Benchmarks**

### Research Question
**Does ARMA-GARCH provide statistically significant forecast improvements over naive models (random walk, historical mean) for Bitcoin returns?**

### Why This is Novel
- Many papers fit ARMA-GARCH but **don't compare** to simple benchmarks
- Cryptocurrency markets may be **too noisy** for linear models to beat random walk
- **Diebold-Mariano test** rarely used in crypto ARMA-GARCH literature

### What You'll Do
1. **Implement** three models:
   - ARMA(p,q)-GARCH(1,1)
   - Random Walk (forecast = 0)
   - Historical Mean (forecast = rolling mean)
2. **Generate** out-of-sample forecasts (last 20% of data)
3. **Compare** RMSE, MAE, directional accuracy
4. **Statistical test**: Diebold-Mariano test for forecast superiority
5. **Interpret**: When does ARMA-GARCH add value vs. when is market too unpredictable?

### Your Contribution
- **Honest assessment** of ARMA-GARCH's actual predictive power
- **Statistical rigor** (significance testing, not just point estimates)
- **Practical guidance** for traders/researchers

### Data & Programming Requirements ✅
- ✅ You already have the data
- ✅ Python: `statsmodels`, `arch`, `scipy` (for DM test)
- ✅ Code complexity: Medium
- ✅ Time: 1.5-2 weeks

### Frontier References (2024)
1. **Forecasting Bitcoin returns: Econometric vs. ML** (2024)
   - Explicitly compares econometric models to benchmarks
   - Random Walk is notoriously hard to beat

2. **Bitcoin Forecasting with Classical Time Series Models** (arXiv 2024)
   - Evaluates classical models on prices and volatility
   - Documents when ARMA fails

### Python Implementation Outline
```python
from arch import arch_model
from scipy.stats import t as t_dist

# Split data: 80% training, 20% testing
split = int(0.8 * len(r))
r_train, r_test = r[:split], r[split:]

# Model 1: ARMA-GARCH
model = arch_model(r_train, p=2, q=2, vol='GARCH', p_garch=1, q_garch=1)
res = model.fit()
forecasts_garch = res.forecast(horizon=len(r_test)).mean.values[-1, :]

# Model 2: Random Walk
forecasts_rw = np.zeros(len(r_test))

# Model 3: Historical Mean
forecasts_hmean = np.full(len(r_test), np.mean(r_train))

# Evaluate
rmse_garch = np.sqrt(np.mean((r_test - forecasts_garch)**2))
rmse_rw = np.sqrt(np.mean((r_test - forecasts_rw)**2))
rmse_hmean = np.sqrt(np.mean((r_test - forecasts_hmean)**2))

print(f"RMSE: GARCH={rmse_garch:.5f}, RW={rmse_rw:.5f}, HMean={rmse_hmean:.5f}")

# Diebold-Mariano test (implement or use library)
```

---

## 🎯 **RECOMMENDED TOPIC #4: COVID-19 Regime Change Analysis**

### Title (Chinese)
**COVID-19疫情前后比特币波动率结构变化的ARMA-GARCH实证研究**

### Title (English)
**Empirical Study of Bitcoin Volatility Structural Changes Before and After COVID-19 Using ARMA-GARCH**

### Research Question
**Did COVID-19 fundamentally change Bitcoin's volatility dynamics, and can ARMA-GARCH models capture this regime shift?**

### Why This is Novel
- COVID-19 caused **massive volatility spike** in March 2020
- Few studies **explicitly model** pre/post-COVID as separate regimes
- Your data likely spans 2017-2025, covering both periods

### What You'll Do
1. **Split** data: Pre-COVID (before March 2020) vs Post-COVID (after)
2. **Estimate** separate ARMA-GARCH for each period
3. **Compare** parameters: α (ARCH), β (GARCH), φ (AR), θ (MA)
4. **Test** structural break: Chow test or LR test
5. **Analyze**: Did volatility persistence change? Did mean-reversion speed change?

### Your Contribution
- **Document regime change** in YOUR specific data
- **Quantify differences** in volatility structure
- **Policy implication**: How should risk models adapt post-pandemic?

### Data & Programming Requirements ✅
- ✅ Your data likely spans COVID period
- ✅ Python: `statsmodels`, `arch`
- ✅ Code complexity: Medium (split data + comparison)
- ✅ Time: 1.5-2 weeks

### Frontier References (2024)
1. **The Cryptocurrency Market Through Volatility Clustering** (2023)
   - Data includes COVID period (Oct 2019 - Sept 2023)
   - Discusses structural changes

2. Many 2024 papers use data spanning COVID but **don't explicitly compare regimes** ← Your opportunity!

### Python Implementation Outline
```python
import pandas as pd
from arch import arch_model

# Define COVID split
covid_date = pd.Timestamp('2020-03-01')
r_pre = r[time < covid_date]
r_post = r[time >= covid_date]

# Fit ARMA(2,2)-GARCH(1,1) for both periods
model_pre = arch_model(r_pre, p=2, q=2, vol='GARCH', p_garch=1, q_garch=1)
res_pre = model_pre.fit()

model_post = arch_model(r_post, p=2, q=2, vol='GARCH', p_garch=1, q_garch=1)
res_post = model_post.fit()

# Compare parameters
print("Pre-COVID:", res_pre.params)
print("Post-COVID:", res_post.params)

# Persistence = α + β
persist_pre = res_pre.params['alpha[1]'] + res_pre.params['beta[1]']
persist_post = res_post.params['alpha[1]'] + res_post.params['beta[1]']
print(f"Volatility persistence: Pre={persist_pre:.3f}, Post={persist_post:.3f}")
```

---

## 📊 Comparison Matrix

| Topic | Novelty | Difficulty | Time | Frontier Papers | Coding |
|-------|---------|-----------|------|----------------|--------|
| **#1 Asymmetric GARCH** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 2-3 weeks | 2024-2025 | Medium |
| **#2 Rolling Window** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 2-3 weeks | 2024-2025 | Medium-High |
| **#3 Benchmark Comparison** | ⭐⭐⭐ | ⭐⭐ | 1.5-2 weeks | 2024 | Medium |
| **#4 COVID Regime Change** | ⭐⭐⭐⭐ | ⭐⭐ | 1.5-2 weeks | 2023-2024 | Medium |

---

## 🎓 My Recommendation: **Topic #1 (Asymmetric GARCH)**

### Why?
1. ✅ **Most papers cited**: 3 major 2024-2025 papers you can reference
2. ✅ **Clear research gap**: Conflicting results about Bitcoin leverage effect
3. ✅ **Natural extension**: Builds directly on your current ARMA work
4. ✅ **Practical value**: Asymmetric models improve risk management
5. ✅ **Manageable scope**: 2-3 weeks with clear deliverables

### Your Unique Angle
**"Asymmetric Volatility in Bitcoin: GARCH Family Comparison with High-Frequency Daily Data"**

- Most studies use **monthly/weekly data** → You use **daily** for more precision
- Explicitly test **"inverted leverage effect hypothesis"** for crypto
- Provide **reproducible Python code** (many papers lack this!)

---

## 📚 How to Structure Your Essay

### Section 1: Introduction (1.5 pages)
- Background: Bitcoin volatility importance for risk management
- Research gap: Conflicting evidence on asymmetric effects
- Your question: Which GARCH variant best captures BTC volatility?
- Contribution: Empirical comparison on recent daily data

### Section 2: Literature Review (2 pages)
- GARCH models for financial volatility (Engle, Bollerslev)
- Asymmetric GARCH extensions (EGARCH, GJR-GARCH)
- Recent crypto studies: Cite 2024-2025 papers I found
- Identify gap: No consensus for Bitcoin specifically

### Section 3: Methodology (2 pages)
- ARMA(p,q) for mean equation
- GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1) specifications
- Parameter estimation: MLE
- Model selection: AIC, BIC, log-likelihood
- Forecast evaluation: RMSE, MAE, QLIKE

### Section 4: Data (1 page)
- Source: Binance BTC/USDT
- Period: [Your data period]
- Preprocessing: Log returns, outlier checks
- Descriptive stats: Mean, std, skewness, kurtosis
- Stationarity tests: ADF (you already did this!)

### Section 5: Results (3 pages)
- Table 1: Model comparison (AIC, BIC, LL)
- Table 2: Parameter estimates with standard errors
- Figure 1: Conditional volatility from each model
- Figure 2: Forecast comparison
- **Key finding**: Which model wins? Is γ significant?

### Section 6: Discussion (1.5 pages)
- Interpret asymmetry parameter γ
- Economic meaning: Leverage effect or inverted?
- Compare to other studies (agree/disagree?)
- Limitations: Sample period, frequency

### Section 7: Conclusion (1 page)
- Summary of findings
- Practical implications for traders/risk managers
- Future work: GARCH-in-Mean, multivariate GARCH

### References (1 page)
- 15-20 references (I can help you build this!)

**Total: 12-15 pages** (perfect for undergrad/master's thesis chapter)

---

## 🚀 Next Steps

1. **Choose your topic** (I recommend #1!)
2. **Install Python libraries**:
   ```bash
   pip install arch statsmodels numpy pandas matplotlib
   ```
3. **Read 2-3 key papers** I listed for your chosen topic
4. **Modify your `test.py`** to implement the models
5. **Create outline** following the structure above
6. **Run analysis** and collect results
7. **Write draft** section by section

Would you like me to:
- ✅ Create a detailed bibliography for your chosen topic?
- ✅ Write starter Python code for your chosen model?
- ✅ Help outline your specific research question?
- ✅ Find additional papers?

---

## 📖 Quick Start Bibliography (Topic #1)

### Core GARCH Theory
- Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. *Econometrica*, 50(4), 987-1007.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
- Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach. *Econometrica*, 59(2), 347-370. [EGARCH]
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance*, 48(5), 1779-1801. [GJR-GARCH]

### Recent Crypto Applications (2024-2025)
1. **Volatility dynamics of cryptocurrencies: a comparative analysis using GARCH-family models** (2025). *Future Business Journal*, Article in press. DOI: 10.1186/s43093-025-00568-w

2. **Volatility Forecasting Using GARCH Versus EGARCH Models for Cryptocurrencies, Indonesian Stocks, and U.S. Stocks** (2024). *IJBE (Integrated Journal of Business and Economics)*, Vol. [TBD].

3. **The Cryptocurrency Market Through the Scope of Volatility Clustering and Leverage Effects** (2023). *Advances in Time Series Analysis and Machine Science*, 1(3). DOI: 10.61187/atams010302

### Implementation & Methods
- Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and Statistical Modeling with Python. *Proceedings of the 9th Python in Science Conference*, 92-96.

**You can build from this!**
