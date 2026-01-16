import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Portfolio Input from CSV
# -------------------------------
portfolio = pd.read_csv("portfolio.csv")

stocks = portfolio["Ticker"].tolist()
weights = portfolio["Weight"].values

# Validation
if round(weights.sum(), 2) != 1.0:
    raise ValueError("Portfolio weights must sum to 1")


start_date = "2021-01-01"
end_date = "2024-01-01"
risk_free_rate = 0.05  # 5%

# -------------------------------
# 2. Download Data
# -------------------------------
data = yf.download(stocks, start=start_date, end=end_date)["Close"]

# -------------------------------
# 3. Daily Returns
# -------------------------------
returns = data.pct_change().dropna()

# -------------------------------
# 4. Portfolio Returns
# -------------------------------
portfolio_returns = returns.dot(weights)

# -------------------------------
# 5. Performance Metrics
# -------------------------------
cagr = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = (cagr - risk_free_rate) / volatility

# -------------------------------
# 6. Maximum Drawdown
# -------------------------------
cumulative_returns = (1 + portfolio_returns).cumprod()
peak = cumulative_returns.cummax()
drawdown = (cumulative_returns - peak) / peak
max_drawdown = drawdown.min()

# -------------------------------
# 7. Benchmark Comparison (NIfty 50)
# -------------------------------
benchmark = yf.download("^NSEI", start=start_date, end=end_date)["Close"]
benchmark_returns = benchmark.pct_change().dropna()
benchmark_cum = (1 + benchmark_returns).cumprod()

# -------------------------------
# 8. Plot Results
# -------------------------------
plt.figure(figsize=(12,6))
plt.plot(cumulative_returns, label="Portfolio")
plt.plot(benchmark_cum, label="Nifty 50", alpha=0.7)
plt.title("Portfolio vs Market Performance")
plt.xlabel("Date")
plt.ylabel("Growth of 1rs")
plt.legend()
plt.show()

# -------------------------------
# 9. Output Summary
# -------------------------------
print("📊 PORTFOLIO PERFORMANCE SUMMARY")
print("--------------------------------")
print(f"CAGR: {round(cagr*100,2)}%")
print(f"Volatility: {round(volatility*100,2)}%")
print(f"Sharpe Ratio: {round(sharpe_ratio,2)}")
print(f"Max Drawdown: {round(max_drawdown*100,2)}%")
