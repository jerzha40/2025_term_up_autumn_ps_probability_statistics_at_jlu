import logging
import numpy as np
from utilsarma.core import DataProcessor
from pathlib import Path

# ========= 日志配置 =========
LOG_FILE = "arma.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).parent / LOG_FILE, encoding="utf-8"
        ),  # 写入文件
        # logging.StreamHandler(),  # 同时在控制台显示
    ],
)

# ===== 使用 DataProcessor =====
DP = DataProcessor(Path(__file__).parent.parent / "data" / "BTCUSDT_1d.parquet")
DP.load_data()
logging.info(f"{DP.table}")
logging.info(f"{DP.table['Close']}")
logging.info(f"{type(DP.table['Close'])}")

close_ca = DP.table["Close"].combine_chunks()  # ChunkedArray -> Array(ListType)
close_list = close_ca.to_pylist()[:-1]
close = np.asarray(close_list, dtype=np.float64)  # 价格字符串 -> float64

open_time_ca = DP.table["Open Time"].combine_chunks()  # 同理
open_time_list = open_time_ca.to_pylist()[:-1]  # -> Python list of int(ms)
time = np.asarray(open_time_list, dtype="datetime64[ms]")  # 毫秒 -> datetime64[ms]

logging.info(f"{open_time_ca}")
logging.info(f"{open_time_list}")
logging.info(f"{time}")
logging.info(f"{len(close)}")

r = np.diff(np.log(close))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(10, 4))

# ax.set_title("BTC/USDT Close")
ax.set_xlabel("Time (UTC)")
ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)  # 主网格
ax.minorticks_on()  # 打开次刻度（更细的刻度线）
ax.grid(
    True, which="minor", linestyle=":", linewidth=0.4, alpha=0.5
)  # 次网格ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
fig.tight_layout()

ax.set_ylabel("Price (USDT)")
ax.plot(time, np.log(close) / 100, ".-")
# ax.clear()

ax.set_ylabel("r (1)")
ax.plot(time[:-1], r, ".-")
# ax.clear()

ax.set_ylabel("r**2 (1)")
ax.plot(time[:-1], r**2, ".-")
# ax.clear()

output_path = Path(__file__).parent.parent / "report" / "figs"
output_path.mkdir(parents=True, exist_ok=True)  # 确保目录存在
# fig.savefig(output_path / "close_np.png", dpi=150)
# plt.show()
plt.close(fig)


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

# 假设 r 已经算好了，是一维 numpy 数组

fig_acf, ax_acf = plt.subplots(figsize=(8, 4))
plot_acf(r, lags=40, ax=ax_acf)
ax_acf.set_title("ACF of BTC/USDT log-returns")
fig_acf.tight_layout()

fig_pacf, ax_pacf = plt.subplots(figsize=(8, 4))
plot_pacf(r, lags=40, ax=ax_pacf, method="ywm")
ax_pacf.set_title("PACF of BTC/USDT log-returns")
fig_pacf.tight_layout()

output_path = Path(__file__).parent.parent / "report" / "figs"
output_path.mkdir(parents=True, exist_ok=True)

fig_acf.savefig(output_path / "acf.png", dpi=150)
fig_pacf.savefig(output_path / "pacf.png", dpi=150)

plt.close(fig_acf)
plt.close(fig_pacf)


import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

print("statsmodels version:", sm.__version__)

from statsmodels.tsa.stattools import adfuller

result = adfuller(r)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch

# 基础健诊
assert r.ndim == 1
assert len(r) == len(close) - 1
assert np.all(np.isfinite(r)), "r 里有 NaN/Inf"
print("n, mean, std, min, max =", len(r), r.mean(), r.std(), r.min(), r.max())

# ADF（带关键信息）
adf_stat, adf_p, _, _, adf_crit, _ = adfuller(r, regression="c", autolag="AIC")
print("ADF stat =", adf_stat, "p =", adf_p)
print("ADF critical values:", adf_crit)

# ARCH-LM（看波动聚集）
arch_stat, arch_p, _, _ = het_arch(r)
print("ARCH LM p-value =", arch_p)

# 交叉验证 1：KPSS（原假设=平稳）
kpss_stat, kpss_p, _, kpss_crit = kpss(r, regression="c", nlags="auto")
print("KPSS stat =", kpss_stat, "p =", kpss_p)
print("KPSS critical values:", kpss_crit)


from statsmodels.stats.diagnostic import acorr_ljungbox

# 对收益率本身
print(acorr_ljungbox(r, lags=[10, 20, 30], return_df=True))
# 对平方收益率（检验波动聚集）
print(acorr_ljungbox(r**2, lags=[10, 20, 30], return_df=True))


import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import statsmodels.api as sm

results = []
max_p = 3
max_q = 3

for p in range(max_p + 1):
    for q in range(max_q + 1):
        if p == 0 and q == 0:
            continue
        try:
            model = sm.tsa.ARIMA(r, order=(p, 0, q))
            fitted = model.fit()
            results.append(
                {
                    "p": p,
                    "q": q,
                    "logL": fitted.llf,
                    "AIC": fitted.aic,
                    "BIC": fitted.bic,
                }
            )
        except Exception as e:
            print(f"ARMA({p},{q}) failed: {e}")

df = pd.DataFrame(results)
print("\n按 AIC 排序：")
print(df.sort_values("AIC").head(10))

print("\n按 BIC 排序：")
print(df.sort_values("BIC").head(10))


model = sm.tsa.ARIMA(r, order=(0, 0, 2))
fitted = model.fit()
print(fitted.summary())
resid = fitted.resid

import numpy as np
import matplotlib.pyplot as plt

# 假设 fitted 已经是 ARMA(0,2) 拟合结果
# 1. 取最后 200 个点作为测试集（你可以改成别的长度）
split = -200
r_train = r[:split]
r_test = r[split:]

model = sm.tsa.ARIMA(r_train, order=(0, 0, 2))
fitted_train = model.fit()

# 2. 对测试期做一步预测（滚动）
pred = fitted_train.get_forecast(steps=len(r_test))
mean_forecast = pred.predicted_mean
conf_int = pred.conf_int(alpha=0.05)

# 3. 计算预测误差指标
err = r_test - mean_forecast
rmse = np.sqrt(np.mean(err**2))
mae = np.mean(np.abs(err))
mape = np.mean(np.abs(err / r_test)) * 100

print("RMSE =", rmse)
print("MAE  =", mae)
print("MAPE =", mape)

# 4. 画预测图（真实 vs 预测）
plt.figure(figsize=(10, 4))
plt.plot(r_test, label="read")
plt.plot(mean_forecast, label="predict")
plt.fill_between(
    np.arange(len(mean_forecast)),
    conf_int[:, 0],  # 左列
    conf_int[:, 1],  # 右列
    alpha=0.2,
    label="95% Belief range",
)
plt.legend()
plt.tight_layout()
plt.savefig("report/figs/forecast.png", dpi=150)
plt.close()
