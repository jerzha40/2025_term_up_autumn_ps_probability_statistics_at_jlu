"""
x=(ax+b)(mod m)
R=x/m
"""

a = 37
b = 7
m = 2**8
x = [0]
R = [x[0] / m]
N = 40000
s = 1000
for i in range(N):
    x.append((a * x[i] + b) % m)
    R.append(x[i + 1] / m)
print([f"{it:.2f}" for it in R[:20]])
print([f"{it:.2f}" for it in x[:20]])

import matplotlib.pyplot as plt

plt.hist(R[s:], bins=2**8, density=True, alpha=0.6, color="g")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.title("PDF (via Histogram)")
plt.show()

import numpy as np

P = []
λ = 2
for r in R[s:]:
    F = np.exp(-λ)
    C = F
    k = 0
    while True:
        if r <= F:
            break
        k += 1
        F += C / k * λ
    P.append(k)

# 统计频数
values, counts = np.unique(P, return_counts=True)

# 转换成概率（频率）
probs = counts / counts.sum()

# 画分布列（PMF）
plt.bar(values, probs, width=0.6, alpha=0.7, edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.title("Probability Mass Function ")
plt.show()
