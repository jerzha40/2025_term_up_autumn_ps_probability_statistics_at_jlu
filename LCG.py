"""
x=(ax+b)(mod m)
R=x/m
"""

a = 37
b = 7
m = 2**8
x = [0]
R = [x[0] / m]
N = 400000
s = 10000
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
