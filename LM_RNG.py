"""
x=ax(1-x)
"""

a = 4
x = [0.4]
N = 4000000
s = 100000
for i in range(N):
    x.append(a * x[i] * (1 - x[i]))
print([f"{it:.2f}" for it in x[s : s + 20]])

import matplotlib.pyplot as plt

plt.hist(x[s:], bins=2**8, density=True, alpha=0.6, color="g")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.title("PDF (via Histogram)")
plt.show()
