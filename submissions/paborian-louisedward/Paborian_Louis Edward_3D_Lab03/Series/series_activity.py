import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================
# EXERCISE 1: Convergence of (1 + 1/n)^n -> e
# ==========================================

periods = {
    "yearly": 1,
    "twice a year": 2,
    "quarterly": 4,
    "monthly": 12,
    "weekly": 52,
    "daily": 365,
    "hourly": 365 * 24,
    "every minute": 365 * 24 * 60,
    "every second": 365 * 24 * 3600,
    "every millisecond": 365 * 24 * 3600 * 1000,
    "every microsecond": 365 * 24 * 3600 * 1000000,
    "every nanosecond": 365 * 24 * 3600 * 1000000000,
}

labels_ex1 = list(periods.keys())
n_values_ex1 = [periods[k] for k in labels_ex1]
calc_values_ex1 = [(1 + 1 / np.float64(n)) ** np.float64(n) for n in n_values_ex1]
errors_ex1 = [abs(val - math.e) for val in calc_values_ex1]

print("Exercise 1: (1 + 1/n)^n")
print(f"{'How often':<20} {'n':>8} {'(1 + 1/n)^n':>14}")
for label, n, value in zip(labels_ex1, n_values_ex1, calc_values_ex1):
    print(f"{label:<20} {n:>8} {value:>14.6f}")

fig1, (ax1_left, ax1_right) = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle("Exercise 1: Convergence of $(1 + 1/n)^n$ to $e$", fontweight="bold")

bars1 = ax1_left.bar(labels_ex1, calc_values_ex1, color="#2b5cdd")
ax1_left.axhline(y=math.e, color="red", linestyle="--", label=f"e = {math.e:.6f}")
ax1_left.set_ylim(1.9, 3.05)
ax1_left.set_ylabel("$(1 + 1/n)^n$")
ax1_left.set_title("Value per compounding period")
ax1_left.set_xticklabels(labels_ex1, rotation=45, ha="right")
ax1_left.grid(axis="y", linestyle=":", alpha=0.5)
ax1_left.legend(loc="lower right")

for bar, val in zip(bars1, calc_values_ex1):
    ax1_left.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.6f}",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=7,
    )

ax1_right.bar(labels_ex1, errors_ex1, color="#ff9900", edgecolor="#885500")
ax1_right.set_yscale("log")
ax1_right.set_ylabel("$|(1 + 1/n)^n - e|$ (log scale)")
ax1_right.set_title("Error shrinks like $e/(2n)$")
ax1_right.set_xticklabels(labels_ex1, rotation=45, ha="right")
ax1_right.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("exercise1.png", dpi=150)


# ==========================================
# EXERCISE 2: Difference Quotient (a^h - 1)/h
# ==========================================

h_values = [0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-07]
a_bases = [2, math.e, 3]
a_labels = ["a=2  (ln a = 0.6931)", "a=e  (ln a = 1.0000)", "a=3  (ln a = 1.0986)"]
colors_ex2 = ["#2b5cdd", "#109618", "#dc3912"]

results_ex2 = {a: [(a**h - 1) / h for h in h_values] for a in a_bases}

print("\nExercise 2: (a^h - 1)/h")
print(f"{'h':>8} {'a = 2':>12} {'a = 2.71828...':>18} {'a = 3':>12}")
for h, val2, vale, val3 in zip(h_values, results_ex2[2], results_ex2[math.e], results_ex2[3]):
    print(f"{h:>8.4g} {val2:>12.6f} {vale:>18.6f} {val3:>12.6f}")

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.suptitle("Exercise 2: $(a^h - 1)/h$ settling to $\ln(a)$", fontweight="bold")
ax2.set_title("Bars: difference quotient per $h$. Dashed lines: the limit $\ln(a)$.", fontsize=10)

x_indices = np.arange(len(h_values))
width = 0.25

for idx, (a, label, color) in enumerate(zip(a_bases, a_labels, colors_ex2)):
    vals = results_ex2[a]
    ax2.bar(
        x_indices + (idx - 1) * width,
        vals,
        width,
        label=label,
        color=color,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.axhline(y=math.log(a), color=color, linestyle="--", linewidth=1)

ax2.set_xticks(x_indices)
ax2.set_xticklabels([f"h = {h}" for h in h_values])
ax2.set_ylabel("$(a^h - 1)/h$")
ax2.set_ylim(0.6, 1.25)
ax2.grid(axis="y", linestyle=":", alpha=0.5)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("exercise2.png", dpi=150)


# ==========================================
# EXERCISE 3: Taylor Series for e^x (x = 1)
# ==========================================

N_terms = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100, 1000, 10000]
partial_sums = [sum(1 / math.factorial(k) for k in range(N)) for N in N_terms]
errors_ex3 = [abs(S - math.e) for S in partial_sums]

print("\nExercise 3: e^x = sum_{n=0}^∞ x^n / n! (x = 1)")
for N, S in zip(N_terms, partial_sums):
    print(f"N = {N:>5}  S_N = {S:.12f}  error = {abs(S - math.e):.3e}")

fig3, (ax3_left, ax3_right) = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle(
    "Exercise 3: $e^x = \sum x^n / n!$ ($x = 1$), summation up to $N = 10,000$ terms",
    fontweight="bold",
)

x_labels_ex3 = [f"{n:,}" for n in N_terms]
bars3 = ax3_left.bar(x_labels_ex3, partial_sums, color="#7030a0", edgecolor="purple")
ax3_left.axhline(y=math.e, color="red", linestyle="--", label=f"e = {math.e:.6f}")
ax3_left.set_ylim(0.9, 3.05)
ax3_left.set_xlabel("number of terms N in the summation")
ax3_left.set_ylabel("$S_N = \sum_{n=0}^{N-1} 1/n!$")
ax3_left.set_title("Histogram of the partial sums")
ax3_left.set_xticklabels(x_labels_ex3, rotation=45, ha="right")
ax3_left.grid(axis="y", linestyle=":", alpha=0.5)
ax3_left.legend(loc="lower right")

for bar, val in zip(bars3, partial_sums):
    ax3_left.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.6f}",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=6,
    )

ax3_right.plot(N_terms, errors_ex3, color="#2f3542", marker="o", linewidth=1)
ax3_right.set_xscale("log")
ax3_right.set_yscale("log")
ax3_right.set_xlabel("number of terms N in the summation (log scale)")
ax3_right.set_ylabel("$|S_N - e|$ (log scale)")
ax3_right.set_title("How many correct digits each N buys (log-log)")
ax3_right.axhspan(1e-17, 1e-15, color="#0066cc", alpha=0.15, label="machine precision zone")
ax3_right.text(
    1.1,
    2e-16,
    "machine precision zone: from N = 20 on, adding terms cannot improve a float sum",
    fontsize=7,
    color="#0044aa",
)
ax3_right.grid(True, which="both", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("exercise3.png", dpi=150)

print("\nSaved plot images: exercise1.png, exercise2.png, exercise3.png")
