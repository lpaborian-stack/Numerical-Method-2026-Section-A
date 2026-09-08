import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Input Data (1990 - 2014 World Electric Power Consumption per capita)
x = np.array([
    1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
    2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
    2010, 2011, 2012, 2013, 2014
], dtype=float)

y = np.array([
    2126, 2137, 2118, 2130, 2158, 2204, 2252, 2284, 2311, 2352,
    2425, 2442, 2492, 2572, 2670, 2752, 2845, 2933, 2958, 2906,
    3031, 3086, 3124, 3171, 3222
], dtype=float)

n = len(x)

# 2. Compute summations required for Least Squares
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x**2)

x_bar = np.mean(x)
y_bar = np.mean(y)

# 3. Least Squares Formulas for slope (a1) and intercept (a0)
# a1 = (n * sum(x*y) - sum(x)*sum(y)) / (n * sum(x^2) - (sum(x))^2)
# a0 = y_bar - a1 * x_bar
a1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
a0 = y_bar - a1 * x_bar

# 4. Fitted values and residual calculations
y_pred = a0 + a1 * x
residuals = y - y_pred

# Sum of Squared Errors / Residuals (Sr)
Sr = np.sum(residuals**2)

# Total Sum of Squares around mean (St)
St = np.sum((y - y_bar)**2)

# Coefficient of Determination (r^2)
r2 = (St - Sr) / St

# Standard Error of the Estimate (s_{y/x})
sy_x = np.sqrt(Sr / (n - 2))

# Display quantitative summary
print(f"Intercept (a0): {a0:.6f}")
print(f"Slope (a1): {a1:.6f}")
print(f"Sum of Squared Errors (Sr): {Sr:.6f}")
print(f"Coefficient of Determination (r^2): {r2:.6f}")
print(f"Standard Error of Estimate (s_{y/x}): {sy_x:.6f}")

# 5. Plotting Regression Fit and Residuals
plt.figure(figsize=(12, 5))

# Plot 1: Regression Line vs Data
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='blue', label='Observed Data')
plt.plot(x, y_pred, color='red', linewidth=2, label=f'y = {a0:.2f} + {a1:.2f}x')
plt.xlabel('Year (x)')
plt.ylabel('Electric Power Consumption [kWh per capita] (y)')
plt.title('Simple Linear Regression Fit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Residual Plot
plt.subplot(1, 2, 2)
plt.scatter(x, residuals, color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Year (x)')
plt.ylabel('Residuals (y - y_pred) [kWh per capita]')
plt.title('Residual Plot')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('regression_analysis.png', dpi=300)
plt.show()