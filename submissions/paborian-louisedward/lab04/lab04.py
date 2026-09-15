import math
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GEOMETRIC SERIES
# ==========================================
def geometric_sum(x, N):
    """Calculate partial sum 1 + x + x^2 + ... + x^N without closed-form equation."""
    total = 0.0
    for k in range(N + 1):
        total += x ** k
    return total

# ==========================================
# PART 2: POWER SERIES
# ==========================================
def power_series(x, coefficients):
    """Evaluate polynomial P_N(x) with given coefficients [a0, a1, a2, ...]."""
    result = 0.0
    for k, ak in enumerate(coefficients):
        result += ak * (x ** k)
    return result

# ==========================================
# PART 3: MACLAURIN SERIES FOR sin(theta)
# ==========================================
def sin_maclaurin(theta_rad, N):
    """Approximate sin(theta) with N non-zero terms centered at 0."""
    result = 0.0
    for n in range(N):
        sign = (-1) ** n
        factorial = math.factorial(2 * n + 1)
        term = sign * (theta_rad ** (2 * n + 1)) / factorial
        result += term
    return result

# ==========================================
# PART 5: TAYLOR SERIES FOR sin(theta)
# ==========================================
def sin_taylor(theta_rad, a_rad, N):
    """Approximate sin(theta) using Taylor expansion centered at a (in radians)."""
    result = 0.0
    sin_a = math.sin(a_rad)
    cos_a = math.cos(a_rad)
    
    for n in range(N):
        derivative_pattern = n % 4
        if derivative_pattern == 0:
            f_deriv = sin_a
        elif derivative_pattern == 1:
            f_deriv = cos_a
        elif derivative_pattern == 2:
            f_deriv = -sin_a
        else:
            f_deriv = -cos_a
            
        term = f_deriv * ((theta_rad - a_rad) ** n) / math.factorial(n)
        result += term
    return result

# ==========================================
# PART 4 & 6: DATA CALCULATIONS & TABLES
# ==========================================
L = 20.0  # Structural length in meters
angles_deg = [1, 2, 5, 10, 15, 20, 30]
a_deg = 10.0  # Taylor center point
a_rad = math.radians(a_deg)

print("=" * 75)
print("PART 4: MACLAURIN APPROXIMATIONS FOR y = L * sin(theta)")
print("=" * 75)

for N in range(1, 5):
    print(f"\n--- Maclaurin Series with {N} Term(s) ---")
    print(f"{'Angle (deg)':<12}{'Exact y (m)':<15}{'Approx y (m)':<15}{'Abs Error (m)':<15}{'Pct Error (%)':<15}")
    print("-" * 72)
    for deg in angles_deg:
        rad = math.radians(deg)
        exact_y = L * math.sin(rad)
        approx_y = L * sin_maclaurin(rad, N)
        abs_err = abs(exact_y - approx_y)
        pct_err = (abs_err / abs(exact_y)) * 100 if exact_y != 0 else 0.0
        print(f"{deg:<12.1f}{exact_y:<15.6f}{approx_y:<15.6f}{abs_err:<15.6e}{pct_err:<15.6f}")

print("\n" + "=" * 75)
print("PART 6: MINIMUM TERMS REQUIRED FOR < 0.1% ERROR TOLERANCE")
print("=" * 75)
print(f"{'Angle (deg)':<12}{'Maclaurin Terms':<18}{'Taylor (10°) Terms':<20}{'Small-Angle (sin(x)~x) Error (%)':<30}")
print("-" * 75)

for deg in angles_deg:
    rad = math.radians(deg)
    exact_y = L * math.sin(rad)
    
    # Min Maclaurin terms
    m_terms = 1
    while True:
        app = L * sin_maclaurin(rad, m_terms)
        err = (abs(exact_y - app) / abs(exact_y)) * 100
        if err < 0.1:
            break
        m_terms += 1

    # Min Taylor terms
    t_terms = 1
    while True:
        app = L * sin_taylor(rad, a_rad, t_terms)
        err = (abs(exact_y - app) / abs(exact_y)) * 100
        if err < 0.1:
            break
        t_terms += 1
        
    # Small angle approximation error (1-term Maclaurin)
    small_angle_y = L * rad
    small_angle_err = (abs(exact_y - small_angle_y) / abs(exact_y)) * 100

    print(f"{deg:<12.1f}{m_terms:<18}{t_terms:<20}{small_angle_err:<30.4f}")

# ==========================================
# PLOTS
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. Convergence Plot
for deg in [5, 15, 30]:
    rad = math.radians(deg)
    exact_y = L * math.sin(rad)
    errors = []
    terms_list = list(range(1, 6))
    for n in terms_list:
        app = L * sin_maclaurin(rad, n)
        err = (abs(exact_y - app) / abs(exact_y)) * 100
        errors.append(err)
    axs[0, 0].plot(terms_list, errors, marker='o', label=f'θ = {deg}°')
axs[0, 0].set_yscale('log')
axs[0, 0].axhline(0.1, color='red', linestyle='--', label='0.1% Tolerance')
axs[0, 0].set_title('Convergence Plot')
axs[0, 0].set_xlabel('Number of Terms')
axs[0, 0].set_ylabel('Percentage Error (%) [Log Scale]')
axs[0, 0].grid(True)
axs[0, 0].legend()

# 2. Function Comparison Plot
import numpy as np
x_deg = np.linspace(0, 30, 100)
x_rad = np.radians(x_deg)
exact_curve = L * np.sin(x_rad)
mac_1 = [L * sin_maclaurin(r, 1) for r in x_rad]
mac_2 = [L * sin_maclaurin(r, 2) for r in x_rad]
axs[0, 1].plot(x_deg, exact_curve, 'k-', linewidth=2, label='Exact y')
axs[0, 1].plot(x_deg, mac_1, 'r--', label='Maclaurin N=1')
axs[0, 1].plot(x_deg, mac_2, 'b-.', label='Maclaurin N=2')
axs[0, 1].set_title('Function Comparison')
axs[0, 1].set_xlabel('Angle (degrees)')
axs[0, 1].set_ylabel('Vertical Component y (m)')
axs[0, 1].grid(True)
axs[0, 1].legend()

# 3. Error Comparison Plot (Maclaurin vs Taylor)
mac_errs = []
taylor_errs = []
for deg in x_deg:
    rad = math.radians(deg)
    ex = L * math.sin(rad)
    m_app = L * sin_maclaurin(rad, 2)
    t_app = L * sin_taylor(rad, a_rad, 2)
    mac_errs.append((abs(ex - m_app) / ex) * 100 if ex != 0 else 0)
    taylor_errs.append((abs(ex - t_app) / ex) * 100 if ex != 0 else 0)

axs[1, 0].plot(x_deg, mac_errs, label='Maclaurin (Center=0°, N=2)')
axs[1, 0].plot(x_deg, taylor_errs, label='Taylor (Center=10°, N=2)')
axs[1, 0].axhline(0.1, color='red', linestyle='--', label='0.1% Tolerance')
axs[1, 0].set_title('Error Comparison (N=2)')
axs[1, 0].set_xlabel('Angle (degrees)')
axs[1, 0].set_ylabel('Percentage Error (%)')
axs[1, 0].set_ylim(0, 2)
axs[1, 0].grid(True)
axs[1, 0].legend()

# Remove 4th unused subplot box
fig.delaxes(axs[1, 1])
plt.tight_layout()
plt.show()

