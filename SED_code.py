# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:41:21 2025

@author: HB
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the Gaussian function
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

# Define the derivative of the Gaussian function
def gaussian_derivative(x, a, b, c):
    return -a * (x - b) / c**2 * np.exp(-(x - b)**2 / (2 * c**2))

# Parameters
a = 1       # Amplitude (height of the peak)
b = 0       # Mean (center of the peak)

# Generate 10 standard deviations around c = 1.0
c_values = np.linspace(0.1, 1.9, 11)  # 10 values from 0.5 to 1.5

# Generate x values
x = np.linspace(0, 5, 1000)

# Compute the Gaussian functions and their derivatives
gaussian_functions = []
derivative_functions = []

for c in c_values:
    y = gaussian(x, a, b, c)
    dy = gaussian_derivative(x, a, b, c) * -1
    gaussian_functions.append(y)
    derivative_functions.append(dy)

# Compute the sum of the Gaussian functions and their derivatives
y_sum = np.sum(gaussian_functions, axis=0) / len(c_values)
dy_sum = np.sum(derivative_functions, axis=0) / len(c_values)

# Plot the positive part of the Gaussian functions
plt.figure(figsize=(12, 8))

# Plot Gaussian functions
plt.subplot(2, 1, 1)
for i, c in enumerate(c_values):
    plt.plot(x, gaussian_functions[i], label=f"Gaussian (c = {c:.2f})")
plt.plot(x, y_sum, label="Sum of Gaussians", color="black", linewidth=2)
plt.title("10 Gaussian Functions with Different Curvatures")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")  # Horizontal line at y=0
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")  # Vertical line at x=0
plt.legend()
plt.grid(True)

# Plot derivatives of Gaussian functions
plt.subplot(2, 1, 2)
for i, c in enumerate(c_values):
    plt.plot(x, derivative_functions[i], label=f"Derivative (c = {c:.2f})", linestyle="--")
plt.plot(x, dy_sum, label="Derivative (Sum of Gaussians)", color="black", linewidth=3, linestyle="--")
plt.title("Derivatives of Gaussian Functions")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")  # Horizontal line at y=0
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")  # Vertical line at x=0
plt.legend()
plt.grid(True)
plt.ylim([0.0,1.01])

plt.tight_layout()
plt.show()