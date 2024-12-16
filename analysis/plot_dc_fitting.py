import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.ticker as ticker
from matplotlib import container

plt.style.use('classic')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
    'figure.figsize': (6, 5),
})

# Load the data
data = pd.read_csv('fibril_analysis_results.csv')

# Filter data for contact distance > 15 for chains
filtered_data = data[data['Contact Distance'] > 15]
x_filtered = filtered_data['Contact Distance']
y_chains_filtered = filtered_data['Number of Chains']

# All data for radius
x_all = data['Contact Distance']
y_radius = data['Fibril Radius (Å)']
y_radius_error = data['Radius Error']

def power_law(x, a, b):
    return a * x**b

def linear(x, a, b):
    return a * x + b

# Fit for chains (contact distance > 15)
popt_chains, _ = curve_fit(power_law, x_filtered, y_chains_filtered)
y_chains_best_fit = power_law(x_filtered, *popt_chains)
y_chains_simplified = 1.6 * x_filtered**1.6

# Fit for radius (all data)
popt_radius, _ = curve_fit(linear, x_all, y_radius)
y_radius_best_fit = linear(x_all, *popt_radius)
y_radius_simplified = 1.1*x_all + 65

# Calculate R-squared for both best fit and simplified equations
r2_chains_best = r2_score(y_chains_filtered, y_chains_best_fit)
r2_chains_simplified = r2_score(y_chains_filtered, y_chains_simplified)
r2_radius_best = r2_score(y_radius, y_radius_best_fit)
r2_radius_simplified = r2_score(y_radius, y_radius_simplified)

def create_plot(x, y, y_error, y_simplified, xlabel, ylabel, equation, filename):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.errorbar(x, y, yerr=y_error, fmt='o', capsize=3, color='blue', ecolor='blue', 
                markersize=5, alpha=0.6, label='Measured')
    ax.plot(x, y_simplified, '--', color='red', linewidth=2, label='Fit')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Get handles and labels for legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove the errorbars
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    # Use them in the legend
    ax.legend(handles, labels, frameon=False, loc='lower right')

    # Set axis limits with padding
    x_padding = (x.max() - x.min()) * 0.1
    y_padding = (y.max() - y.min()) * 0.1
    ax.set_xlim(x.min() - x_padding, x.max() + x_padding)
    ax.set_ylim(y.min() - y_padding, y.max() + y_padding)

    # Set tick locators
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(10))

    # Add equation as text annotation
    ax.text(0.05, 0.95, equation, transform=ax.transAxes, verticalalignment='top', fontsize=18, color='red')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Create plots
create_plot(x_filtered, y_chains_filtered, y_chains_filtered*0.05, y_chains_simplified,
            'Contact Distance $\it{dc}$ (Å)', 'Number of Chains',
            'y = 1.6x$^{1.6}$', 'chains_fit.png')

create_plot(x_all, y_radius, y_radius_error, y_radius_simplified,
            'Contact Distance $\it{dc}$ (Å)', 'Fibril Radius (Å)',
            'y = 1.1x + 65', 'radius_fit.png')

# Write best fit equations to log file
with open('fit_equations.log', 'w') as f:
    f.write("Number of Chains (Contact Distance > 15 Å):\n")
    f.write(f"Original Fit Equation: y = {popt_chains[0]:.4f}x^{popt_chains[1]:.4f}\n")
    f.write(f"R-squared (Original): {r2_chains_best:.4f}\n")
    f.write(f"Simplified Equation: y = 1.6x^1.6\n")
    f.write(f"R-squared (Simplified): {r2_chains_simplified:.4f}\n\n")

    f.write("Fibril Radius (All Data):\n")
    f.write(f"Original Fit Equation: y = {popt_radius[0]:.4f}x + {popt_radius[1]:.4f}\n")
    f.write(f"R-squared (Original): {r2_radius_best:.4f}\n")
    f.write(f"Simplified Equation: y = 1.1x + 65\n")
    f.write(f"R-squared (Simplified): {r2_radius_simplified:.4f}\n")

print("Plots have been saved as 'chains_fit.png' and 'radius_fit.png'.")
print("Best fit equations have been written to 'fit_equations.log'.")