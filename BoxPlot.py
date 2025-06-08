import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Set style for aesthetics
sns.set(style="whitegrid")

# Simulate PAC data across 4 frequencies for 38 subjects
np.random.seed(42)
conditions = ['7 Hz', '9 Hz', '11 Hz', '13 Hz']
means = [0.12, 0.18, 0.25, 0.34]  # Simulated group-level PAC averages
stds = [0.04, 0.035, 0.03, 0.025]  # Reduced variability at higher frequencies

# Build the dataset
data = []
for condition, mean, std in zip(conditions, means, stds):
    pac_values = np.random.normal(loc=mean, scale=std, size=38)
    for value in pac_values:
        data.append({'Frequency': condition, 'PAC (Alpha-Gamma)': value})

df = pd.DataFrame(data)

# Create the boxplot with overlayed stripplot
plt.figure(figsize=(8, 5))
sns.boxplot(x='Frequency', y='PAC (Alpha-Gamma)', data=df, palette='Set2')
sns.stripplot(x='Frequency', y='PAC (Alpha-Gamma)', data=df, color='black', size=3, alpha=0.6)

# Customize labels and grid
plt.title('Figure 2: Alpha-Gamma Phase-Amplitude Coupling Across Frequencies', fontsize=14)
plt.ylabel('PAC (Modulation Index)', fontsize=12)
plt.xlabel('Auditory Stimulation Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and/or show
plt.savefig("figure_2_pac_boxplot.png", dpi=300)
plt.show()
