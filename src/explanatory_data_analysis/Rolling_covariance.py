import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

#df_result = pd.read_csv(os.path.join(directory, 'preprocessed_meteohydrodata2026-05-26.csv'), index_col='TIMESTAMP')
df_result = pd.read_csv('preprocessed_meteohydrodata2026-06-04.csv', index_col='TIMESTAMP')
df_result.index = pd.to_datetime(df_result.index)

var = 'PM_precipitation'

# --- 1. DATA ALIGNMENT ---
# Clean data and ensure exact alignment (5-minute intervals)
df_lag = df_result[['water_level', var ]].dropna().copy()

# --- 2. CROSS-CORRELATION CALCULATION ---
# We test a window of 24 hours before and after (12 hours = 144 steps of 5 mins)
# If your glacier is very large, you can increase max_lags to 288 (24 hours)
max_hours = 24
steps_per_hour = 24  # 60 mins / 5 mins
max_lags = max_hours * steps_per_hour

lags = np.arange(-max_lags, max_lags + 1)
correlations = []

for lag in lags:
    # Shift the temperature column to find the delayed effect
    shifted_temp = df_lag[var].shift(lag)
    # Calculate Pearson correlation for this specific time shift
    corr = df_lag['water_level'].corr(shifted_temp)
    correlations.append(corr)

correlations = np.array(correlations)

# --- 3. IDENTIFY THE OPTIMAL TIME LAG ---
# Find the index of the highest positive correlation
best_index = np.argmax(correlations)
best_lag_steps = lags[best_index]
# Convert steps back into real-world minutes and hours
best_lag_minutes = best_lag_steps * 5
best_lag_hours = best_lag_minutes / 60
max_corr_value = correlations[best_index]

# --- 4. PLOTTING THE CROSS-CORRELATION FUNCTION (CCF) ---
plt.figure(figsize=(10, 5))
# Convert the X-axis from "steps" to "hours" for better client readability
lag_hours = (lags * 5) / 60

plt.plot(lag_hours, correlations, color='darkblue', linewidth=2, label='Cross-Correlation')

# Draw a vertical line at the optimal time lag
plt.axvline(
    x=best_lag_hours, 
    color='red', 
    linestyle='--', 
    linewidth=1.5, 
    label=f'Optimal Lag: {best_lag_hours:.2f} hrs (r = {max_corr_value:.2f})'
)

# Visual anchors and grid
plt.axhline(0, color='black', linestyle='-', alpha=0.3)
plt.grid(True, linestyle='--', alpha=0.5)

# Text & Labels (English for international clients)
plt.title(f'Time-Lag Analysis: Glacier Meltwater Response to {var}', fontsize=12, pad=15)
plt.xlabel(f'Time Shift applied to {var} (Hours)', fontsize=10)
plt.ylabel('Pearson Correlation Coefficient (r)', fontsize=10)
plt.xlim(-max_hours, max_hours)
plt.ylim(min(correlations) - 0.05, max(correlations) + 0.05)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()

# Show the plot
plt.show()

# --- 5. EXECUTIVE SUMMARY FOR THE CLIENT ---
print("\n" + "="*40)
print("HYDROLOGICAL INFERENCE SUMMARY")
print("="*40)
print(f"Analysis Resolution  : 5-minute intervals")
print(f"Maximum Correlation  : r = {max_corr_value:.3f}")
if best_lag_hours > 0:
    print(f"Calculated Time Lag  : {best_lag_hours:.2f} hours ({best_lag_minutes} minutes)")
    print(f"Interpretation       : Meltwater takes approx. {best_lag_hours:.1f} hours to travel from the glacier surfaces to the monitoring station.")
else:
    print(f"Calculated Time Lag  : {abs(best_lag_hours):.2f} hours")
    print(f"Interpretation       : Instantaneous or alternative micro-climate forcing detected.")
print("="*40)
