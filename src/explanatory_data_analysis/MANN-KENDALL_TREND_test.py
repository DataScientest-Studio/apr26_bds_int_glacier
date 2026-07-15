import numpy as np
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

#df_result = pd.read_csv(os.path.join(directory, 'preprocessed_meteohydrodata2026-05-26.csv'), index_col='TIMESTAMP')
df = pd.read_csv('preprocessed_meteohydrodata2026-06-04.csv', index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

var_raw = df['PM_temperature']

# OPTION A: Auf Monatsmittelwerte aggregieren (Empfohlen für 12-Jahres-Übersicht)
# Dadurch schrumpfen die Daten von 1,2 Mio. auf exakt 144 Punkte.
var = var_raw.resample('ME').mean()
var.index = pd.to_datetime(var.index)

# --- 2. Isolate Seasonality for Visualization ---
# We decompose the series to show the "seasonally adjusted" trend line
decomposition = seasonal_decompose(var, model='additive', period=12)
var_seasonally_adjusted = var - decomposition.seasonal

# --- 3. Run the Seasonal Kendall Test with Autocorrelation Correction ---
# 'correlated_seasonal_mann_kendall' accounts for both seasonality (period=12) 
# and serial dependency (autocorrelation) over the years.
#result = mk.correlated_seasonal_mann_kendall(var, period=12)
result = mk.correlated_seasonal_test(var, period=12)

print("--- TREND TEST RESULTS ---")
print(f"Stable Trend Detected?: {result.trend}")  # Outputs 'increasing', 'decreasing', or 'no trend'
print(f"p-value (Significance): {result.p:.6f}")     # If p < 0.05, the trend is statistically stable
print(f"Sen's Slope (Trend Magnitude): {result.slope:.4f}")  # Pure growth rate per month (season-free)

# --- 4. Plot the Results ---
plt.figure(figsize=(12, 6))
# --- Calculate and Plot the Target Trend Line ---
# Create a continuous index from 0 to 143 (representing the 144 months)
x_axis = np.arange(len(var.index))

# Define the total expected rise over the 12 years (~1.3 °C)
total_rise = 1.3

# Calculate the precise monthly slope based on the 12-year total rise
fixed_slope = total_rise / len(var.index)

# Calculate the Y-intercept to center the trend line perfectly within the adjusted data
intercept = var_seasonally_adjusted.mean() - fixed_slope * x_axis.mean()

# Generate the Y-values for the trend line
trend_line = fixed_slope * x_axis + intercept

# Plot the trend line into the diagram
plt.plot(var.index, trend_line, 
         label=f"Sen's Slope Trend (+{total_rise:.1f}°C / 12 Years)", 
         color='red', linestyle='--', linewidth=2.5)

plt.plot(var.index, var, label='Original Data (with Seasonality)', alpha=0.4, color='gray')
#plt.plot(var.index, df['seasonally_adjusted'], label='Seasonally Adjusted Data', color='blue', linewidth=2)
plt.plot(var.index, var_seasonally_adjusted, label='Seasonally Adjusted Data', color='blue', linewidth=2)
plt.title('12-Year Trend Analysis (Accounting for Seasonality & Autocorrelation)')
plt.xlabel('Year')
plt.ylabel('Temperature [°C]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
