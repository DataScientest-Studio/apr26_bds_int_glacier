import pandas as pd

df = pd.read_csv('mod_meteohydrodata.csv',
                 index_col='TIMESTAMP')
df.index = pd.to_datetime(df.index)

df_numerisch = df.select_dtypes(include=["number"])

# 1. Kreuzkorrelationsmatrix (Werte zwischen -1 und +1) - Standardisiert
korrelations_matrix = df_numerisch.corr(method="pearson")

# 2. Kovarianzmatrix (Nicht standardisiert, zeigt die gemeinsame Varianz)
kovarianz_matrix = df_numerisch.cov()

# Optionale Anzeige in der Konsole
print("Korrelationsmatrix:")
print(korrelations_matrix)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot-Größe definieren (passend für 10 Variablen)
plt.figure(figsize=(10, 8))

# Heatmap zeichnen
sns.heatmap(
    korrelations_matrix,
    annot=True,  # Schreibt die exakten Werte in die Kästchen
    cmap="coolwarm",  # Blau = negativ, Rot = positiv korreliert
    fmt=".2f",  # 2 Nachkommastellen
    linewidths=0.5,  # Dünne Trennlinien zwischen den Kästchen
    vmin=-1,
    vmax=1,  # Skala von -1 bis +1 festlegen
)

plt.title("Kreuzkorrelationsmatrix der Meteohydro-Daten", fontsize=14, pad=20)
plt.tight_layout()
plt.show()
