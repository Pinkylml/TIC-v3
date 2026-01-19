import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# VISUALIZACIÓN DE HABILIDADES REFORZADAS (S + P)
# ==============================================================================

# 1. Cargar el Dataset YA REFORZADO (que acabamos de generar)
df_final = pd.read_csv('../03_Data_preparation/Base_Dataset_Fase3.csv')

# 2. Filtrar solo empleados (para ver velocidad de inserción pura dentro de los casos de éxito)
df_emp = df_final[df_final['Evento'] == 1].copy()

# 3. Calcular Correlación de Spearman
# (Negativo = Acelera la inserción, Positivo = Retrasa/Neutro)
cols_s = [c for c in df_final.columns if c.startswith('S_')]
corr_reinforced = df_emp[cols_s + ['T']].corr(method='spearman')['T'].drop('T').sort_values()

# 4. Graficar
plt.figure(figsize=(12, 7))
# Colores: Verde fuerte si acelera mucho (< -0.05), Verde claro si acelera (< 0), Rojo si retrasa (> 0)
colors = ['#27ae60' if x < -0.05 else '#2ecc71' if x < 0 else '#e74c3c' for x in corr_reinforced.values]

ax = corr_reinforced.plot(kind='barh', color=colors, edgecolor='black', linewidth=0.8)

plt.title('IMPACTO REAL: Habilidades Reforzadas vs Tiempo de Inserción\n(Nivel 1-5 + Bono "Me Ayudó")', 
          fontweight='bold', fontsize=14)
plt.xlabel('Correlación Spearman (Más Negativo = Consigue trabajo MÁS RÁPIDO)', fontsize=11)
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Etiquetas
for index, value in enumerate(corr_reinforced):
    offset = 0.002 if value >= 0 else -0.002
    ha = 'left' if value >= 0 else 'right'
    plt.text(value + offset, index, f'{value:.3f}', va='center', ha=ha, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print("🏁 CONCLUSIÓN DEL REFUERZO:")
print("Las Barras Verdes a la izquierda son las habilidades que, sumando Nivel + Ayuda Percibida,")
print("predicen mejor que un graduado consiga trabajo RÁPIDO.")
