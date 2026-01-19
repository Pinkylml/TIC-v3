import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# VISUALIZACIÓN COMPARATIVA: ORIGINAL VS REFORZADO
# ==============================================================================

# 1. Cargar Datos
try:
    df = pd.read_csv('03_Data_preparation/Base_Dataset_Fase3.csv')
except:
    df = pd.read_csv('../03_Data_preparation/Base_Dataset_Fase3.csv')

# 2. Filtrar Empleados (Evento=1)
df_emp = df[df['Evento'] == 1].copy()

# 3. Preparar Datos para Comparación
skills = [
    'S_Comunicacion_ESP', 'S_Comunicacion_ING', 'S_Etica_Profesional',
    'S_Responsabilidad_Soc', 'S_Gestion_Proyectos', 'S_Aprendizaje_Dig',
    'S_Liderazgo_Equipo'
]

results = []

print(f"{'HABILIDAD':<25} | {'ORIGINAL':<10} | {'REFORZADO':<10} | {'CAMBIO'}")
print("-" * 65)

for s_col in skills:
    p_col = s_col.replace('S_', 'P_')
    
    if s_col in df_emp.columns and p_col in df_emp.columns:
        # Recuperar Original: S_Reforzado - P (Porque en el CSV ya está sumado)
        original_vals = df_emp[s_col] - df_emp[p_col]
        
        # Calcular Correlaciones (Spearman con T)
        corr_orig = original_vals.corr(df_emp['T'], method='spearman')
        corr_reinf = df_emp[s_col].corr(df_emp['T'], method='spearman')
        
        results.append({
            'Habilidad': s_col.replace('S_', ''),
            'Original': corr_orig,
            'Reforzado': corr_reinf
        })
        
        diff = corr_reinf - corr_orig
        print(f"{s_col.replace('S_', ''):<25} | {corr_orig:.4f}     | {corr_reinf:.4f}     | {diff:+.4f}")

# 4. Graficar
df_plot = pd.DataFrame(results).sort_values('Reforzado')

plt.figure(figsize=(14, 8))
y = np.arange(len(df_plot))
height = 0.35

# Barras Originales (Gris suave)
plt.barh(y + height/2, df_plot['Original'], height, label='Original (Solo Nivel)', color='#95a5a6', alpha=0.7)

# Barras Reforzadas (Color según valor)
# Verde fuerte si mejora mucho la inserción (más negativo)
colors = ['#27ae60' if x < -0.05 else '#2ecc71' if x < 0 else '#e74c3c' for x in df_plot['Reforzado']]
plt.barh(y - height/2, df_plot['Reforzado'], height, label='Reforzado (Nivel + Ayuda)', color=colors)

plt.yticks(y, df_plot['Habilidad'], fontsize=12, fontweight='bold')
plt.xlabel('Correlación Spearman con Tiempo de Inserción (Más Negativo = Mejor)', fontsize=11)
plt.title('IMPACTO DEL REFUERZO: ¿Cómo cambia la predicción al incluir la percepción de ayuda?', fontweight='bold', fontsize=14)
plt.axvline(0, color='black', linewidth=1)
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Añadir flechas o etiquetas de cambio
for i, row in df_plot.iterrows():
    # Posición para el texto del valor reforzado
    val = row['Reforzado']
    offset_x = -0.01 if val < 0 else 0.01
    ha = 'right' if val < 0 else 'left'
    plt.text(val + offset_x, i - height/2, f"{val:.3f}", va='center', ha=ha, fontsize=9, fontweight='bold', color='black')

plt.tight_layout()
plt.show()
