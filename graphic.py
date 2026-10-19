import pandas as pd
import matplotlib.pyplot as plt

# =============================
# CONFIGURAÇÃO
# =============================
ficheiro_csv = "consolidado_metricas_2016_2025.csv"  # <-- altera para o teu ficheiro

# =============================
# 1. LER DADOS
# =============================
df = pd.read_csv(ficheiro_csv, header=None)

df.columns = [
    "data", "id", "material", "tipo",
    "coords_3d", "polygon", "area", "path"
]

# =============================
# 2. TRATAMENTO
# =============================

# Converter datas
df["data"] = pd.to_datetime(df["data"], errors="coerce")

# Garantir área numérica
df["area"] = pd.to_numeric(df["area"], errors="coerce")

# Remover linhas inválidas
df = df.dropna(subset=["data", "area"])

# Extrair nome da imagem (Figura_XXXX)
df["imagem"] = df["id"].str.split("#").str[0]

# =============================
# 3. AGRUPAR DADOS
# =============================
df_grouped = (
    df.groupby(["data", "imagem"])["area"]
    .sum()
    .reset_index()
    .sort_values("data")
)

# =============================
# 4. PLOT
# =============================
plt.figure(figsize = (25,25))

for img in df_grouped["imagem"].unique():
    subset = df_grouped[df_grouped["imagem"] == img]
    plt.plot(subset["data"], subset["area"], marker='o', label=img)

plt.xlabel("Data")
plt.ylabel("Área (Cm2)")
plt.title("Evolução da área de corrosão ao longo do tempo")
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()