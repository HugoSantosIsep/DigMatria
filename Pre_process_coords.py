

import shapely
import pandas as pd
import re
import numpy as np

from sklearn.preprocessing import LabelEncoder
from shapely.geometry import Polygon
from sklearn.decomposition import PCA



# ============================================================
# PARSE DAS COORDENADAS DO POLÍGONO
# ============================================================

def parse_polygon(coord_string):
    """
    Converte uma string de coordenadas para uma lista de pares
    [(x1, y1), (x2, y2), ...].

    Aceita formatos como:
        [(10,20), (30,40), (50,60)]
        (10,20),(30,40),(50,60)
    """

    if pd.isna(coord_string):
        return []

    # Encontrar todos os pares (x,y)
    matches = re.findall(
        r'\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?',
        str(coord_string)
    )

    points = [
        (float(x), float(y))
        for x, y in matches
    ]

    return points


# ============================================================
# ORIENTAÇÃO DO POLÍGONO
# ============================================================

def polygon_orientation(polygon):
    """
    Calcula a orientação principal do polígono através de PCA.

    Retorna:
        orientation_deg : orientação em [0, 180)
        anisotropy      : medida da anisotropia da forma
    """

    # Verificar se o polígono é válido
    if polygon is None or polygon.is_empty:
        return 0.0, 0.0

    # Obter coordenadas do exterior do polígono
    coords = np.asarray(
        polygon.exterior.coords
    )

    # Remover o último ponto, que normalmente duplica o primeiro
    if len(coords) > 1 and np.allclose(
        coords[0],
        coords[-1]
    ):
        coords = coords[:-1]

    # É necessário ter pelo menos 2 pontos
    if len(coords) < 2:
        return 0.0, 0.0

    # PCA
    pca = PCA(n_components=2)
    pca.fit(coords)

    # Primeiro componente principal
    principal_axis = pca.components_[0]

    dx = principal_axis[0]
    dy = principal_axis[1]

    # Ângulo em graus
    angle = np.degrees(
        np.arctan2(dy, dx)
    )

    # Como a orientação é axial:
    # 0° == 180°
    angle = angle % 180

    # Anisotropia
    explained = pca.explained_variance_ratio_

    if len(explained) >= 2:
        anisotropy = (
            explained[0] / explained[1]
            if explained[1] > 0
            else np.inf
        )
    else:
        anisotropy = 0.0

    return angle, anisotropy


# ============================================================
# DELTA ANGULAR
# ============================================================

def axial_angle_delta(series):
    """
    Calcula a diferença angular respeitando a periodicidade
    da orientação [0, 180).
    """

    delta = series.diff()

    # Colocar no intervalo [-90, 90)
    delta = (delta + 90) % 180 - 90

    return delta


# ============================================================
# CONFIGURAÇÃO
# ============================================================

TRAIN_FILE = "consolidado_metricas_2016_2025.csv"
TRAIN_OUTPUT = "dataset/Coords_data_training.csv"

TEST_FILE = (
    "obra/2026/01_01_2026/pred/ps/"
    "metricas_anomalias_faces_malha_degradacao_t20_10a_00m.csv"
)
TEST_OUTPUT = "dataset/Coords_data_test.csv"


# ============================================================
# FUNÇÃO PARA CALCULAR DELTA ANGULAR
# ============================================================
#
# A orientação está definida no intervalo [0, 180).
#
# Exemplo:
#
# 179 -> 1
#
# O delta geométrico não é:
#
# 1 - 179 = -178
#
# mas sim:
#
# +2 graus
#
# Esta função corrige esse comportamento.
# ============================================================

def axial_angle_delta(series):
    delta = series.diff()

    # Colocar o delta no intervalo [-90, 90)
    delta = (delta + 90) % 180 - 90

    return delta


# ============================================================
# PREPROCESSAMENTO
# ============================================================

def preprocess_data(
    input_file,
    output_file,
    le_facade,
    le_anomaly,
    le_figura,
    training=False
):

    print("\n" + "=" * 70)
    print(
        "PROCESSAMENTO:",
        "TRAINING" if training else "TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Ler dataset
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    print(f"Registos iniciais: {len(df)}")

    # --------------------------------------------------------
    # 2. Separar AnomalyID
    # --------------------------------------------------------
    #
    # Exemplo:
    #
    # ABC_001#3
    #
    # Figura_name = ABC_001
    # ID_Figura   = 3
    # --------------------------------------------------------

    split_id = df['AnomalyID'].str.split('#', expand=True)

    df['Figura_name'] = split_id[0]
    df['ID_Figura'] = split_id[1]

    # Garantir que ID_Figura é tratado como string
    df['ID_Figura'] = df['ID_Figura'].astype(str)

    # --------------------------------------------------------
    # 3. Filtrar apenas corrosão e faces válidas
    # --------------------------------------------------------

    df = df[
        (df['FacadeRoofID'] != 'F?') &
        (df['AnomalyType'] == 'corrosion')
    ].copy()

    print(f"Registos após filtro: {len(df)}")

    # --------------------------------------------------------
    # 4. Converter coordenadas do polígono
    # --------------------------------------------------------

    df['polygon_points'] = (
        df['Coordinates (x,y)']
        .apply(parse_polygon)
    )

    # --------------------------------------------------------
    # 5. Criar geometria
    # --------------------------------------------------------

    df["geometry"] = (
        df["polygon_points"]
        .apply(Polygon)
    )

    # --------------------------------------------------------
    # 6. Calcular centróide
    # --------------------------------------------------------

    df['centroid'] = (
        df['geometry']
        .apply(shapely.centroid)
    )

    df['centroid_x'] = (
        df['centroid']
        .apply(lambda p: p.x)
    )

    df['centroid_y'] = (
        df['centroid']
        .apply(lambda p: p.y)
    )

    # --------------------------------------------------------
    # 7. Calcular orientação do polígono
    # --------------------------------------------------------

    results = (
        df["geometry"]
        .apply(polygon_orientation)
    )

    df[
        ["orientation_deg", "anisotropy"]
    ] = pd.DataFrame(
        results.tolist(),
        index=df.index
    )

    # --------------------------------------------------------
    # 8. Representação angular
    # --------------------------------------------------------
    #
    # A orientação é axial:
    #
    # 0° = 180°
    # 45° = 225°
    # 90° = 270°
    #
    # Por isso usamos sin(2θ) e cos(2θ).
    # --------------------------------------------------------

    df['sin_2theta'] = np.sin(
        np.radians(
            2 * df['orientation_deg']
        )
    )

    df['cos_2theta'] = np.cos(
        np.radians(
            2 * df['orientation_deg']
        )
    )

    # --------------------------------------------------------
    # 9. Ordenar os dados
    # --------------------------------------------------------
    #
    # MUITO IMPORTANTE:
    #
    # anteriormente estavas a agrupar apenas por Figura_name.
    #
    # Isso fazia:
    #
    # dano 1 -> dano 2 -> dano 3
    #
    # quando pertenciam à mesma imagem.
    #
    # Agora cada dano possui a sua própria sequência:
    #
    # Figura_name + ID_Figura
    #
    # --------------------------------------------------------

    GROUP_COLS = [
        'Figura_name',
        'ID_Figura'
    ]

    df_balanced = df.sort_values(
        by=GROUP_COLS + ['data'],
        ascending=[True, True, True]
    ).copy()

    # --------------------------------------------------------
    # 10. Delta do centróide
    # --------------------------------------------------------

    df_balanced['centroid_x_delta'] = (
        df_balanced
        .groupby(GROUP_COLS)['centroid_x']
        .diff()
    )

    df_balanced['centroid_y_delta'] = (
        df_balanced
        .groupby(GROUP_COLS)['centroid_y']
        .diff()
    )

    # --------------------------------------------------------
    # 11. Delta da orientação
    # --------------------------------------------------------
    #
    # Não usar simplesmente:
    #
    # orientation(t) - orientation(t-1)
    #
    # porque:
    #
    # 179 -> 1
    #
    # daria -178°.
    #
    # A função axial_angle_delta transforma isso em +2°.
    # --------------------------------------------------------

    df_balanced['orientation_deg_delta'] = (
        df_balanced
        .groupby(GROUP_COLS)['orientation_deg']
        .transform(axial_angle_delta)
    )

    # --------------------------------------------------------
    # 12. Delta da representação angular
    # --------------------------------------------------------

    df_balanced['sin_2theta_delta'] = (
        df_balanced
        .groupby(GROUP_COLS)['sin_2theta']
        .diff()
    )

    df_balanced['cos_2theta_delta'] = (
        df_balanced
        .groupby(GROUP_COLS)['cos_2theta']
        .diff()
    )

    # --------------------------------------------------------
    # 13. Preencher NaN
    # --------------------------------------------------------
    #
    # O primeiro registo de cada dano não possui observação
    # anterior.
    #
    # Portanto os deltas ficam a 0.
    # --------------------------------------------------------

    delta_columns = [
        'centroid_x_delta',
        'centroid_y_delta',
        'orientation_deg_delta',
        'sin_2theta_delta',
        'cos_2theta_delta'
    ]

    for column in delta_columns:
        df_balanced[column] = (
            df_balanced[column]
            .fillna(0)
        )

    # --------------------------------------------------------
    # 14. Encoding das variáveis categóricas
    # --------------------------------------------------------

    if training:

        # -----------------------------------------------
        # FIT dos encoders
        # -----------------------------------------------

        le_facade.fit(
            df_balanced['FacadeRoofID']
        )

        le_anomaly.fit(
            df_balanced['AnomalyType']
        )

        le_figura.fit(
            df_balanced['Figura_name']
        )

    else:

        # -----------------------------------------------
        # Verificar categorias desconhecidas
        # -----------------------------------------------

        unknown_facade = set(
            df_balanced['FacadeRoofID']
        ) - set(
            le_facade.classes_
        )

        unknown_anomaly = set(
            df_balanced['AnomalyType']
        ) - set(
            le_anomaly.classes_
        )

        unknown_figura = set(
            df_balanced['Figura_name']
        ) - set(
            le_figura.classes_
        )

        if unknown_facade:
            raise ValueError(
                "FacadeRoofID desconhecido no TEST: "
                f"{unknown_facade}"
            )

        if unknown_anomaly:
            raise ValueError(
                "AnomalyType desconhecido no TEST: "
                f"{unknown_anomaly}"
            )

        if unknown_figura:
            raise ValueError(
                "Figura_name desconhecido no TEST: "
                f"{unknown_figura}\n\n"
                "Isto acontece porque o LabelEncoder foi "
                "treinado com Figura_name diferentes."
            )

    # -----------------------------------------------
    # Transform
    # -----------------------------------------------

    df_balanced['FacadeRoofID'] = (
        le_facade.transform(
            df_balanced['FacadeRoofID']
        )
    )

    df_balanced['AnomalyType'] = (
        le_anomaly.transform(
            df_balanced['AnomalyType']
        )
    )

    df_balanced['Figura_name'] = (
        le_figura.transform(
            df_balanced['Figura_name']
        )
    )

    # --------------------------------------------------------
    # 15. Remover colunas desnecessárias
    # --------------------------------------------------------

    columns_to_drop = [
        'ImagePath',
        'Coordinates',
        'Area (cm2)',
        'AnomalyID',
        'Coordinates (x,y)',
        'polygon_points',
        'centroid',
        'geometry',
        'data'
    ]

    # Remover apenas as que existem
    columns_to_drop = [
        column
        for column in columns_to_drop
        if column in df_balanced.columns
    ]

    df_balanced.drop(
        columns=columns_to_drop,
        inplace=True
    )

    # --------------------------------------------------------
    # 16. Guardar dataset
    # --------------------------------------------------------

    df_balanced.to_csv(
        output_file,
        index=False
    )

    print(
        f"Dataset guardado em: {output_file}"
    )

    print(
        f"Registos finais: {len(df_balanced)}"
    )

    print(
        f"Número de danos únicos: "
        f"{df_balanced.groupby(['Figura_name', 'ID_Figura']).ngroups}"
    )

    return df_balanced


# ============================================================
# ENCODERS
# ============================================================

le_facade = LabelEncoder()
le_anomaly = LabelEncoder()
le_figura = LabelEncoder()


# ============================================================
# TRAINING
# ============================================================

df_train = preprocess_data(
    input_file=TRAIN_FILE,
    output_file=TRAIN_OUTPUT,
    le_facade=le_facade,
    le_anomaly=le_anomaly,
    le_figura=le_figura,
    training=True
)


# ============================================================
# TEST
# ============================================================

df_test = preprocess_data(
    input_file=TEST_FILE,
    output_file=TEST_OUTPUT,
    le_facade=le_facade,
    le_anomaly=le_anomaly,
    le_figura=le_figura,
    training=False
)
