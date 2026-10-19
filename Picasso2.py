import joblib
import pandas as pd
import numpy as np
import cv2
import math
import os
import json
import re


# =====================================================================
# 1. EXTRAIR NÚMERO DA FIGURA
# =====================================================================
def extrair_numero_figura(nome):
    """
    Converte:
        Figura_25
        Figura_25.png
        figura_25
        figura_25_overlay.png

    para:
        25
    """

    if nome is None:
        return None

    nome = str(nome).strip()

    match = re.search(
        r"Figura[_\-]?(\d+)",
        nome,
        re.IGNORECASE
    )

    if match is None:
        return None

    return int(match.group(1))


# =====================================================================
# 2. NORMALIZAR ID DO DANO
# =====================================================================
def normalizar_det_id(det_id):

    if det_id is None:
        return None

    det_id = str(det_id).strip()

    match = re.match(
        r"Figura[_\-]?(\d+)(?:\.0)?#(\d+)",
        det_id,
        re.IGNORECASE
    )

    if match:

        figura = int(match.group(1))
        dano = int(match.group(2))

        return f"Figura_{figura}#{dano}"

    return det_id


# =====================================================================
# 3. CARREGAR JSON
# =====================================================================
def carregar_danos_json(caminho_json):

    if not os.path.exists(caminho_json):

        raise FileNotFoundError(
            f"JSON não encontrado: {caminho_json}"
        )

    with open(
        caminho_json,
        "r",
        encoding="utf-8"
    ) as f:

        dados_json = json.load(f)

    data_dict = dados_json.get(
        "data",
        {}
    )

    danos_por_figura = {}

    total_danos = 0

    for nome_imagem, lista_danos in data_dict.items():

        figura = extrair_numero_figura(
            nome_imagem
        )

        if figura is None:

            print(
                f"[AVISO] Não foi possível obter "
                f"a figura de: {nome_imagem}"
            )

            continue

        if figura not in danos_por_figura:

            danos_por_figura[figura] = []

        for item in lista_danos:

            det_id = normalizar_det_id(
                item.get("det_id")
            )

            contornos = item.get(
                "contornos_2d",
                []
            )

            if det_id is None:
                continue

            if not contornos:
                continue

            try:

                poligono = np.array(
                    contornos,
                    dtype=np.int32
                )

                if len(poligono) < 3:
                    continue

                danos_por_figura[figura].append(
                    {
                        "det_id": det_id,
                        "classe_nome": item.get("classe_nome", ""),
                        "poligono": poligono
                    }
                )

                total_danos += 1

            except Exception as e:

                print(
                    f"[ERRO] Polígono {det_id}: {e}"
                )

    print(
        f"\nJSON:"
    )

    print(
        f"  Figuras: {len(danos_por_figura)}"
    )

    print(
        f"  Danos: {total_danos}"
    )

    return danos_por_figura


# =====================================================================
# 4. RECONSTRUIR O LABEL ENCODER
# =====================================================================
def construir_mapa_label_encoder(
    danos_por_figura
):
    """
    Reconstrói o mapeamento equivalente ao LabelEncoder.

    O LabelEncoder ordena alfabeticamente as classes.
    """

    nomes_figuras = []

    for figura in danos_por_figura.keys():

        nomes_figuras.append(
            f"Figura_{figura}"
        )

    # IMPORTANTE:
    # exatamente a lógica de ordenação usada pelo LabelEncoder
    nomes_figuras = sorted(
        nomes_figuras
    )

    encoded_to_original = {}

    for encoded, nome_original in enumerate(
        nomes_figuras
    ):

        encoded_to_original[
            encoded
        ] = nome_original

    print(
        "\nMapeamento reconstruído:"
    )

    for encoded, original in (
        encoded_to_original.items()
    ):

        print(
            f"  {encoded} -> {original}"
        )

    return encoded_to_original


# =====================================================================
# 5. CALCULAR CENTRÓIDE
# =====================================================================
def calcular_centroide(poligono):

    M = cv2.moments(
        poligono
    )

    if M["m00"] != 0:

        cx = (
            M["m10"] /
            M["m00"]
        )

        cy = (
            M["m01"] /
            M["m00"]
        )

    else:

        pontos = poligono.reshape(
            -1,
            2
        )

        cx = np.mean(
            pontos[:, 0]
        )

        cy = np.mean(
            pontos[:, 1]
        )

    return cx, cy


# =====================================================================
# 6. ADICIONAR TEXTO
# =====================================================================
def adicionar_texto(
    img,
    texto,
    posicao,
    escala=0.55
):

    fonte = cv2.FONT_HERSHEY_SIMPLEX

    espessura = 2

    (largura, altura), baseline = (
        cv2.getTextSize(
            texto,
            fonte,
            escala,
            espessura
        )
    )

    x, y = posicao

    x = max(
        5,
        x
    )

    y = max(
        altura + baseline + 5,
        y
    )

    # Fundo preto
    cv2.rectangle(
        img,
        (
            x - 4,
            y - altura - baseline - 4
        ),
        (
            x + largura + 4,
            y + 4
        ),
        (0, 0, 0),
        -1
    )

    # Texto branco
    cv2.putText(
        img,
        texto,
        (x, y),
        fonte,
        escala,
        (255, 255, 255),
        espessura,
        cv2.LINE_AA
    )


# =====================================================================
# 7. DESENHAR DANO
# =====================================================================
def desenhar_dano(
    img,
    dano,
    dados_predicao
):

    det_id = dano["det_id"]
    poligono = dano["poligono"]

    # ---------------------------------------------------------------
    # Área do dano
    # ---------------------------------------------------------------
    overlay = img.copy()

    cv2.fillPoly(
        overlay,
        [poligono],
        (255, 255, 0)
    )

    cv2.polylines(
        overlay,
        [poligono],
        True,
        (255, 100, 0),
        2
    )

    cv2.addWeighted(
        overlay,
        0.4,
        img,
        0.6,
        0,
        dst=img
    )

    # ---------------------------------------------------------------
    # Centróide
    # ---------------------------------------------------------------
    cx, cy = calcular_centroide(
        poligono
    )

    cX = int(
        round(cx)
    )

    cY = int(
        round(cy)
    )

    # Centro
    cv2.circle(
        img,
        (cX, cY),
        8,
        (0, 255, 0),
        -1
    )

    # ---------------------------------------------------------------
    # Desenhar Seta usando a direção de movimento prevista (movement_angle)
    # ---------------------------------------------------------------
    if dados_predicao is not None and "movement_angle" in dados_predicao:
        angulo_mov = dados_predicao["movement_angle"]
        angulo_rad = math.radians(angulo_mov)

        comprimento = 120

        # Sistema de coordenadas da imagem
        fim_X = int(
            cX
            + comprimento
            * math.cos(angulo_rad)
        )

        fim_Y = int(
            cY
            - comprimento
            * math.sin(angulo_rad)
        )

        # Seta Laranja/Branca para indicar propagação/movimento
        cv2.arrowedLine(
            img,
            (cX, cY),
            (fim_X, fim_Y),
            (0, 165, 255),  # Laranja destacado
            4,
            tipLength=0.25
        )

    # ---------------------------------------------------------------
    # Textos informativos no painel
    # ---------------------------------------------------------------
    dano_id = det_id.split("#")[-1]

    # 1. ID do Dano
    adicionar_texto(
        img,
        det_id,
        (cX - 90, cY - 15)
    )

    # 2. Ângulo de Orientação (Forma)
    if dados_predicao is not None:
        texto_orientacao = f"ORIENT: {dados_predicao['orientation_deg']:.1f} deg"
    else:
        texto_orientacao = "ORIENT: N/A"

    adicionar_texto(
        img,
        texto_orientacao,
        (cX - 90, cY + 10)
    )

    # 3. Ângulo de Direção (Movimento)
    if dados_predicao is not None:
        texto_direcao = f"MOV: {dados_predicao['movement_angle']:.1f} deg"
    else:
        texto_direcao = "MOV: N/A"

    adicionar_texto(
        img,
        texto_direcao,
        (cX - 90, cY + 35)
    )


# =====================================================================
# 8. DESENHAR IMAGEM COMPLETA
# =====================================================================
def desenhar_imagem(
    caminho_imagem,
    danos,
    predicoes,
    caminho_saida
):

    img = cv2.imread(
        caminho_imagem
    )

    if img is None:

        print(
            f"[ERRO] Não foi possível abrir: "
            f"{caminho_imagem}"
        )

        return

    for dano in danos:

        # Processar apenas corrosão
        if dano.get("classe_nome", "").lower() != "corrosion":
            continue

        det_id = dano["det_id"]
        dados_predicao = predicoes.get(det_id)

        desenhar_dano(
            img,
            dano,
            dados_predicao
        )

    os.makedirs(
        os.path.dirname(
            caminho_saida
        ),
        exist_ok=True
    )

    cv2.imwrite(
        caminho_saida,
        img
    )


# =====================================================================
# 9. CARREGAR MODELO
# =====================================================================
def carregar_modelo(
    caminho_modelo
):

    print(
        "\nCarregando modelo..."
    )

    pacote = joblib.load(
        caminho_modelo
    )

    if (
        isinstance(
            pacote,
            dict
        )
        and
        "model" in pacote
    ):

        modelo = pacote[
            "model"
        ]

        metadata = pacote.get(
            "metadata",
            {}
        )

    else:

        modelo = pacote

        metadata = {}

    print(
        "Modelo carregado."
    )

    print(
        f"Metadata: {metadata}"
    )

    return modelo


# =====================================================================
# 10. MAIN
# =====================================================================
if __name__ == "__main__":

    # ================================================================
    # CAMINHOS
    # ================================================================
    caminho_modelo = (
        "model/orientation_model.pkl"
    )

    caminho_csv = (
        "dataset/Coords_data_test.csv"
    )

    caminho_json = (
        "contornos_segmentados.json"
    )

    pasta_imagens = (
        "previews_degradacao_temporal/"
        "metricas_anomalias_faces_malha_degradacao_t20_10a_00m/"
        "overlay"
    )

    pasta_resultados = (
        "result_image"
    )

    # ================================================================
    # 1. CARREGAR JSON PRIMEIRO
    # ================================================================
    danos_por_figura = carregar_danos_json(
        caminho_json
    )

    # ================================================================
    # 2. RECONSTRUIR MAPEAMENTO DO LABEL ENCODER
    # ================================================================
    encoded_to_original = (
        construir_mapa_label_encoder(
            danos_por_figura
        )
    )

    # ================================================================
    # 3. CARREGAR MODELO
    # ================================================================
    modelo = carregar_modelo(
        caminho_modelo
    )

    # ================================================================
    # 4. CSV
    # ================================================================
    print(
        "\nCarregando CSV..."
    )

    df_test = pd.read_csv(
        caminho_csv
    )

    print(
        f"Linhas: {len(df_test)}"
    )

    # ================================================================
    # 5. FEATURES PARA O MODELO (Atualizado para ignorar os 4 alvos)
    # ================================================================
    colunas_target = [
        "sin_2theta",
        "cos_2theta",
        "movement_sin",
        "movement_cos",
        "movement_angle"
    ]

    colunas_features = [
        c
        for c in df_test.columns
        if c not in colunas_target
    ]

    X_test = df_test[
        colunas_features
    ]

    # ================================================================
    # 6. PREVISÃO (Agora recolhe as 4 saídas do MultiOutput)
    # ================================================================
    print(
        "\nExecutando previsões..."
    )

    y_pred = modelo.predict(
        X_test
    )

    # 1. Forma (Orientação)
    sin_form = y_pred[:, 0]
    cos_form = y_pred[:, 1]
    pred_orientation = (np.degrees(0.5 * np.arctan2(sin_form, cos_form)) + 180) % 180

    # 2. Direção de Movimento
    sin_mov = y_pred[:, 2]
    cos_mov = y_pred[:, 3]
    pred_direction = np.degrees(np.arctan2(sin_mov, cos_mov))

    df_test["angulo_predito_orientacao"] = pred_orientation
    df_test["angulo_predito_direcao"] = pred_direction

    # ================================================================
    # 7. CRIAR MAPA DE PREVISÕES
    # ================================================================
    predicoes = {}

    for _, row in df_test.iterrows():

        try:
            figura_encoded = int(float(row["Figura_name"]))
        except Exception:
            continue

        if figura_encoded not in encoded_to_original:
            continue

        figura_original = encoded_to_original[figura_encoded]
        figura = extrair_numero_figura(figura_original)

        if figura is None:
            continue

        try:
            dano_id = int(float(row["ID_Figura"]))
        except Exception:
            continue

        det_id = normalizar_det_id(f"Figura_{figura}#{dano_id}")

        predicoes[det_id] = {
            "orientation_deg": float(row["angulo_predito_orientacao"]),
            "movement_angle": float(row["angulo_predito_direcao"])
        }

    # ================================================================
    # 8. PROCESSAR TODAS AS IMAGENS DO CSV
    # ================================================================
    figuras_para_processar = set()

    for det_id in predicoes.keys():
        figura = extrair_numero_figura(det_id)
        if figura is not None:
            figuras_para_processar.add(figura)

    # ================================================================
    # 9. GERAR IMAGENS
    # ================================================================
    print("\n========================================")
    print("PROCESSAMENTO")
    print("========================================")

    for figura in sorted(figuras_para_processar):

        danos = danos_por_figura.get(figura, [])

        if not danos:
            continue

        nome_imagem = f"figura_{figura}_overlay.png"
        caminho_entrada = os.path.join(pasta_imagens, nome_imagem)
        caminho_saida = os.path.join(pasta_resultados, f"pred_{nome_imagem}")

        if not os.path.exists(caminho_entrada):
            continue

        print(f"\nFigura_{figura}")
        print(f"  Danos no JSON: {len(danos)}")

        com_previsao = 0

        for dano in danos:
            det_id = dano["det_id"]
            if det_id in predicoes:
                com_previsao += 1
                p = predicoes[det_id]
                print(f"    {det_id} -> Orient: {p['orientation_deg']:.2f}°, Mov: {p['movement_angle']:.2f}°")
            else:
                print(f"    {det_id} -> SEM PREVISÃO")

        print(f"  Previsões: {com_previsao}/{len(danos)}")

        # ------------------------------------------------------------
        # DESENHAR
        # ------------------------------------------------------------
        desenhar_imagem(
            caminho_imagem=caminho_entrada,
            danos=danos,
            predicoes=predicoes,
            caminho_saida=caminho_saida
        )

        print(f"  [OK] {caminho_saida}")

    print("\n========================================")
    print("CONCLUÍDO")
    print("========================================")