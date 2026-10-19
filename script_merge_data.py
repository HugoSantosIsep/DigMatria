import os
import pandas as pd

# Configurações
input_dir = './obra'
output_file = 'consolidado_metricas_2016_2025.csv'

# Inclui 2016 até 2025
anos_interesse = list(range(2016, 2026))


def consolidar_dados():

    lista_dfs = []

    print("Iniciando a consolidação...")
    print(f"Anos considerados: {anos_interesse}")
    print()

    for root, dirs, files in os.walk(input_dir):

        for file in files:

            # Apenas ficheiros .xlsx com "metricas" no nome
            if not file.lower().endswith('.xlsx'):
                continue

            if 'metricas' not in file.lower():
                continue

            file_path = os.path.join(root, file)

            try:

                # Ler Excel
                df = pd.read_excel(file_path)

                # Verificar coluna data
                if 'data' not in df.columns:
                    print(f"Sem coluna 'data': {file_path}")
                    continue

                # Converter data
                df['data'] = pd.to_datetime(
                    df['data'],
                    errors='coerce'
                )

                # Remover datas inválidas
                df = df.dropna(subset=['data'])

                # Filtrar anos
                df = df[
                    df['data'].dt.year.isin(anos_interesse)
                ]

                if not df.empty:

                    lista_dfs.append(df)

                    print(
                        f"Adicionado: {file_path} "
                        f"({len(df)} linhas)"
                    )

            except Exception as e:

                print(
                    f"Erro ao ler {file_path}: {e}"
                )

    # ---------------------------------------------------------
    # CONSOLIDAÇÃO
    # ---------------------------------------------------------

    if lista_dfs:

        df_final = pd.concat(
            lista_dfs,
            ignore_index=True
        )

        # Ordenar por data
        df_final = df_final.sort_values(
            by='data'
        ).reset_index(drop=True)

        # Guardar CSV
        df_final.to_csv(
            output_file,
            index=False
        )

        print()
        print("=" * 60)
        print("SUCESSO")
        print("=" * 60)

        print(
            f"Ficheiro consolidado: {output_file}"
        )

        print(
            f"Total de linhas: {len(df_final)}"
        )

        print(
            f"Total de colunas: {len(df_final.columns)}"
        )

        print(
            f"Data inicial: {df_final['data'].min()}"
        )

        print(
            f"Data final: {df_final['data'].max()}"
        )

        print("=" * 60)

    else:

        print()
        print(
            "Nenhum dado encontrado "
            "para os anos especificados."
        )


if __name__ == "__main__":
    consolidar_dados()