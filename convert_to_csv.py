import os
import pandas as pd

input_dir = './obra'


def converter_xlsx_para_csv():

    convertidos = 0
    erros = 0

    print("A procurar ficheiros Excel...\n")

    for root, dirs, files in os.walk(input_dir):

        for file in files:

            # Apenas ficheiros XLSX
            if not file.lower().endswith('.xlsx'):
                continue

            # Apenas ficheiros de métricas
            if 'metricas' not in file.lower():
                continue

            file_path = os.path.join(root, file)

            # Nome do CSV
            nome_csv = os.path.splitext(file)[0] + '.csv'
            output_path = os.path.join(root, nome_csv)

            try:

                # Ler Excel
                df = pd.read_excel(file_path)

                # Guardar como CSV
                df.to_csv(
                    output_path,
                    index=False,
                    encoding='utf-8-sig'
                )

                convertidos += 1

                print(f"Convertido:")
                print(f"  XLSX: {file_path}")
                print(f"  CSV : {output_path}\n")

            except Exception as e:

                erros += 1

                print(f"ERRO: {file_path}")
                print(f"      {e}\n")

    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"Ficheiros convertidos: {convertidos}")
    print(f"Erros: {erros}")


if __name__ == "__main__":
    converter_xlsx_para_csv()