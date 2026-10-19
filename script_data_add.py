import os
import pandas as pd
from datetime import datetime
import re

input_dir = './obra'

def extrair_data_de_string(s):
    """
    Procura uma data no formato dd_mm_yyyy ou dd-mm-yyyy.
    """
    padrao = r'(\d{2})[_-](\d{2})[_-](\d{4})'

    match = re.search(padrao, s)

    if match:
        dia, mes, ano = match.groups()

        try:
            return datetime(int(ano), int(mes), int(dia))
        except ValueError:
            pass

    return None


def adicionar_data_em_ficheiros():

    print("Procurando ficheiros...")

    for root, dirs, files in os.walk(input_dir):

        for file in files:

            # Procurar apenas ficheiros Excel .xlsx
            if not file.lower().endswith('.xlsx'):
                continue

            # Procurar ficheiros com "metricas" no nome
            if 'metricas' not in file.lower():
                continue

            file_path = os.path.join(root, file)

            print(f"\nProcessando: {file_path}")

            try:

                # Ler Excel
                df = pd.read_excel(file_path)

                # Verificar se já existe a coluna data
                if 'data' in df.columns:
                    print("  -> Já possui coluna 'data'.")
                    continue

                # Procurar a data no caminho
                data_extraida = extrair_data_de_string(file_path)

                if data_extraida is None:
                    print("  -> Não foi possível encontrar a data.")
                    continue

                # Adicionar data como primeira coluna
                df.insert(
                    0,
                    'data',
                    data_extraida.strftime('%Y-%m-%d')
                )

                # Guardar ficheiro
                df.to_excel(
                    file_path,
                    index=False
                )

                print(
                    f"  -> Data adicionada: "
                    f"{data_extraida.strftime('%Y-%m-%d')}"
                )

            except Exception as e:
                print(f"  -> Erro: {e}")


if __name__ == "__main__":
    adicionar_data_em_ficheiros()