
#%%
# ============================================================
# 1. IMPORTAÇÕES
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score


#%%
# ============================================================
# 2. CARREGAR OS DADOS
# ============================================================

train_data = pd.read_csv('dataset_area/final.csv')
test_data = pd.read_csv('dataset_area/final_test.csv')

target = 'Delta_area'
group_col = 'AnomalyID'

cat_features = [
    'AnomalyID',
    'FacadeRoofID',
    'AnomalyType',
    'Figura_name'
]

date_cols = ['data', 'valid_time_']

# Verificar se as colunas necessárias existem
required_cols = [target] + cat_features + date_cols

for col in required_cols:
    if col not in train_data.columns:
        raise ValueError(f"Coluna '{col}' não encontrada no treino.")

for col in required_cols:
    if col not in test_data.columns:
        raise ValueError(f"Coluna '{col}' não encontrada no teste.")

if 'prev_area' not in test_data.columns:
    raise ValueError("A coluna 'prev_area' não existe no teste.")

# Converter datas
for df in [train_data, test_data]:
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Não permitir datas inválidas para a validação temporal
if train_data['data'].isna().any():
    raise ValueError("Existem datas inválidas na coluna 'data' do treino.")

if test_data['data'].isna().any():
    raise ValueError("Existem datas inválidas na coluna 'data' do teste.")

# Ordenar cronologicamente
train_data = train_data.sort_values('data').reset_index(drop=True)
test_data = test_data.sort_values('data').reset_index(drop=True)

# Guardar as datas originais para construir os folds temporais.
# Todas as observações da mesma data ficam no mesmo fold.
train_dates = train_data['data'].copy()

print("=== DIMENSÕES DOS DADOS ===")
print(f"Treino: {train_data.shape}")
print(f"Teste:  {test_data.shape}")

print(f"\nDatas únicas no treino: {train_dates.nunique()}")
print(f"Datas únicas no teste:  {test_data['data'].nunique()}")

# Verificar valores nulos no target
if train_data[target].isna().any():
    raise ValueError("Existem valores nulos no target do treino.")

if test_data[target].isna().any():
    raise ValueError("Existem valores nulos no target do teste.")


#%%
# ============================================================
# 3. PRÉ-PROCESSAMENTO
# ============================================================

# Criar variáveis de ano e mês
for df in [train_data, test_data]:
    for col in date_cols:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month

    # Remover as datas originais
    df.drop(columns=date_cols, inplace=True)

# Garantir que as variáveis categóricas são strings.
# Evita problemas com valores categóricos em falta.
for df in [train_data, test_data]:
    for col in cat_features:
        df[col] = df[col].fillna('__MISSING__').astype(str)

# Separar variáveis preditoras e target
X_train_full = train_data.drop(columns=[target])
y_train_full = train_data[target].copy()

X_test = test_data.drop(columns=[target])
y_test = test_data[target].copy()

# Garantir que treino e teste têm as mesmas features e a mesma ordem
missing_test_cols = set(X_train_full.columns) - set(X_test.columns)
extra_test_cols = set(X_test.columns) - set(X_train_full.columns)

if missing_test_cols or extra_test_cols:
    raise ValueError(
        f"Colunas diferentes entre treino e teste.\n"
        f"Em falta no teste: {missing_test_cols}\n"
        f"Extra no teste: {extra_test_cols}"
    )

X_test = X_test[X_train_full.columns]

# Verificar se as colunas categóricas existem
for col in cat_features:
    if col not in X_train_full.columns:
        raise ValueError(f"Feature categórica '{col}' não encontrada.")

print("\n=== FEATURES ===")
print(f"Número de features: {X_train_full.shape[1]}")
print(f"Features categóricas: {cat_features}")


#%%
# ============================================================
# 4. VALIDAÇÃO TEMPORAL SEM DIVIDIR DATAS REPETIDAS
# ============================================================

# Aplicar TimeSeriesSplit às datas únicas, em vez de às linhas.
# Assim, todas as anomalias observadas na mesma data ficam
# exclusivamente no treino ou na validação de cada fold.

unique_dates = np.array(sorted(train_dates.unique()))

n_splits = 4

if len(unique_dates) <= n_splits:
    raise ValueError(
        f"São necessárias mais de {n_splits} datas únicas "
        f"para executar esta validação temporal."
    )

date_splitter = TimeSeriesSplit(n_splits=n_splits)

cv_splits = []

for train_date_idx, val_date_idx in date_splitter.split(unique_dates):

    fold_train_dates = unique_dates[train_date_idx]
    fold_val_dates = unique_dates[val_date_idx]

    # Obter os índices das linhas correspondentes a cada conjunto
    train_idx = np.flatnonzero(
        train_dates.isin(fold_train_dates).to_numpy()
    )

    val_idx = np.flatnonzero(
        train_dates.isin(fold_val_dates).to_numpy()
    )

    cv_splits.append((train_idx, val_idx))

    print(
        f"Treino: {train_dates.iloc[train_idx].min().date()} "
        f"a {train_dates.iloc[train_idx].max().date()} | "
        f"Validação: {train_dates.iloc[val_idx].min().date()} "
        f"a {train_dates.iloc[val_idx].max().date()} | "
        f"Linhas: {len(train_idx)}/{len(val_idx)}"
    )


#%%
# ============================================================
# 5. FUNÇÃO DE AVALIAÇÃO
# ============================================================

# O target é Delta_area. Se os valores negativos não forem
# fisicamente válidos, aplicamos o mesmo clipping na validação
# e no teste.

def clipped_neg_mae(estimator, X, y):
    predictions = estimator.predict(X)
    predictions = np.clip(predictions, 0, None)

    return -mean_absolute_error(y, predictions)


#%%
# ============================================================
# 6. DEFINIR MODELO E GRELHA DE HIPERPARÂMETROS
# ============================================================

base_model = CatBoostRegressor(
    loss_function='RMSE',
    eval_metric='MAE',
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    thread_count=1
)

param_grid = {
    'iterations': [300, 800, 1300, 1800],
    'learning_rate': [0.03, 0.05, 0.07, 0.09],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [3, 5, 7, 9]
}

print("\n=== CONFIGURAÇÃO DO GRID SEARCH ===")
print(f"Número de combinações: "
      f"{np.prod([len(v) for v in param_grid.values()])}")
print(f"Número de folds: {len(cv_splits)}")
print(f"Número total de avaliações: "
      f"{len(param_grid['iterations']) * len(param_grid['learning_rate']) * len(param_grid['depth']) * len(param_grid['l2_leaf_reg']) * len(cv_splits)}")


#%%
# ============================================================
# 7. EXECUTAR GRID SEARCH
# ============================================================

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring=clipped_neg_mae,
    cv=cv_splits,
    n_jobs=1,
    verbose=2,
    refit=True,
    error_score='raise',
    return_train_score=True
)

print("\n=== A EXECUTAR GRID SEARCH ===")

grid_search.fit(
    X_train_full,
    y_train_full,
    cat_features=cat_features
)

print("\n=== MELHORES HIPERPARÂMETROS ===")
print(grid_search.best_params_)

print("\n=== MELHOR MAE MÉDIO DE VALIDAÇÃO ===")
print(f"MAE: {-grid_search.best_score_:.4f}")


#%%
# ============================================================
# 8. RESULTADOS DE TODAS AS COMBINAÇÕES
# ============================================================

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results['MAE_validacao'] = -cv_results['mean_test_score']
cv_results['MAE_treino'] = -cv_results['mean_train_score']

cv_results = cv_results.sort_values(
    by='MAE_validacao',
    ascending=True
)

columns_to_show = [
    'param_iterations',
    'param_learning_rate',
    'param_depth',
    'param_l2_leaf_reg',
    'MAE_treino',
    'MAE_validacao',
    'std_test_score',
    'rank_test_score'
]

print("\n=== TOP 10 CONFIGURAÇÕES ===")
print(cv_results[columns_to_show].head(10).to_string(index=False))

cv_results.to_csv(
    'grid_search_results.csv',
    index=False
)


#%%
# ============================================================
# 9. MODELO FINAL
# ============================================================

# O GridSearchCV treina automaticamente o melhor modelo
# com todos os dados de treino, porque refit=True.

final_model = grid_search.best_estimator_

print("\n=== MODELO FINAL ===")
print(final_model.get_params())


#%%
# ============================================================
# 10. AVALIAÇÃO NO CONJUNTO DE TESTE
# ============================================================

test_preds_raw = final_model.predict(X_test)

# Restringir a previsão a valores não negativos
test_preds = np.clip(test_preds_raw, 0, None)

test_mae = mean_absolute_error(y_test, test_preds)
test_r2 = r2_score(y_test, test_preds)

print("\n=== TESTE FINAL ===")
print(f"MAE: {test_mae:.4f}")
print(f"R²:  {test_r2:.4f}")

print("\n=== COMPARAÇÃO COM A VALIDAÇÃO ===")
print(f"MAE validação: {-grid_search.best_score_:.4f}")
print(f"MAE teste:     {test_mae:.4f}")


#%%
# ============================================================
# 11. COMPARAÇÃO DAS ÁREAS
# ============================================================

comparacao = pd.DataFrame({
    'AnomalyID': test_data['AnomalyID'].values,
    'Area': test_data['prev_area'].values,
    'Delta_Real': y_test.values,
    'Delta_Previsto': test_preds,
    'Erro_Abs_Delta': np.abs(y_test.values - test_preds),

    'Area_Final_Real': (
        test_data['prev_area'].values + y_test.values
    ),

    'Area_Final_Prevista': (
        test_data['prev_area'].values + test_preds
    )
})

print("\n=== COMPARAÇÃO DAS PREVISÕES ===")
print(comparacao.head(1000).to_string(index=False))

comparacao.to_csv(
    'Result_final.csv',
    index=False
)


#%%
# ============================================================
# 12. SOMAS GLOBAIS DAS ÁREAS
# ============================================================

area_total_final_prevista = comparacao[
    'Area_Final_Prevista'
].sum()

area_total_final_real = comparacao[
    'Area_Final_Real'
].sum()

erro_area_total = (
    area_total_final_prevista - area_total_final_real
)

print("\n=== ÁREA FINAL ABSOLUTA ACUMULADA ===")
print(
    f"Área total final prevista: "
    f"{area_total_final_prevista:.4f} cm²"
)

print(
    f"Área total final real:     "
    f"{area_total_final_real:.4f} cm²"
)

print(
    f"Diferença prevista - real: "
    f"{erro_area_total:.4f} cm²"
)


#%%
# ============================================================
# 13. GRÁFICO: VALORES REAIS VS. PREVISTOS
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    test_preds,
    alpha=0.7
)

min_value = min(y_test.min(), test_preds.min())
max_value = max(y_test.max(), test_preds.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle='--'
)

plt.xlabel('Delta_area real')
plt.ylabel('Delta_area previsto')
plt.title('CatBoost com GridSearchCV: Real vs. Previsto')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plts/real_vs_previsto.png', dpi=300)
plt.show()


#%%
# ============================================================
# 14. GRÁFICO: ERRO ABSOLUTO
# ============================================================

plt.figure(figsize=(10, 5))

plt.hist(
    comparacao['Erro_Abs_Delta'],
    bins=20
)

plt.xlabel('Erro absoluto de Delta_area')
plt.ylabel('Frequência')
plt.title('Distribuição do erro absoluto no teste')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plts/distribuicao_erros.png', dpi=300)
plt.show()


#%%
# ============================================================
# 15. GUARDAR O MODELO E OS RESULTADOS
# ============================================================

os.makedirs('model', exist_ok=True)

joblib.dump(
    final_model,
    'model/area_model.pkl'
)

print("\n=== FICHEIROS GUARDADOS ===")
print("Modelo: model/area_model.pkl")
print("Resultados: Result_final.csv")
print("Grid Search: grid_search_results.csv")
print("Gráfico real vs. previsto: real_vs_previsto.png")
print("Gráfico de erros: distribuicao_erros.png")
