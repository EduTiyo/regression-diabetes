"""
Script para testar o pipeline experimental completo com grids reduzidos.
Valida que todo o pipeline funciona end-to-end antes de rodar o experimento completo.
"""

import sys
sys.path.insert(0, 'src')

import os
os.chdir('/Users/eduardotiyo/Documents/diabetes-readmission')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline

from src.preprocessing import load_and_preprocess, RANDOM_STATE
from src.balancing import get_balancing_strategies
from src.models import get_models_and_grids
from src.utils import salvar_csv

print("\n" + "█"*80)
print("TESTE DO PIPELINE EXPERIMENTAL")
print("█"*80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Carregar dados
print("\n[1/3] Carregando dados...")
X, y, X_train, X_test, y_train, y_test = load_and_preprocess()

# Reduzir grids para teste rápido
print("\n[2/3] Configurando grids reduzidos para teste...")
balancing_strategies = get_balancing_strategies()
models_grids_full = get_models_and_grids()

# Simplificar grids
models_grids = {
    'LogisticRegression': {
        'clf': models_grids_full['LogisticRegression']['clf'],
        'grid': {'C': [0.1, 1]}  # Reduzido: 5 → 2
    },
}

print(f"  Modelos: {list(models_grids.keys())}")
print(f"  Estratégias: {list(balancing_strategies.keys())}")
print(f"  Combinações para teste: {len(models_grids) * len(balancing_strategies)} (ao invés de 15)")

# Executar uma combinação de teste
print("\n[3/3] Executando teste (Baseline + LogisticRegression)...")

sampler = None  # Baseline
cfg_modelo = models_grids['LogisticRegression']
clf_base = cfg_modelo['clf']
param_grid = cfg_modelo['grid']

# Pipeline
steps = [('scaler', StandardScaler())]
if sampler:
    steps.append(('sampler', sampler))
steps.append(('clf', clf_base))
pipe = ImbPipeline(steps)

grid_prefixado = {f'clf__{k}': v for k, v in param_grid.items()}

# Grid Search (5 folds)
print("\n  Grid Search...")
cv_interno = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
gs = GridSearchCV(pipe, grid_prefixado, cv=cv_interno, scoring='f1', n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
print(f"  ✓ Grid Search completo")
print(f"    Melhor F1: {gs.best_score_:.4f}")

# Avaliação (10 folds)
print("\n  Avaliação final (10 folds)...")
cv_final = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
metricas = ['f1', 'roc_auc', 'recall', 'precision', 'matthews_corrcoef']

cv_results = cross_validate(
    gs.best_estimator_, X, y,
    cv=cv_final,
    scoring=metricas,
    n_jobs=-1
)

print(f"  ✓ Avaliação completa")
print(f"    F1: {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
print(f"    AUC-ROC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
print(f"    MCC: {cv_results['test_matthews_corrcoef'].mean():.4f} ± {cv_results['test_matthews_corrcoef'].std():.4f}")

# Salvar resultados de teste
print("\n  Salvando resultados de teste...")
resultados = []
for fold_idx in range(10):
    resultados.append({
        'balanceamento': 'Baseline',
        'modelo': 'LogisticRegression',
        'fold': fold_idx + 1,
        'f1': cv_results['test_f1'][fold_idx],
        'roc_auc': cv_results['test_roc_auc'][fold_idx],
        'recall': cv_results['test_recall'][fold_idx],
        'precision': cv_results['test_precision'][fold_idx],
        'mcc': cv_results['test_matthews_corrcoef'][fold_idx],
    })

df_test = pd.DataFrame(resultados)
salvar_csv(df_test, 'results/tables/test_scores.csv', index=False)

print("\n" + "█"*80)
print("✓ TESTE CONCLUÍDO COM SUCESSO!")
print("█"*80)
print("\nPróximas etapas:")
print("  1. Revisar results/tables/test_scores.csv")
print("  2. Se sucesso, rodar: python src/experiment.py")
print("  3. Depois: python src/evaluation.py")
