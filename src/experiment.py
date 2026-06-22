"""
Loop principal do experimento fatorial.
Design: 3 algoritmos × 5 estratégias de balanceamento = 15 combinações.
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from imblearn.pipeline import Pipeline as ImbPipeline

from preprocessing import load_and_preprocess, RANDOM_STATE
from balancing import get_balancing_strategies
from models import get_models_and_grids
from utils import salvar_csv

warnings.filterwarnings('ignore')


def criar_pipeline(sampler, clf):
    """Cria pipeline imblearn com scaler, sampler opcional e classificador."""
    steps = [('scaler', StandardScaler())]

    if sampler is not None:
        steps.append(('sampler', sampler))

    steps.append(('clf', clf))

    return ImbPipeline(steps)


def prefixar_grid(grid, prefix='clf__'):
    """Prefixar parâmetros do grid para funcionar dentro do pipeline."""
    return {f'{prefix}{k}': v for k, v in grid.items()}


def executar_experimento():
    """Executa o loop fatorial completo do experimento."""

    print("\n" + "="*80)
    print("EXPERIMENTO FATORIAL - PREDIÇÃO DE READMISSÃO")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random State: {RANDOM_STATE}")

    # Carregar e preprocessar dados
    print("\n" + "-"*80)
    print("CARREGANDO E PREPROCESSANDO DADOS")
    print("-"*80)
    X, y, X_train, X_test, y_train, y_test = load_and_preprocess()

    # Carregar estratégias e modelos
    balancing_strategies = get_balancing_strategies()
    models_grids = get_models_and_grids()

    print("\n" + "-"*80)
    print("CONFIGURAÇÃO DO EXPERIMENTO")
    print("-"*80)
    print(f"Estratégias de balanceamento: {list(balancing_strategies.keys())}")
    print(f"Modelos: {list(models_grids.keys())}")
    print(f"Total de combinações: {len(balancing_strategies) * len(models_grids)}")
    print(f"CV interno (grid search): 5 folds estratificados")
    print(f"CV final (avaliação): 10 folds estratificados")

    # Lista para coletar resultados de cada fold
    resultados_folds = []

    combinacao_idx = 0
    total_combinacoes = len(balancing_strategies) * len(models_grids)

    # Loop fatorial
    for nome_balanceamento, sampler in balancing_strategies.items():

        for nome_modelo, cfg_modelo in models_grids.items():
            combinacao_idx += 1

            print("\n" + "="*80)
            print(f"COMBINAÇÃO {combinacao_idx}/{total_combinacoes}")
            print(f"Balanceamento: {nome_balanceamento} | Modelo: {nome_modelo}")
            print("="*80)

            try:
                # Criar pipeline
                clf_base = cfg_modelo['clf']
                param_grid = cfg_modelo['grid']

                pipe = criar_pipeline(sampler, clf_base)
                grid_prefixado = prefixar_grid(param_grid)

                # Grid Search no treino (5 folds)
                print(f"\n[1/2] Grid Search no treino (5 folds)...")
                cv_interno = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

                gs = GridSearchCV(
                    pipe,
                    grid_prefixado,
                    cv=cv_interno,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0
                )

                gs.fit(X_train, y_train)

                print(f"  Melhor F1 (grid search): {gs.best_score_:.4f}")
                print(f"  Melhores hiperparâmetros: {gs.best_params_}")

                # Avaliação final no dataset completo (10 folds)
                print(f"\n[2/2] Avaliação final no dataset completo (10 folds)...")
                cv_final = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

                metricas = ['f1', 'roc_auc', 'recall', 'precision', 'matthews_corrcoef']
                cv_results = cross_validate(
                    gs.best_estimator_,
                    X,
                    y,
                    cv=cv_final,
                    scoring=metricas,
                    return_train_score=False,
                    n_jobs=-1
                )

                # Processar resultados por fold
                n_folds = 10
                for fold_idx in range(n_folds):
                    resultado_fold = {
                        'balanceamento': nome_balanceamento,
                        'modelo': nome_modelo,
                        'fold': fold_idx + 1,
                        'f1': cv_results['test_f1'][fold_idx],
                        'roc_auc': cv_results['test_roc_auc'][fold_idx],
                        'recall': cv_results['test_recall'][fold_idx],
                        'precision': cv_results['test_precision'][fold_idx],
                        'mcc': cv_results['test_matthews_corrcoef'][fold_idx],
                    }
                    resultados_folds.append(resultado_fold)

                # Resumo da combinação
                print(f"\n  Resultados (10 folds):")
                print(f"    F1: {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
                print(f"    AUC-ROC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
                print(f"    Recall: {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
                print(f"    Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
                print(f"    MCC: {cv_results['test_matthews_corrcoef'].mean():.4f} ± {cv_results['test_matthews_corrcoef'].std():.4f}")

            except Exception as e:
                print(f"\n✗ ERRO na combinação {combinacao_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Salvar resultados
    print("\n" + "="*80)
    print("SALVANDO RESULTADOS")
    print("="*80)

    df_resultados = pd.DataFrame(resultados_folds)
    caminho_csv = 'results/tables/scores_por_fold.csv'
    salvar_csv(df_resultados, caminho_csv, index=False)

    print(f"\nTotais:")
    print(f"  - Resultados coletados: {len(df_resultados)} (10 folds × {total_combinacoes} combinações)")
    print(f"  - Arquivo: {caminho_csv}")

    print("\n" + "="*80)
    print("EXPERIMENTO CONCLUÍDO")
    print("="*80)
    print(f"Timestamp final: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return df_resultados


if __name__ == '__main__':
    df_resultados = executar_experimento()
    print("\n✓ Experimento executado com sucesso!")
