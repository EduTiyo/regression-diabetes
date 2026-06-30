"""
Análise de importância de atributos (feature importance).

Treina UMA combinação (por padrão a melhor: SMOTEENN + XGBoost) e calcula a
relevância dos atributos por dois métodos complementares:

  1. Importância por GANHO (gain) nativa do XGBoost — quanto cada feature
     contribui para reduzir a impureza nas divisões das árvores.
  2. PERMUTATION IMPORTANCE — embaralha cada feature e mede a queda no F1 sobre
     o conjunto de teste. Mais confiável (mede impacto preditivo real e funciona
     para qualquer modelo).

NÃO reexecuta o experimento fatorial completo. Apenas:
  - carrega os dados (CSV em cache, ~segundos)
  - treina 1 modelo (com Grid Search só dessa combinação, salvo --no-grid)
  - extrai e salva as importâncias

Uso:
    cd src
    python feature_importance.py                      # SMOTEENN + XGBoost (padrão)
    python feature_importance.py --no-grid            # pula grid search (mais rápido)
    python feature_importance.py --balanceamento SMOTE --modelo RandomForest
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline

from preprocessing import load_and_preprocess, RANDOM_STATE
from balancing import get_balancing_strategies
from models import get_models_and_grids
from utils import salvar_csv

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

TOP_N = 20  # quantos atributos mostrar nos gráficos


def construir_pipeline(sampler, clf):
    """Pipeline imblearn: scaler -> (sampler opcional) -> classificador."""
    steps = [('scaler', StandardScaler())]
    if sampler is not None:
        steps.append(('sampler', sampler))
    steps.append(('clf', clf))
    return ImbPipeline(steps)


def treinar_modelo(X_train, y_train, balanceamento, modelo, usar_grid=True):
    """
    Treina a combinação escolhida no conjunto de treino (80%).

    Se usar_grid=True, faz Grid Search (5-fold, scoring='f1') APENAS para essa
    combinação para obter os melhores hiperparâmetros. Caso contrário usa os
    valores padrão do estimador.
    """
    estrategias = get_balancing_strategies()
    modelos = get_models_and_grids()

    if balanceamento not in estrategias:
        raise ValueError(f"Balanceamento inválido: {balanceamento}. Opções: {list(estrategias)}")
    if modelo not in modelos:
        raise ValueError(f"Modelo inválido: {modelo}. Opções: {list(modelos)}")

    sampler = estrategias[balanceamento]
    clf = modelos[modelo]['clf']
    grid = modelos[modelo]['grid']

    pipe = construir_pipeline(sampler, clf)

    if usar_grid:
        print(f"\n[Grid Search] Otimizando {modelo} ({balanceamento}) — 5-fold, scoring=f1...")
        grid_prefixado = {f'clf__{k}': v for k, v in grid.items()}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(pipe, grid_prefixado, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        print(f"  Melhor F1 (CV interno): {gs.best_score_:.4f}")
        print(f"  Melhores hiperparâmetros: {gs.best_params_}")
        return gs.best_estimator_
    else:
        print(f"\n[Treino] Ajustando {modelo} ({balanceamento}) com hiperparâmetros padrão...")
        pipe.fit(X_train, y_train)
        return pipe


def importancia_por_ganho(pipe, feature_names):
    """Importância nativa do classificador (gain p/ XGBoost, impureza p/ RF)."""
    clf = pipe.named_steps['clf']
    if not hasattr(clf, 'feature_importances_'):
        print("  ⚠ Classificador não expõe feature_importances_ (ex.: LogisticRegression).")
        # Para modelos lineares, usar |coeficiente| como proxy
        if hasattr(clf, 'coef_'):
            valores = np.abs(clf.coef_).ravel()
        else:
            return None
    else:
        valores = clf.feature_importances_

    df = pd.DataFrame({'feature': feature_names, 'importancia_ganho': valores})
    df = df.sort_values('importancia_ganho', ascending=False).reset_index(drop=True)
    return df


def importancia_por_permutacao(pipe, X_test, y_test, feature_names):
    """Permutation importance sobre o teste, medindo queda no F1."""
    print("\n[Permutation Importance] Calculando sobre o conjunto de teste (scoring=f1)...")
    result = permutation_importance(
        pipe, X_test, y_test,
        scoring='f1',
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    df = pd.DataFrame({
        'feature': feature_names,
        'importancia_perm_media': result.importances_mean,
        'importancia_perm_std': result.importances_std,
    })
    df = df.sort_values('importancia_perm_media', ascending=False).reset_index(drop=True)
    return df


def plot_barras(df, coluna_valor, titulo, caminho, coluna_erro=None):
    """Gráfico de barras horizontais das top-N features."""
    top = df.head(TOP_N).iloc[::-1]  # inverter p/ maior no topo

    fig, ax = plt.subplots(figsize=(10, 8))
    erro = top[coluna_erro] if coluna_erro else None
    ax.barh(top['feature'], top[coluna_valor], xerr=erro,
            color='steelblue', edgecolor='black', capsize=3, alpha=0.85)
    ax.set_xlabel(titulo, fontsize=12, fontweight='bold')
    ax.set_title(f'Top {TOP_N} Atributos — {titulo}', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figura salva: {caminho}")


def main():
    parser = argparse.ArgumentParser(description='Análise de importância de atributos.')
    parser.add_argument('--balanceamento', default='SMOTEENN',
                        help='Estratégia de balanceamento (padrão: SMOTEENN)')
    parser.add_argument('--modelo', default='XGBoost',
                        help='Modelo (padrão: XGBoost)')
    parser.add_argument('--no-grid', action='store_true',
                        help='Pula o Grid Search e usa hiperparâmetros padrão (mais rápido)')
    args = parser.parse_args()

    print("\n" + "█" * 80)
    print(f"IMPORTÂNCIA DE ATRIBUTOS — {args.balanceamento} + {args.modelo}")
    print("█" * 80)

    # 1. Carregar dados (rápido: CSV em cache)
    X, y, X_train, X_test, y_train, y_test = load_and_preprocess()
    feature_names = list(X.columns)

    # 2. Treinar a combinação escolhida
    pipe = treinar_modelo(X_train, y_train, args.balanceamento, args.modelo,
                          usar_grid=not args.no_grid)

    sufixo = f"{args.balanceamento}_{args.modelo}".lower()

    # 3. Importância por ganho (nativa)
    print("\n" + "=" * 80)
    print("IMPORTÂNCIA NATIVA (GANHO / IMPUREZA / |COEF|)")
    print("=" * 80)
    df_ganho = importancia_por_ganho(pipe, feature_names)
    if df_ganho is not None:
        print(df_ganho.head(15).to_string(index=False))
        salvar_csv(df_ganho, f'results/tables/feature_importance_ganho_{sufixo}.csv')
        plot_barras(df_ganho, 'importancia_ganho', 'Importância por Ganho (XGBoost)',
                    f'results/figures/feature_importance_ganho_{sufixo}.png')

    # 4. Permutation importance (sobre o teste)
    print("\n" + "=" * 80)
    print("PERMUTATION IMPORTANCE (queda no F1 sobre o teste)")
    print("=" * 80)
    df_perm = importancia_por_permutacao(pipe, X_test, y_test, feature_names)
    print(df_perm.head(15).to_string(index=False))
    salvar_csv(df_perm, f'results/tables/feature_importance_permutacao_{sufixo}.csv')
    plot_barras(df_perm, 'importancia_perm_media', 'Permutation Importance (queda no F1)',
                f'results/figures/feature_importance_permutacao_{sufixo}.png',
                coluna_erro='importancia_perm_std')

    # 5. Tabela combinada (top 15 de cada, com rank)
    print("\n" + "=" * 80)
    print("RESUMO — TOP 15 ATRIBUTOS MAIS RELEVANTES (permutation)")
    print("=" * 80)
    top15 = df_perm.head(15).copy()
    top15.insert(0, 'rank', range(1, len(top15) + 1))
    print(top15.to_string(index=False))
    salvar_csv(top15, f'results/tables/feature_importance_top15_{sufixo}.csv')

    print("\n" + "█" * 80)
    print("ANÁLISE CONCLUÍDA")
    print("█" * 80)


if __name__ == '__main__':
    main()
