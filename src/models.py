"""
Configuração dos 3 algoritmos e seus grids de hiperparâmetros.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42


def get_models_and_grids():
    """
    Retorna dicionário com modelos e seus grids de hiperparâmetros.

    Estrutura:
    {
        'NomeModelo': {
            'clf': <estimador sklearn>,
            'grid': {
                'param1': [val1, val2, ...],
                'param2': [val1, val2, ...],
            }
        }
    }
    """

    models_grids = {
        'LogisticRegression': {
            'clf': LogisticRegression(
                penalty='l2',
                solver='saga',
                max_iter=1000,
                class_weight=None,
                random_state=RANDOM_STATE
            ),
            'grid': {
                'C': [0.01, 0.1, 1, 10, 100],
            }
        },

        'RandomForest': {
            'clf': RandomForestClassifier(
                class_weight=None,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'grid': {
                'n_estimators': [100, 200, 500],
                'max_depth': [10, 20, None],
                'criterion': ['gini', 'entropy'],
            }
        },

        'XGBoost': {
            'clf': XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            'grid': {
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 6, 9],
                'n_estimators': [100, 300, 500],
            }
        },
    }

    return models_grids


if __name__ == '__main__':
    models = get_models_and_grids()
    print("Modelos e grids de hiperparâmetros:")
    for nome, cfg in models.items():
        print(f"\n{nome}:")
        print(f"  Estimador: {cfg['clf']}")
        print(f"  Grid: {cfg['grid']}")
