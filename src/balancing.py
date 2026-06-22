"""
Estratégias de balanceamento de classes para o experimento.
"""

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN


RANDOM_STATE = 42


def get_balancing_strategies():
    """
    Retorna dicionário com as 5 estratégias de balanceamento:
    1. Baseline (sem balanceamento)
    2. SMOTE
    3. ADASYN
    4. BorderlineSMOTE
    5. SMOTEENN
    """

    strategies = {
        'Baseline': None,
        'SMOTE': SMOTE(k_neighbors=5, random_state=RANDOM_STATE),
        'ADASYN': ADASYN(n_neighbors=5, random_state=RANDOM_STATE),
        'BorderlineSMOTE': BorderlineSMOTE(k_neighbors=5, kind='borderline-1', random_state=RANDOM_STATE),
        'SMOTEENN': SMOTEENN(random_state=RANDOM_STATE),
    }

    return strategies


if __name__ == '__main__':
    strategies = get_balancing_strategies()
    print("Estratégias de balanceamento disponíveis:")
    for nome, estrategia in strategies.items():
        print(f"  - {nome}: {estrategia}")
