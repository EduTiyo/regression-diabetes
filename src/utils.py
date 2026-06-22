"""
Funções auxiliares para o pipeline experimental.
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path


RANDOM_STATE = 42


def set_random_seeds(seed=RANDOM_STATE):
    """Fixa seeds aleatórias para reproducibilidade."""
    np.random.seed(seed)
    random.seed(seed)


def criar_diretorios():
    """Cria estrutura de diretórios necessária se não existir."""
    diretorios = [
        'data',
        'results/tables',
        'results/figures',
        'notebooks'
    ]
    for diretorio in diretorios:
        Path(diretorio).mkdir(parents=True, exist_ok=True)
        print(f"✓ Diretório '{diretorio}' verificado/criado")


def salvar_csv(df, caminho, index=False):
    """Salva DataFrame em CSV com log."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=index)
    print(f"✓ Salvo: {caminho} ({len(df)} linhas, {len(df.columns)} colunas)")


def carregar_csv(caminho):
    """Carrega CSV com log."""
    df = pd.read_csv(caminho)
    print(f"✓ Carregado: {caminho} ({len(df)} linhas, {len(df.columns)} colunas)")
    return df


def relatorio_dataset(df, titulo="Dataset"):
    """Imprime relatório descritivo do dataset."""
    print(f"\n{'='*60}")
    print(f"{titulo}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Colunas: {list(df.columns)}")
    print(f"\nMissing values (%):")
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print("Nenhum valor faltante!")
    print(f"\nTipos de dados:\n{df.dtypes}")
    print(f"{'='*60}\n")
