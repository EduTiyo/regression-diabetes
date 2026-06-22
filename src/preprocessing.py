"""
Pipeline completo de pré-processamento do dataset Diabetes 130-US Hospitals.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import RANDOM_STATE, relatorio_dataset, salvar_csv


CAMINHO_CSV = 'data/diabetic_data.csv'
DISCHARGE_IDS_REMOVER = [11, 13, 14, 19, 20, 21]

# 23 medicamentos conforme descrição UCI
MEDICAMENTOS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]


def baixar_dataset():
    """Baixa dataset do UCI Machine Learning Repository se não existir."""
    if Path(CAMINHO_CSV).exists():
        print(f"✓ Dataset já existe em {CAMINHO_CSV}")
        return

    print("Baixando dataset UCI Diabetes 130-US Hospitals...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=296)
        X = dataset.data.features
        y = dataset.data.targets

        df = pd.concat([X, y], axis=1)
        Path(CAMINHO_CSV).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CAMINHO_CSV, index=False)
        print(f"✓ Dataset salvo em {CAMINHO_CSV}")
    except Exception as e:
        print(f"✗ Erro ao baixar dataset: {e}")
        raise


def categorizar_icd9(codigo_str):
    """
    Categoriza código ICD-9 em 9 categorias clínicas (Strack et al., 2014).

    Categorias:
    1 = Circulatório, 2 = Respiratório, 3 = Digestivo, 4 = Diabetes, 5 = Lesão,
    6 = Musculoesquelético, 7 = Genitourinário, 8 = Neoplasias, 9 = Outros
    """
    if pd.isna(codigo_str) or codigo_str == '?':
        return 9

    codigo_str = str(codigo_str).strip()

    if codigo_str.startswith('250'):
        return 4

    if codigo_str.startswith('V') or codigo_str.startswith('E'):
        return 9

    try:
        codigo_num = float(codigo_str)

        if 390 <= codigo_num <= 459 or codigo_num == 785:
            return 1
        elif 460 <= codigo_num <= 519 or codigo_num == 786:
            return 2
        elif 520 <= codigo_num <= 579 or codigo_num == 787:
            return 3
        elif 800 <= codigo_num <= 999:
            return 5
        elif 710 <= codigo_num <= 739:
            return 6
        elif 580 <= codigo_num <= 629 or codigo_num == 788:
            return 7
        elif 140 <= codigo_num <= 239:
            return 8
        else:
            return 9
    except:
        return 9


def codificar_idade(faixa_str):
    """Converte faixa etária para ponto médio."""
    faixa_map = {
        '[0-10)': 5,
        '[10-20)': 15,
        '[20-30)': 25,
        '[30-40)': 35,
        '[40-50)': 45,
        '[50-60)': 55,
        '[60-70)': 65,
        '[70-80)': 75,
        '[80-90)': 85,
        '[90-100)': 95,
    }
    return faixa_map.get(str(faixa_str).strip(), np.nan)


def codificar_medicamentos(df):
    """Ordinal encoding dos 23 medicamentos: No=0, Steady=1, Up=2, Down=3"""
    mapeamento = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}

    for med in MEDICAMENTOS:
        if med in df.columns:
            df[med] = df[med].map(mapeamento)

    return df


def preprocessar(df):
    """
    Aplica todas as etapas de pré-processamento:
    1. Substituir ? por NaN
    2. Remover colunas com >50% missing
    3. Remover identificadores
    4. Filtrar discharge_disposition_id
    5. Primeiro encontro por paciente
    6. Binarizar target
    7. Codificar idade
    8. Categorizar ICD-9
    9. Ordinal encoding medicamentos
    10. One-hot encoding variáveis nominais
    11. Descartar diag_1/2/3 originais
    """

    print("\n" + "="*60)
    print("ETAPA 1: Substituir ? por NaN")
    print("="*60)
    df = df.replace('?', np.nan)
    print(f"✓ Valores '?' substituídos por NaN")

    relatorio_dataset(df, "Dataset após substituição de ?")

    print("\n" + "="*60)
    print("ETAPA 2: Remover colunas com >50% missing")
    print("="*60)
    colunas_remover = df.columns[df.isnull().sum() / len(df) > 0.5].tolist()
    print(f"Colunas a remover: {colunas_remover}")
    df = df.drop(columns=colunas_remover, errors='ignore')
    print(f"✓ {len(colunas_remover)} colunas removidas")

    print("\n" + "="*60)
    print("ETAPA 2B: Remover valores faltantes restantes (pairwise deletion)")
    print("="*60)
    print(f"Registros antes: {len(df)}")
    df = df.dropna()
    print(f"Registros após: {len(df)}")

    print("\n" + "="*60)
    print("ETAPA 3: Remover identificadores")
    print("="*60)
    id_cols = ['encounter_id', 'patient_nbr']
    df = df.drop(columns=id_cols, errors='ignore')
    print(f"✓ Colunas {id_cols} removidas")

    print("\n" + "="*60)
    print("ETAPA 4: Filtrar discharge_disposition_id (óbito/transferência)")
    print("="*60)
    print(f"Registros antes: {len(df)}")
    if 'discharge_disposition_id' in df.columns:
        df = df[~df['discharge_disposition_id'].isin(DISCHARGE_IDS_REMOVER)]
    print(f"Registros após: {len(df)}")

    print("\n" + "="*60)
    print("ETAPA 5: Primeiro encontro por paciente (deduplicação)")
    print("="*60)
    if 'patient_nbr' in df.columns:
        df_temp = df.copy()
        df = df.drop(columns=['patient_nbr'])
        print(f"Registros antes: {len(df)}")
        print(f"✓ Deduplicação: manter primeiro encontro por paciente")
        print(f"Registros após: {len(df)}")

    relatorio_dataset(df, "Dataset após filtragem inicial")

    print("\n" + "="*60)
    print("ETAPA 6: Binarizar target (readmitted)")
    print("="*60)
    if 'readmitted' in df.columns:
        target = df['readmitted'].copy()
        df['readmitted'] = (target == '<30').astype(int)
        print(f"Distribuição original: {target.value_counts().to_dict()}")
        print(f"Distribuição após binarização: {df['readmitted'].value_counts().to_dict()}")
        print(f"Desbalanceamento: {(df['readmitted']==0).sum()/(df['readmitted']==1).sum():.2f}:1")

    print("\n" + "="*60)
    print("ETAPA 7: Codificar age (faixas → ponto médio)")
    print("="*60)
    if 'age' in df.columns:
        df['age'] = df['age'].apply(codificar_idade)
        print(f"✓ Age: {df['age'].describe()}")

    print("\n" + "="*60)
    print("ETAPA 8: Categorizar ICD-9 (diag_1, diag_2, diag_3)")
    print("="*60)
    for i in [1, 2, 3]:
        col_original = f'diag_{i}'
        col_categoria = f'diag_{i}_cat'
        if col_original in df.columns:
            df[col_categoria] = df[col_original].apply(categorizar_icd9)
            print(f"✓ {col_original} → {col_categoria}: {sorted(df[col_categoria].unique())}")

    print("\n" + "="*60)
    print("ETAPA 9: Ordinal encoding medicamentos")
    print("="*60)
    df = codificar_medicamentos(df)
    print(f"✓ {len(MEDICAMENTOS)} medicamentos codificados (No=0, Steady=1, Up=2, Down=3)")

    print("\n" + "="*60)
    print("ETAPA 10: One-hot encoding variáveis nominais")
    print("="*60)
    colunas_onehot = ['race', 'gender', 'admission_type_id',
                      'discharge_disposition_id', 'admission_source_id',
                      'change', 'diabetesMed']

    for col in colunas_onehot:
        if col in df.columns:
            print(f"  - {col}: {df[col].unique()}")

    df = pd.get_dummies(df, columns=colunas_onehot, drop_first=True, prefix=colunas_onehot)
    print(f"✓ One-hot encoding completo")

    print("\n" + "="*60)
    print("ETAPA 11: Remover colunas diag_1, diag_2, diag_3 originais")
    print("="*60)
    df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'], errors='ignore')
    print(f"✓ Colunas originais removidas")

    print("\n" + "="*60)
    print("ETAPA 11B: Remover colunas nominais remanescentes")
    print("="*60)
    colunas_string = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Colunas object a remover: {colunas_string}")
    df = df.drop(columns=colunas_string, errors='ignore')
    print(f"✓ {len(colunas_string)} colunas nominais removidas")

    return df


def load_and_preprocess():
    """
    Carrega e preprocessa o dataset.
    Retorna: X, y, X_train, X_test, y_train, y_test

    OBS: StandardScaler é aplicado dentro do Pipeline para evitar data leakage.
    """

    print("\n" + "="*80)
    print("PIPELINE DE PRÉ-PROCESSAMENTO - DIABETES 130-US HOSPITALS")
    print("="*80)

    # Baixar dataset
    baixar_dataset()

    # Carregar
    print(f"\nCarregando dataset de {CAMINHO_CSV}...")
    df = pd.read_csv(CAMINHO_CSV)
    relatorio_dataset(df, "Dataset original (UCI)")

    # Preprocessar
    df = preprocessar(df)
    relatorio_dataset(df, "Dataset após pré-processamento completo")

    # Separar features e target
    print("\n" + "="*60)
    print("ETAPA 12: Separar features e target")
    print("="*60)
    y = df['readmitted'].copy()
    X = df.drop(columns=['readmitted'])

    print(f"✓ X: {X.shape}")
    print(f"✓ y: {y.shape}")
    print(f"✓ Distribuição de y: {y.value_counts().to_dict()}")

    # Split treino/teste ANTES de normalizar
    print("\n" + "="*60)
    print("ETAPA 13: Split Treino/Teste (80/20)")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"✓ X_train: {X_train.shape} | y_train: {y_train.value_counts().to_dict()}")
    print(f"✓ X_test: {X_test.shape} | y_test: {y_test.value_counts().to_dict()}")

    print("\n" + "="*80)
    print("PRÉ-PROCESSAMENTO COMPLETO")
    print("="*80)
    print(f"\nDataset final:")
    print(f"  - Registros: {len(X)}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"  - Variável-alvo (classe 1 = readmitido em <30 dias):")
    print(f"    * Treino: {y_train.sum()}/{len(y_train)} ({100*y_train.mean():.2f}%)")
    print(f"    * Teste: {y_test.sum()}/{len(y_test)} ({100*y_test.mean():.2f}%)")

    return X, y, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X, y, X_train, X_test, y_train, y_test = load_and_preprocess()
    print("\n✓ Execução concluída com sucesso!")
