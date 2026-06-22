# Predição de Readmissão Hospitalar Precoce em Pacientes Diabéticos

**Artigo científico para PPGCC21 — Reconhecimento de Padrões (UTFPR Campo Mourão)**

> Um estudo comparativo de técnicas de balanceamento de classes para predição de readmissão hospitalar em menos de 30 dias

---

## 📋 Contexto e Motivação

Este projeto implementa um pipeline experimental completo para investigar qual combinação de **técnica de balanceamento de classes** + **algoritmo de aprendizado de máquina** é mais eficaz para identificar pacientes diabéticos com risco de readmissão hospitalar precoce (< 30 dias).

### Lacuna Científica

O dataset **Diabetes 130-US Hospitals** (UCI Machine Learning Repository) contém 101.766 registros de internações e é fortemente desbalanceado:
- Classe positiva (<30 dias): ~11%
- Classe negativa (NO + >30 dias): ~89%
- Razão de desbalanceamento: 8:1

**Nenhum trabalho anterior comparou sistematicamente múltiplas técnicas de balanceamento neste dataset.** O trabalho mais similar (Emi-Johnson et al., 2025, Cureus) usou apenas ponderação de classes e obteve F1-score da classe positiva de apenas **0.003 com Random Forest**, evidenciando que o desbalanceamento é o principal desafio.

---

## 🎯 Delineamento Experimental

### Design Fatorial
- **Técnicas de balanceamento:** 5 (Baseline, SMOTE, ADASYN, Borderline-SMOTE, SMOTEENN)
- **Algoritmos:** 3 (Regressão Logística, Random Forest, XGBoost)
- **Combinações:** 3 × 5 = **15 combinações**

### Protocolo de Avaliação
1. **Pré-processamento:** Filtragem, codificação de ICD-9, encoding de medicamentos
2. **Split:** Treino (80%) / Teste (20%), estratificado
3. **Grid Search:** 5 folds estratificados sobre o treino (otimização de hiperparâmetros)
4. **Avaliação final:** 10 folds estratificados sobre o dataset completo
5. **Métrica primária:** F1-score da classe positiva
6. **Métricas secundárias:** AUC-ROC, MCC, Recall, Precisão

### Testes Estatísticos
- **Friedman test:** Detecta diferenças significativas entre combinações
- **Post-hoc de Nemenyi:** Identifica pares de combinações significativamente diferentes

---

## 🏗️ Estrutura de Arquivos

```
diabetes-readmission/
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências Python
├── .gitignore
├── data/
│   └── diabetic_data.csv              # Dataset (baixado automaticamente)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Pré-processamento completo
│   ├── balancing.py                   # Estratégias de balanceamento
│   ├── models.py                      # Modelos e grids de hiperparâmetros
│   ├── experiment.py                  # Loop fatorial principal
│   ├── evaluation.py                  # Análises e visualizações
│   └── utils.py                       # Funções auxiliares
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  # EDA
└── results/
    ├── tables/
    │   ├── scores_por_fold.csv        # Scores brutos de cada fold
    │   ├── tabela_resumo.csv          # Resumo: média ± std
    │   └── nemenyi_pvalues.csv        # Matriz de p-valores post-hoc
    └── figures/
        ├── barras_f1.png              # Gráfico de barras (F1-score)
        ├── heatmap_f1.png             # Heatmap F1
        ├── heatmap_mcc.png            # Heatmap MCC
        ├── heatmap_roc_auc.png        # Heatmap AUC-ROC
        └── nemenyi_heatmap.png        # Heatmap de p-valores
```

---

## 🚀 Como Executar

### Instalação

```bash
# Criar ambiente virtual (opcional mas recomendado)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou: venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Execução Completa do Pipeline

```bash
cd src

# Etapa 1: Pré-processamento (valida e gera dataset processado)
python preprocessing.py

# Etapa 2: Experimento fatorial (pode levar 1-2 horas com grids completos)
python experiment.py

# Etapa 3: Avaliação e visualizações
python evaluation.py
```

**Tempo estimado:** 90-120 minutos (dependendo de hardware)

### Executar Etapas Individuais

```bash
# Apenas pré-processamento (valida o pipeline de dados)
python preprocessing.py

# Apenas avaliação (requer scores_por_fold.csv já gerado)
python evaluation.py
```

---

## 📊 Etapas de Pré-processamento

O pipeline de pré-processamento segue a ordem:

1. **Download automático** via `ucimlrepo` (UCI Machine Learning Repository)
2. **Substituição de `?` por NaN**
3. **Remoção de colunas** com >50% de valores ausentes (`weight`, `payer_code`, `medical_specialty`)
4. **Remoção de registros** com `discharge_disposition_id` correspondente a óbito ou transferência (IDs: 11, 13, 14, 19, 20, 21)
5. **Deduplicação por paciente** — manter apenas o primeiro encontro (`encounter_id` mínimo)
6. **Binarização do alvo** — `<30 dias` → 1, caso contrário → 0
7. **Codificação etária** — faixas `[X-Y)` → ponto médio (ex: `[0-10)` → 5)
8. **Categorização de diagnósticos ICD-9** — códigos → 9 categorias clínicas (Strack et al., 2014):
   - 1 = Circulatório
   - 2 = Respiratório
   - 3 = Digestivo
   - 4 = Diabetes
   - 5 = Lesão
   - 6 = Musculoesquelético
   - 7 = Genitourinário
   - 8 = Neoplasias
   - 9 = Outros
9. **Ordinal encoding de medicamentos** — 23 variáveis: No=0, Steady=1, Up=2, Down=3
10. **One-hot encoding** das variáveis nominais (`race`, `gender`, `admission_type_id`, etc.)
11. **Normalização (z-score)** aplicada **dentro do Pipeline** para evitar data leakage

---

## 🔬 Modelos e Hiperparâmetros

### Regressão Logística
- **Parâmetros fixos:** `penalty='l2', solver='saga', max_iter=1000`
- **Grid Search:** `C=[0.01, 0.1, 1, 10, 100]`

### Random Forest
- **Parâmetros fixos:** `class_weight=None, n_jobs=-1`
- **Grid Search:**
  - `n_estimators=[100, 200, 500]`
  - `max_depth=[10, 20, None]`
  - `criterion=['gini', 'entropy']`

### XGBoost
- **Parâmetros fixos:** `eval_metric='logloss', n_jobs=-1`
- **Grid Search:**
  - `learning_rate=[0.01, 0.1, 0.3]`
  - `max_depth=[3, 6, 9]`
  - `n_estimators=[100, 300, 500]`
- **Nota:** Baseline com XGBoost adiciona `scale_pos_weight≈8` para comparação justa

---

## ⚖️ Estratégias de Balanceamento

Todas as estratégias são aplicadas **exclusivamente no conjunto de treino** de cada fold, usando `imblearn.pipeline.Pipeline`:

1. **Baseline** — Sem balanceamento (controle)
2. **SMOTE** — Synthetic Minority Over-sampling Technique (k=5)
3. **ADASYN** — Adaptive Synthetic Sampling (n_neighbors=5)
4. **Borderline-SMOTE** — Variante focada em amostras de fronteira (k=5, borderline-1)
5. **SMOTEENN** — Combinação de SMOTE + Edited Nearest Neighbors

---

## 📈 Métricas Coletadas

Por cada fold e cada combinação:

- **F1-score** (classe 1) — métrica primária
- **AUC-ROC** — área sob a curva ROC
- **Recall** (classe 1) — sensibilidade
- **Precisão** (classe 1) — valor preditivo positivo
- **MCC** — Matthews Correlation Coefficient

---

## 🔍 Outputs da Avaliação

### Tabelas
- `scores_por_fold.csv` — Scores brutos (1 linha por fold, 10×15 = 150 linhas)
- `tabela_resumo.csv` — Média ± desvio padrão por combinação (15 linhas)
- `nemenyi_pvalues.csv` — Matriz de p-valores do teste post-hoc

### Figuras
- `barras_f1.png` — Barras agrupadas por algoritmo (F1-score)
- `heatmap_f1.png` — Matriz balanceamento × algoritmo (F1)
- `heatmap_mcc.png` — Matriz balanceamento × algoritmo (MCC)
- `heatmap_roc_auc.png` — Matriz balanceamento × algoritmo (AUC-ROC)
- `nemenyi_heatmap.png` — Matriz de p-valores com coloração

---

## 🎓 Reproducibilidade

Todas as operações usam `random_state=42` para reproducibilidade total:
- Splits treino/teste
- Grid Search
- Cross-validation
- Amostragem sintética (SMOTE, ADASYN, etc.)
- Inicialização de modelos

---

## 📚 Referências

1. Emi-Johnson, F. et al. (2025). "Clinical Decision Support System for Predicting Hospital Readmission in Diabetic Patients: A Machine Learning Approach." *Cureus*, DOI: 10.7759/cureus.82437

2. Strack, B. et al. (2014). "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of the UCI Diabetes 130-US Hospitals Dataset." *Journal of Diabetes Research*.

3. UCI Machine Learning Repository. (2014). Diabetes 130-US Hospitals for Years 1999–2008. Retrieved from: https://archive.ics.uci.edu/dataset/296/

---

## ✨ Notas Importantes

- **Data leakage:** `StandardScaler` é sempre aplicado dentro do Pipeline, nunca antes do split
- **Reamostragem:** Ocorre apenas no treino de cada fold, nunca no teste/validação
- **imblearn.pipeline:** Usado em vez de `sklearn.pipeline` para garantir comportamento correto
- **Tempo de execução:** O Grid Search completo (especialmente RF + XGBoost com 500 estimadores) pode levar 1-2 horas
- **Recursos:** Recomenda-se máquina com ≥8GB RAM e processador multi-core

---

## 📧 Contato

**Autor:** Eduardo Tiyo  
**Email:** etiyo@tsp.tech  
**Instituição:** UTFPR Campo Mourão — Pós-Graduação em Ciência da Computação
