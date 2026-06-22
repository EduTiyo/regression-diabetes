# Guia de Execução — Pipeline Experimental

Instruções passo-a-passo para executar o pipeline experimental completo.

---

## ⚠️ Pré-requisitos

### Sistema
- Conda (Anaconda/Miniconda)
- macOS: `brew install libomp` (necessário para XGBoost)
- Linux: geralmente OpenMP já está disponível
- Windows: instalar via MSVC build tools

### Hardware
- **Recomendado:** 8GB+ RAM, processador multi-core
- **Tempo estimado:** 90-120 minutos para o experimento completo

---

## 1️⃣ Setup Inicial (uma única vez)

### Criar e ativar ambiente Conda

```bash
cd ~/Documents/diabetes-readmission

# Criar ambiente
conda create -y -n diabetes-readmission python=3.10

# Ativar
conda activate diabetes-readmission
```

### Instalar dependências

```bash
# macOS: instalar OpenMP via Homebrew (NECESSÁRIO)
brew install libomp

# Instalar pacotes Python
pip install -r requirements.txt
```

### Verificar instalação

```bash
python -c "import pandas, sklearn, imblearn, xgboost; print('✓ Todas as dependências OK')"
```

---

## 2️⃣ Teste Rápido (validação)

Executar uma combinação simples para validar que tudo está funcionando:

```bash
python test_experiment.py
```

**Esperado:** Deve completar em 2-3 minutos com sucesso e salvar `results/tables/test_scores.csv`

---

## 3️⃣ Experimento Completo

O experimento fatorial completo tem 3 etapas:

### Etapa 1: Pré-processamento (validação)

```bash
cd src
python preprocessing.py
```

- Baixa dataset automaticamente do UCI
- Aplica todas as transformações
- Salva dados processados em `../data/diabetic_data.csv`
- **Tempo:** ~2-5 minutos

### Etapa 2: Experimento Fatorial (o main)

```bash
python experiment.py
```

**O que faz:**
- Executa 15 combinações (3 modelos × 5 estratégias de balanceamento)
- Para cada combinação:
  - Grid Search com 5 folds (otimiza hiperparâmetros no treino)
  - Avaliação final com 10 folds (no dataset completo)
- Coleta 5 métricas: F1, AUC-ROC, Recall, Precisão, MCC
- Salva scores brutos em `../results/tables/scores_por_fold.csv`

**Tempo:** 90-120 minutos (depende do hardware)

**Monitor:**
- O script imprime o status de cada combinação
- Log verbose mostra grid search progress
- Ao final: sumário com F1 por combinação

### Etapa 3: Análise e Visualizações

```bash
python evaluation.py
```

**Gera:**
- `../results/tables/tabela_resumo.csv` — média ± std por combinação
- `../results/tables/nemenyi_pvalues.csv` — matriz p-valores teste post-hoc
- `../results/figures/barras_f1.png` — gráfico de barras F1
- `../results/figures/heatmap_f1.png` — heatmap F1
- `../results/figures/heatmap_mcc.png` — heatmap MCC
- `../results/figures/heatmap_roc_auc.png` — heatmap AUC-ROC
- `../results/figures/nemenyi_heatmap.png` — heatmap p-valores

**Tempo:** ~2-5 minutos

---

## 4️⃣ Exploração com Jupyter Notebook

Para análise exploratória interativa:

```bash
cd notebooks
jupyter notebook 01_exploratory_analysis.ipynb
```

Células:
- Distribuição da variável-alvo
- Análise demográfica
- Diagnósticos ICD-9
- Medicamentos
- Valores faltantes

---

## 📊 Estrutura de Outputs

Após execução completa:

```
results/
├── tables/
│   ├── scores_por_fold.csv          (150 linhas = 10 folds × 15 combinações)
│   ├── tabela_resumo.csv            (15 linhas = 1 por combinação)
│   └── nemenyi_pvalues.csv          (matriz 15×15)
└── figures/
    ├── barras_f1.png
    ├── heatmap_f1.png
    ├── heatmap_mcc.png
    ├── heatmap_roc_auc.png
    └── nemenyi_heatmap.png
```

---

## 🔧 Troubleshooting

### XGBoost: "libomp.dylib not found" (macOS)

```bash
brew install libomp
conda deactivate
conda activate diabetes-readmission
pip uninstall xgboost
pip install xgboost
```

### Memory Error

Se receber "out of memory":
- Reduzir `n_jobs` de `-1` a `-2` em `src/models.py`
- Ou reduzir `n_estimators` em Random Forest/XGBoost

### Dataset não baixado

Se script falhar ao baixar via `ucimlrepo`:
- Download manual: https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008
- Salvar como `data/diabetic_data.csv`
- Reexecitar `src/preprocessing.py`

---

## 📈 Interpretação dos Resultados

### F1-Score (Métrica Primária)

- **Baseline:** Controle sem balanceamento
- **SMOTE:** Sobre-amostragem sintética
- **ADASYN:** Over-sampling adaptativo
- **BorderlineSMOTE:** Focado em amostras de fronteira
- **SMOTEENN:** Combinação com sub-amostragem

Espera-se que SMOTE/ADASYN > Baseline (devido ao desbalanceamento 8:1)

### Teste Estatístico (Friedman + Nemenyi)

- Se `p-valor < 0.05`: diferenças significativas entre combinações
- Post-hoc Nemenyi: identifica pares significativos
- Matrix `nemenyi_heatmap.png`: verde = sem diferença, vermelho = significativa

### AUC-ROC

- Métrica robusta ao desbalanceamento
- Esperado: 0.6-0.7 (dataset desafiador)

---

## 🚀 Exemplo Completo

```bash
# 1. Setup (primeira vez)
conda create -y -n diabetes-readmission python=3.10
conda activate diabetes-readmission
brew install libomp  # macOS only
pip install -r requirements.txt

# 2. Teste rápido
python test_experiment.py      # 2-3 min

# 3. Experimento (se teste OK)
cd src
python preprocessing.py         # 2-5 min
python experiment.py            # 90-120 min ⏱️
python evaluation.py            # 2-5 min
cd ..

# 4. Exploração
jupyter notebook notebooks/01_exploratory_analysis.ipynb

# 5. Visualizar resultados
open results/figures/          # macOS
# ou: xdg-open results/figures/  # Linux
# ou: start results/figures/     # Windows
```

---

## 📝 Anotações

- **Random seed:** 42 em todas as etapas (reproducibilidade)
- **Data leakage:** StandardScaler está dentro do Pipeline (jamais antes do split)
- **Reamostragem:** Apenas no treino de cada fold
- **Grid Search:** 5 folds durante otimização
- **Avaliação final:** 10 folds sobre dataset completo
- **Todas as combinações** completam mesmo se houver erro em uma (tratado com try-except)

---

## 📧 Suporte

Se encontrar problemas:
1. Verificar `requirements.txt` — versões compatíveis?
2. Verificar RAM disponível — experimento usa muita memória
3. Verificar conda environment — rodando corretamente?
4. Revisar logs de erro em `src/*.py`

---

**Última atualização:** 2026-06-21
