"""
Avaliação estatística e visualização dos resultados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from utils import salvar_csv


plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


def gerar_tabela_resumo(df_scores):
    """
    Gera tabela de resumo com média ± desvio padrão por combinação.
    """
    print("\n" + "="*80)
    print("GERANDO TABELA DE RESUMO")
    print("="*80)

    metricas = ['f1', 'roc_auc', 'recall', 'precision', 'mcc']

    resumo_linhas = []

    for balanceamento in df_scores['balanceamento'].unique():
        for modelo in df_scores['modelo'].unique():
            mask = (df_scores['balanceamento'] == balanceamento) & (df_scores['modelo'] == modelo)
            dados_combinacao = df_scores[mask]

            if len(dados_combinacao) == 0:
                continue

            linha = {
                'Balanceamento': balanceamento,
                'Modelo': modelo,
            }

            for metrica in metricas:
                media = dados_combinacao[metrica].mean()
                std = dados_combinacao[metrica].std()
                linha[f'{metrica.upper()} (média ± std)'] = f"{media:.4f} ± {std:.4f}"

            resumo_linhas.append(linha)

    df_resumo = pd.DataFrame(resumo_linhas)

    caminho = 'results/tables/tabela_resumo.csv'
    salvar_csv(df_resumo, caminho, index=False)

    print(f"\nTabela resumo (primeiras 10 linhas):")
    print(df_resumo.head(10).to_string())

    return df_resumo


def plot_barras_f1(df_scores):
    """
    Gráfico de barras: F1-score por combinação, agrupado por algoritmo.
    """
    print("\n" + "="*80)
    print("GERANDO GRÁFICO DE BARRAS (F1-SCORE)")
    print("="*80)

    # Agregar por combinação
    df_agg = df_scores.groupby(['balanceamento', 'modelo'])['f1'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))

    modelos = df_agg['modelo'].unique()
    balanceamentos = df_agg['balanceamento'].unique()

    x = np.arange(len(balanceamentos))
    width = 0.25

    for i, modelo in enumerate(modelos):
        dados_modelo = df_agg[df_agg['modelo'] == modelo]
        means = [dados_modelo[dados_modelo['balanceamento'] == b]['mean'].values[0]
                 if len(dados_modelo[dados_modelo['balanceamento'] == b]) > 0
                 else 0 for b in balanceamentos]
        stds = [dados_modelo[dados_modelo['balanceamento'] == b]['std'].values[0]
                if len(dados_modelo[dados_modelo['balanceamento'] == b]) > 0
                else 0 for b in balanceamentos]

        ax.bar(x + i * width, means, width, label=modelo, yerr=stds, capsize=5, alpha=0.8)

    ax.set_xlabel('Estratégia de Balanceamento', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score (Classe Positiva)', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score por Combinação de Balanceamento e Algoritmo', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(balanceamentos, rotation=45, ha='right')
    ax.legend(title='Modelo', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    caminho = 'results/figures/barras_f1.png'
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Gráfico salvo: {caminho}")


def plot_heatmap(df_scores, metrica='f1'):
    """
    Heatmap: matriz balanceamento × algoritmo para uma métrica.
    """
    print(f"\n" + "="*80)
    print(f"GERANDO HEATMAP ({metrica.upper()})")
    print("="*80)

    # Criar matriz
    df_agg = df_scores.groupby(['balanceamento', 'modelo'])[metrica].mean().reset_index()
    pivot = df_agg.pivot(index='balanceamento', columns='modelo', values=metrica)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax, cbar_kws={'label': metrica.upper()})

    ax.set_title(f'Heatmap: {metrica.upper()} por Balanceamento e Algoritmo', fontsize=14, fontweight='bold')
    ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Estratégia de Balanceamento', fontsize=12, fontweight='bold')

    plt.tight_layout()
    caminho = f'results/figures/heatmap_{metrica}.png'
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Heatmap salvo: {caminho}")


def teste_friedman_nemenyi(df_scores):
    """
    Testa significância estatística entre combinações usando Friedman + post-hoc Nemenyi.
    """
    print("\n" + "="*80)
    print("TESTE ESTATÍSTICO: FRIEDMAN + NEMENYI")
    print("="*80)

    # Criar matriz: linhas=folds, colunas=combinações
    combinacoes = df_scores.groupby(['balanceamento', 'modelo']).size().index.tolist()

    # Matriz de dados
    dados = []
    for fold_id in df_scores['fold'].unique():
        row = []
        for bal, mod in combinacoes:
            f1_val = df_scores[(df_scores['fold'] == fold_id) &
                               (df_scores['balanceamento'] == bal) &
                               (df_scores['modelo'] == mod)]['f1'].values
            row.append(f1_val[0] if len(f1_val) > 0 else np.nan)
        dados.append(row)

    dados_array = np.array(dados)

    # Teste de Friedman
    stat, p_value = friedmanchisquare(*dados_array.T)
    print(f"\nTeste de Friedman:")
    print(f"  Estatística: {stat:.4f}")
    print(f"  p-valor: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  ✓ Diferenças significativas detectadas (p < 0.05)")

        # Post-hoc Nemenyi
        print(f"\nTeste post-hoc de Nemenyi...")
        try:
            df_dados = pd.DataFrame(dados_array, columns=[f"{b}_{m}" for b, m in combinacoes])
            resultado_nemenyi = sp.posthoc_nemenyi_friedman(dados_array.T)

            caminho = 'results/tables/nemenyi_pvalues.csv'
            resultado_nemenyi.to_csv(caminho)
            print(f"✓ Matriz de p-valores salva: {caminho}")

            # Visualizar
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(resultado_nemenyi, annot=True, fmt='.3f', cmap='RdYlGn_r',
                        ax=ax, cbar_kws={'label': 'p-valor'}, vmin=0, vmax=0.05)
            ax.set_title('Matriz de p-valores - Teste post-hoc de Nemenyi', fontsize=14, fontweight='bold')
            plt.tight_layout()

            caminho_fig = 'results/figures/nemenyi_heatmap.png'
            Path(caminho_fig).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(caminho_fig, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Heatmap de p-valores salvo: {caminho_fig}")

        except Exception as e:
            print(f"✗ Erro ao executar post-hoc Nemenyi: {e}")
    else:
        print(f"  ✗ Sem diferenças significativas (p >= 0.05)")


def gerar_relatorio_completo(df_scores):
    """Executa todas as análises de avaliação."""

    print("\n" + "█"*80)
    print("ANÁLISE DE RESULTADOS")
    print("█"*80)

    # Tabela resumo
    df_resumo = gerar_tabela_resumo(df_scores)

    # Gráficos
    plot_barras_f1(df_scores)
    plot_heatmap(df_scores, 'f1')
    plot_heatmap(df_scores, 'mcc')
    plot_heatmap(df_scores, 'roc_auc')

    # Testes estatísticos
    teste_friedman_nemenyi(df_scores)

    print("\n" + "█"*80)
    print("ANÁLISE COMPLETA")
    print("█"*80)
    print("✓ Todos os outputs gerados em results/")


if __name__ == '__main__':
    # Carregar resultados
    print("Carregando scores_por_fold.csv...")
    df_scores = pd.read_csv('results/tables/scores_por_fold.csv')
    print(f"✓ {len(df_scores)} registros carregados")

    # Gerar relatório
    gerar_relatorio_completo(df_scores)
    print("\n✓ Avaliação concluída!")
