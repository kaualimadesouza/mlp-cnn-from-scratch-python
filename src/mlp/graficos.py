"""Geracao dos graficos PNG (MSE por epoca, validacao cruzada, matriz de confusao).

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from resultado import ResultadoExperimento
from value_objects import ResultadoTeste


def salvar_grafico_mse(resultado: ResultadoExperimento, caminho: str) -> None:
    """Gera o PNG de evolucao do MSE a partir de um ResultadoExperimento.

    Plota duas curvas (treino e validacao) com titulo e hiperparametros no topo.
    """
    epocas_x = list(range(1, len(resultado.historico_erro) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))

    # MSE de treino (sempre presente)
    ax.plot(
        epocas_x,
        resultado.historico_erro,
        color="#1f77b4",
        linestyle="-",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="MSE Treino",
    )

    # MSE de validacao (se foi coletado durante o treino via `dados_validacao`)
    if resultado.historico_validacao:
        ax.plot(
            epocas_x,
            resultado.historico_validacao,
            color="#d62728",
            linestyle="--",
            marker="s",
            markersize=3,
            linewidth=1.5,
            label="MSE Validacao",
        )
        mse_final = resultado.historico_validacao[-1]
        ax.annotate(
            f"Final: {mse_final:.4f}",
            xy=(epocas_x[-1], mse_final),
            xytext=(-60, 10),
            textcoords="offset points",
            fontsize=9,
            color="#666",
        )

    # Titulo principal (dataset) + subtitulo com hiperparametros
    fig.suptitle(
        f"Evolucao do Erro Medio Quadratico (MSE) - {resultado.data_choice.name}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    ax.set_title(
        f"taxa={resultado.taxa_aprendizado}  |  "
        f"oculta={resultado.num_neuronios_oculta} neuronios  |  "
        f"epocas={resultado.epocas}  |  "
        f"ativacao=sigmoide",
        fontsize=9,
        color="#555",
        pad=10,
    )
    ax.set_xlabel("Epocas", fontsize=11)
    ax.set_ylabel("MSE", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # deixa espaco pro suptitle

    pasta = os.path.dirname(caminho)
    if pasta:
        os.makedirs(pasta, exist_ok=True)

    fig.savefig(caminho, dpi=100)
    plt.close(fig)


def salvar_grafico_validacao_cruzada(acuracias: list[float], caminho: str) -> None:
    """Diagrama k-fold: cada linha é uma rodada e o bloco verde é a validacao."""
    k = len(acuracias)
    media = sum(acuracias) / k
    desvio = (sum((a - media) ** 2 for a in acuracias) / k) ** 0.5

    fig, ax = plt.subplots(figsize=(9, 0.7 * k + 1.5))
    for fold in range(k):
        for bloco in range(k):
            cor = "#b6dbb0" if bloco == fold else "#f7dc94"
            ax.barh(
                fold, 1 / k, left=bloco / k, color=cor, edgecolor="#777", height=0.72
            )
        # acuracia da rodada escrita dentro do bloco de validacao
        ax.text(
            fold / k + 1 / (2 * k),
            fold,
            f"{acuracias[fold]:.0%}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Fold {i + 1}" for i in range(k)])
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legenda = [
        mpatches.Patch(facecolor="#f7dc94", edgecolor="#777", label="Treino"),
        mpatches.Patch(facecolor="#b6dbb0", edgecolor="#777", label="Validacao"),
    ]
    ax.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    ax.set_title(f"Validacao Cruzada ({k}-fold)  -  media {media:.2%} +- {desvio:.2%}")
    fig.tight_layout()
    fig.savefig(caminho, dpi=100)
    plt.close(fig)


def salvar_matriz_confusao(resultados: list[ResultadoTeste], caminho: str) -> None:
    """Salva a matriz de confusao como PNG (linha = real, coluna = predita; diagonal = acertos)."""

    # Descobre quantas classes tem (olha o maior indice que aparece nos resultados).
    n = max(r.classe_esperada for r in resultados) + 1

    # Cria matriz NxN zerada e conta quantas vezes "real=i" virou "predita=j".
    matriz = [[0] * n for _ in range(n)]
    for r in resultados:
        matriz[r.classe_esperada][r.classe_predita] += 1

    # Heatmap vermelho: celula mais escura = mais ocorrencias.
    fig, ax = plt.subplots()
    ax.imshow(matriz, cmap="Reds")

    # Escreve o numero em cada celula.
    for i in range(n):
        for j in range(n):
            ax.text(j, i, matriz[i][j], ha="center", va="center", fontsize=7)

    ax.set_xlabel("Predita")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusao")
    fig.tight_layout()
    fig.savefig(caminho, dpi=100)
    plt.close(fig)
