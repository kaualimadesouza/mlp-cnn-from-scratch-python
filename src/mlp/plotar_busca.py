"""Plota o resultado da busca de parametros (busca_parametros.csv) como tabela PNG.

Rodar: uv run python src/mlp/plotar_busca.py [N]
N = quantas linhas do ranking mostrar (default 15; o CSV vem ordenado do
melhor pro pior MSE de validacao).

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTA = "saidas/mlp/caracteres_completo"
HARDWARE = (
    "Hardware: Intel Core 7 240H (16 threads), 16 GB RAM, WSL2 (Linux) "
    "- Python puro, sem vetorizacao"
)
# Configuracao escolhida pelo grupo (linha destacada na tabela)
ESCOLHIDA = {"taxa_aprendizado": "0.04", "neuronios_oculta": "55", "paciencia": "15"}


def main(top_n: int = 15) -> None:
    with open(f"{PASTA}/busca_parametros.csv") as f:
        busca = list(csv.DictReader(f))
    linhas_csv = busca[:top_n]
    tem_tempo = "tempo_s" in busca[0]

    colunas = [
        "#",
        "MSE val.",
        "Taxa (α)",
        "Neurônios",
        "Paciência",
        "Épocas",
        "Acurácia teste",
    ]
    if tem_tempo:
        colunas.append("Tempo (s)")

    linhas = []
    destaque = None
    for i, l in enumerate(linhas_csv):
        if all(l[k] == v for k, v in ESCOLHIDA.items()):
            destaque = i
        linha = [
            str(i + 1),
            f"{float(l['melhor_mse_val']):.4f}",
            l["taxa_aprendizado"],
            l["neuronios_oculta"],
            l["paciencia"],
            f"{l['epoca_parada']} / 300",
            f"{float(l['acuracia_teste']):.1%}",
        ]
        if tem_tempo:
            linha.append(l["tempo_s"])
        linhas.append(linha)

    fig, ax = plt.subplots(figsize=(1.5 * len(colunas) + 1, 0.45 * len(linhas) + 2.4))
    ax.axis("off")
    tabela = ax.table(
        cellText=linhas, colLabels=colunas, loc="center", cellLoc="center"
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(10)
    tabela.scale(1, 1.5)
    for (lin, _), cel in tabela.get_celld().items():
        cel.set_edgecolor("#D9D9D9")
        if lin == 0:
            cel.set_facecolor("#C55A11")
            cel.set_text_props(color="white", weight="bold")
        elif destaque is not None and lin - 1 == destaque:
            cel.set_facecolor("#FBE5D6")
    ax.set_title(
        f"Busca de hiperparâmetros — top {len(linhas)} de {len(busca)} combinações\n"
        "(CARACTERES COMPLETO — ordenado pelo MSE de validação)",
        fontsize=13,
        weight="bold",
        pad=16,
    )
    rodape = "Linha destacada = configuração escolhida pelo grupo (taxa 0.04, 55 neurônios, paciência 15)."
    if tem_tempo:
        rodape += " Tempo medido durante a busca."
    fig.text(0.5, 0.06, rodape, ha="center", fontsize=8, color="#7F7F7F")
    fig.text(0.5, 0.02, HARDWARE, ha="center", fontsize=8, color="#7F7F7F")

    caminho = f"{PASTA}/busca_parametros.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Tabela salva: {caminho}")


if __name__ == "__main__":
    main(15)
