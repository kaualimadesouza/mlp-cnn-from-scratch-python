"""Busca de hiperparametros (grid search): treina todas as combinacoes de taxa
de aprendizado x neuronios na camada escondida e escolhe pelo MENOR MSE de
validacao (o teste so confere no final - nunca participa da escolha).

A paciencia da parada antecipada nao muda a trajetoria do treino - so decide
onde cortar a curva. Entao cada combinacao é treinada UMA vez (curva completa)
e as paciencias sao SIMULADAS sobre a mesma curva de validacao, de graca.

Rodar: python src/mlp/busca_parametros.py

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

import random
import time

import plotar_busca
from config import config
from datasets import carregar_dados
from entities import MLP
from value_objects import DataChoiceEnum, Dataset

TAXAS = [0.01, 0.02, 0.04, 0.07, 0.1]
NEURONIOS = [20, 40, 55, 70, 90]
EPOCAS = 300  # orcamento por treino; as paciencias sao simuladas sobre a curva
PACIENCIAS = [5, 10, 15, 20, 30]


def simular_parada(hist_val: list[float], paciencia: int) -> int:
    """Epoca (1-based) em que o treino teria parado com essa paciencia."""
    melhor = float("inf")
    sem_melhora = 0
    for i, mse in enumerate(hist_val):
        if mse < melhor:
            melhor, sem_melhora = mse, 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                return i + 1
    return len(hist_val)  # paciencia nunca estourou: treinou ate o teto


def avaliar(dataset: Dataset, taxa: float, neuronios: int) -> list[tuple]:
    """Treina UMA vez (curva completa) e simula cada paciencia sobre a curva.

    Retorna uma tupla (mse_val, taxa, neuronios, paciencia, epoca_parada,
    acuracia) por paciencia. A acuracia é da rede com o treino completo -
    serve so de conferencia; o ranking usa o MSE de validacao no corte.
    """
    mlp = MLP([dataset.num_entradas, neuronios, dataset.num_saidas])
    t0 = time.perf_counter()
    _, hist_val = mlp.treinar(dataset.treino, taxa, EPOCAS, dataset.validacao)
    tempo_por_epoca = (time.perf_counter() - t0) / len(hist_val)
    teste = mlp.testar(dataset.teste)
    acuracia = sum(r.acerto for r in teste) / len(teste)

    linhas = []
    for paciencia in PACIENCIAS:
        parada = simular_parada(hist_val, paciencia)
        mse_no_corte = min(hist_val[:parada])
        tempo = parada * tempo_por_epoca
        linhas.append(
            (mse_no_corte, taxa, neuronios, paciencia, parada, acuracia, tempo)
        )
    return linhas


def salvar_csv(resultados: list[tuple], caminho: str) -> None:
    """Salva uma linha por (combinacao x paciencia), da melhor pra pior."""
    with open(caminho, "w") as f:
        f.write(
            "melhor_mse_val,taxa_aprendizado,neuronios_oculta,paciencia,"
            "epoca_parada,acuracia_teste,tempo_s\n"
        )
        for mse, taxa, n, paciencia, parada, acuracia, tempo in resultados:
            f.write(
                f"{mse:.6f},{taxa},{n},{paciencia},{parada},{acuracia:.4f},{tempo:.1f}\n"
            )


def main() -> None:
    random.seed(42)  # busca reproduzivel
    # Carrega UMA vez: todas as combinacoes usam o mesmo split (comparacao justa).
    dataset = carregar_dados(DataChoiceEnum.CARACTERES_COMPLETO)

    resultados: list[tuple] = []
    for taxa in TAXAS:
        for n in NEURONIOS:
            linhas = avaliar(dataset, taxa, n)
            resultados.extend(linhas)
            mse, _, _, paciencia, parada, acuracia, _ = min(linhas)
            print(
                f"taxa={taxa:<5} oculta={n:<3} -> melhor paciencia={paciencia} "
                f"(para na epoca {parada}, mse_val={mse:.6f})  acuracia={acuracia:.2%}"
            )

    # Ordena pelo MSE de validacao (menor primeiro = vencedor).
    resultados.sort()
    caminho = config.caminho_saida("caracteres_completo", "busca_parametros.csv")
    salvar_csv(resultados, caminho)

    mse, taxa, n, paciencia, parada, acuracia, _ = resultados[0]
    print(
        f"\nVencedor: taxa={taxa} oculta={n} paciencia={paciencia} "
        f"(para na epoca {parada}, mse_val={mse:.6f}, acuracia={acuracia:.2%})"
    )
    print(f"Salvo em: {caminho}")

    # Gera a imagem da tabela (top 15 do ranking) a partir do CSV recem-salvo.
    plotar_busca.main()


if __name__ == "__main__":
    main()
