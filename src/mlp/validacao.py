"""Validacao cruzada (k-fold) para avaliar uma configuracao com metrica robusta.

Divide as amostras em k folds e treina k redes do zero: a cada rodada, um fold
diferente fica de fora como validacao e os outros k-1 treinam. A metrica final
é a media das k acuracias (+- desvio), em vez de depender da sorte de uma
divisao unica - util nos datasets pequenos (ex: CARACTERES_REDUZIDO, em que o
teste de uma divisao unica tem so 2-3 amostras).

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

import random

from entities import MLP
from value_objects import Amostra


def validacao_cruzada(
    amostras: list[Amostra],
    arquitetura: list[int],
    taxa_aprendizado: float,
    epocas: int,
    k: int,
    paciencia: int | None = None,
) -> list[float]:
    """Roda o k-fold e retorna a lista com a acuracia de cada fold."""
    amostras = list(amostras)  # copia pra nao bagunçar a lista do chamador
    random.shuffle(amostras)
    # Fold i = amostras nas posicoes i, i+k, i+2k... (tamanhos quase iguais).
    folds = [amostras[i::k] for i in range(k)]

    acuracias: list[float] = []
    for i, fold_validacao in enumerate(folds):
        # Treina uma rede NOVA (pesos re-sorteados) com todos os folds menos o i.
        treino = [a for j, fold in enumerate(folds) if j != i for a in fold]
        mlp = MLP(arquitetura)
        mlp.treinar(treino, taxa_aprendizado, epocas, fold_validacao, paciencia)

        resultados = mlp.testar(fold_validacao)
        acuracia = sum(r.acerto for r in resultados) / len(resultados)
        acuracias.append(acuracia)
        print(
            f"Fold {i + 1}/{k}: acuracia {acuracia:.2%} ({len(fold_validacao)} amostras)"
        )

    return acuracias
