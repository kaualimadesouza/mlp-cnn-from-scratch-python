"""Carregamento dos datasets usados no EP (portas logicas e caracteres).

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

import random

import numpy as np
import pandas as pd
from value_objects import Amostra, DataChoiceEnum, Dataset

PATHS_DATASETS: dict[DataChoiceEnum, str] = {
    DataChoiceEnum.OR: "data/portas_logicas/or.csv",
    DataChoiceEnum.AND: "data/portas_logicas/and.csv",
    DataChoiceEnum.XOR: "data/portas_logicas/xor.csv",
    DataChoiceEnum.CARACTERES_REDUZIDO: "data/caracteres_reduzido",
    DataChoiceEnum.CARACTERES_COMPLETO: "data/caracteres_completo",
    DataChoiceEnum.CARACTERES_COMPLETO_AUTORAL: "data/caracteres_completo_autoral",
    DataChoiceEnum.IRIS: "data/iris/iris.csv",
}


def _bipolar_para_unipolar(df: pd.DataFrame) -> pd.DataFrame:
    """Substitui -1 por 0 e +1 por 1, pra casar com o range da sigmoide."""
    return df.replace({-1: 0, +1: 1})


def _df_para_amostras(df: pd.DataFrame, num_entradas: int) -> list[Amostra]:
    amostras: list[Amostra] = []
    for _, linha in df.iterrows():
        entrada = linha.iloc[:num_entradas].tolist()
        esperado = linha.iloc[num_entradas:].tolist()
        amostra = Amostra(entrada=entrada, esperado=esperado)
        amostras.append(amostra)
    return amostras


def _carregar_porta_logica(path: str) -> Dataset:
    # CSVs vem em formato bipolar {-1, +1}; convertemos pra {0, 1} pra sigmoide.
    df = _bipolar_para_unipolar(pd.read_csv(path, header=None))

    # O numero de entradas e o numero de colunas menos 1 (a ultima e a saida).
    num_entradas = len(df.columns) - 1

    # As mesmas 4 amostras servem de treino, validacao e teste (dataset muito pequeno pra dividir).
    treino = _df_para_amostras(df, num_entradas)
    validacao = _df_para_amostras(df, num_entradas)
    teste = _df_para_amostras(df, num_entradas)

    return Dataset(
        treino=treino,
        validacao=validacao,
        teste=teste,
        num_entradas=num_entradas,
        num_saidas=1,
    )


def _carregar_caracteres_reduzido(pasta: str) -> Dataset:
    # CSVs vem em formato bipolar {-1, +1}; convertemos pra {0, 1} pra sigmoide.
    limpo = _bipolar_para_unipolar(pd.read_csv(f"{pasta}/limpo.csv", header=None))
    ruido = _bipolar_para_unipolar(pd.read_csv(f"{pasta}/ruido.csv", header=None))
    ruido20 = _bipolar_para_unipolar(pd.read_csv(f"{pasta}/ruido20.csv", header=None))

    num_entradas = max(len(limpo.columns), len(ruido.columns), len(ruido20.columns)) - 7

    todas = (
        _df_para_amostras(limpo, num_entradas)
        + _df_para_amostras(ruido, num_entradas)
        + _df_para_amostras(ruido20, num_entradas)
    )
    random.shuffle(todas)
    total = len(todas)
    n_treino = total * 80 // 100
    n_validacao = total * 10 // 100
    treino = todas[:n_treino]
    validacao = todas[n_treino : n_treino + n_validacao]
    teste = todas[n_treino + n_validacao :]
    return Dataset(
        treino=treino,
        validacao=validacao,
        teste=teste,
        num_entradas=num_entradas,
        num_saidas=7,
    )


def _carregar_caracteres_completo(pasta: str) -> Dataset:
    # X.npy vem em formato de imagem (N, 10, 12, 1) bipolar {-1,+1}; achatamos e
    # convertemos pra unipolar {0, 1} pra casar com o range da sigmoide.
    X = (np.load(f"{pasta}/X.npy").reshape(-1, 120) + 1) / 2
    # Y_classe.npy ja vem em {0, 1} one-hot; mantemos como esta pra sigmoide.
    Y = np.load(f"{pasta}/Y_classe.npy")

    amostras: list[Amostra] = []
    for x, y in zip(X, Y, strict=True):
        amostras.append(Amostra(entrada=x.tolist(), esperado=y.tolist()))

    # Shuffle ANTES do split
    random.shuffle(amostras)

    # Split 80/10/10: treino / validacao / teste
    total = len(amostras)
    n_treino = total * 80 // 100
    n_validacao = total * 10 // 100
    treino = amostras[:n_treino]
    validacao = amostras[n_treino : n_treino + n_validacao]
    teste = amostras[n_treino + n_validacao :]

    return Dataset(
        treino=treino,
        validacao=validacao,
        teste=teste,
        num_entradas=X.shape[1],
        num_saidas=Y.shape[1],
    )


def _carregar_caracteres_completo_autoral(pasta: str) -> Dataset:
    """Treina/valida nos dados ORIGINAIS e testa na variacao AUTORAL com ruido.

    X_ruido.npy tem as mesmas imagens/rotulos do X.npy original, na mesma
    ordem, mas com 10% dos pixels invertidos (gerado por
    src/mlp/gerar_variacao_autoral.py). Mede a robustez da rede a um ruido
    que ela nunca viu no treino.
    """
    X = (np.load("data/caracteres_completo/X.npy").reshape(-1, 120) + 1) / 2
    X_ruido = (np.load(f"{pasta}/X_ruido.npy").reshape(-1, 120) + 1) / 2
    Y = np.load("data/caracteres_completo/Y_classe.npy")

    # Embaralha INDICES (nao as listas) pra treino/teste usarem o mesmo sorteio:
    # a amostra i tem a versao limpa em X[i] e a ruidosa em X_ruido[i].
    indices = list(range(len(X)))
    random.shuffle(indices)
    total = len(indices)
    n_treino = total * 80 // 100
    n_validacao = total * 10 // 100

    treino = [Amostra(X[i].tolist(), Y[i].tolist()) for i in indices[:n_treino]]
    validacao = [
        Amostra(X[i].tolist(), Y[i].tolist())
        for i in indices[n_treino : n_treino + n_validacao]
    ]
    # So o TESTE usa a versao com ruido autoral.
    teste = [
        Amostra(X_ruido[i].tolist(), Y[i].tolist())
        for i in indices[n_treino + n_validacao :]
    ]

    return Dataset(
        treino=treino,
        validacao=validacao,
        teste=teste,
        num_entradas=X.shape[1],
        num_saidas=Y.shape[1],
    )


def _carregar_iris(path: str) -> Dataset:
    """Dataset externo: 4 atributos contínuos normalizados [0,1] e classe one-hot."""
    df = pd.read_csv(path)
    atributos = df.iloc[:, :4]
    # Normalizacao min-max coluna a coluna pra casar com a faixa da sigmoide.
    atributos = (atributos - atributos.min()) / (atributos.max() - atributos.min())

    classes = sorted(df.iloc[:, 4].unique())
    amostras: list[Amostra] = []
    for valores, classe in zip(atributos.to_numpy(), df.iloc[:, 4]):
        esperado = [1.0 if c == classe else 0.0 for c in classes]
        amostras.append(Amostra(entrada=valores.tolist(), esperado=esperado))

    random.shuffle(amostras)
    total = len(amostras)
    n_treino = total * 80 // 100
    n_validacao = total * 10 // 100
    return Dataset(
        treino=amostras[:n_treino],
        validacao=amostras[n_treino : n_treino + n_validacao],
        teste=amostras[n_treino + n_validacao :],
        num_entradas=4,
        num_saidas=len(classes),
    )


def carregar_dados(data_choice: DataChoiceEnum) -> Dataset:
    """Carrega o dataset escolhido, convertendo pra formato de amostras."""
    path = PATHS_DATASETS[data_choice]
    match data_choice:
        case DataChoiceEnum.OR | DataChoiceEnum.AND | DataChoiceEnum.XOR:
            return _carregar_porta_logica(path)
        case DataChoiceEnum.CARACTERES_REDUZIDO:
            return _carregar_caracteres_reduzido(path)
        case DataChoiceEnum.CARACTERES_COMPLETO:
            return _carregar_caracteres_completo(path)
        case DataChoiceEnum.CARACTERES_COMPLETO_AUTORAL:
            return _carregar_caracteres_completo_autoral(path)
        case DataChoiceEnum.IRIS:
            return _carregar_iris(path)
        case _:
            raise ValueError(f"Escolha de dataset invalida: {data_choice!r}")
