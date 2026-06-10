"""Configuracao centralizada (constantes de pasta, arquivos e defaults).

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

import os
from dataclasses import dataclass

from value_objects import DataChoiceEnum, MetodoValidacaoEnum


@dataclass(frozen=True)
class HiperparametrosExperimento:
    """Hiperparametros de um experimento (taxa, epocas, arquitetura).

    Cada dataset tem sua propria instancia no dict `HIPERPARAMETROS`,
    porque cada um tem escala, dificuldade e capacidade minima diferentes.
    """

    taxa_aprendizado: float
    epocas: int
    num_neuronios_oculta: int
    paciencia: int | None = None
    metodo_validacao: MetodoValidacaoEnum = MetodoValidacaoEnum.HOLD_OUT
    k_folds: int | None = None


# Hiperparametros por dataset - tabela publica usada pelo main.
HIPERPARAMETROS: dict[DataChoiceEnum, HiperparametrosExperimento] = {
    DataChoiceEnum.OR: HiperparametrosExperimento(
        taxa_aprendizado=0.4,
        epocas=500,
        num_neuronios_oculta=4,
        paciencia=None,
    ),
    DataChoiceEnum.AND: HiperparametrosExperimento(
        taxa_aprendizado=0.4,
        epocas=500,
        num_neuronios_oculta=4,
        paciencia=None,
    ),
    DataChoiceEnum.XOR: HiperparametrosExperimento(
        taxa_aprendizado=0.4,
        epocas=2000,
        num_neuronios_oculta=4,
        paciencia=None,
    ),
    DataChoiceEnum.CARACTERES_REDUZIDO: HiperparametrosExperimento(
        taxa_aprendizado=0.2,
        epocas=500,
        num_neuronios_oculta=20,
        paciencia=20,
        metodo_validacao=MetodoValidacaoEnum.CROSS_VALIDATION,
        k_folds=7,
    ),
    DataChoiceEnum.CARACTERES_COMPLETO: HiperparametrosExperimento(
        taxa_aprendizado=0.04,
        epocas=300,
        num_neuronios_oculta=55,
        paciencia=None,
    ),
    DataChoiceEnum.CARACTERES_COMPLETO_AUTORAL: HiperparametrosExperimento(
        taxa_aprendizado=0.04,
        epocas=300,
        num_neuronios_oculta=55,
        paciencia=15,
    ),
    DataChoiceEnum.IRIS: HiperparametrosExperimento(
        taxa_aprendizado=0.2,
        epocas=500,
        num_neuronios_oculta=8,
        paciencia=20,
    ),
}


@dataclass(frozen=True)
class Config:
    # Integrantes do grupo
    integrantes: tuple[str, ...] = (
        "Isabelle da Silva Tobias - NUSP 15525991 (T04)",
        "Kevin Rodrigues Nunes    - NUSP 15676030 (T94)",
        "Kauã Lima de Souza       - NUSP 15674702 (T94)",
        "Victor Yodono            - NUSP 13829040 (T94)",
    )

    # Pasta onde os arquivos de saida sao salvos (separada por objetivo: mlp/ ou cnn/)
    pasta_saidas: str = "saidas/mlp"

    # Nomes dos arquivos de saida
    nome_arquivo_erro_por_epoca: str = "erro_por_epoca.csv"
    nome_arquivo_hiperparametros: str = "hiperparametros.txt"
    nome_arquivo_pesos_iniciais: str = "pesos_iniciais.txt"
    nome_arquivo_pesos_finais: str = "pesos_finais.txt"
    nome_arquivo_saidas_teste: str = "saidas_teste.csv"

    def caminho_saida(self, prefixo: str, nome_arquivo: str) -> str:
        """Monta o caminho `<pasta_saidas>/<prefixo>/<nome_arquivo>`.

        Cria a pasta `<pasta_saidas>/<prefixo>/` se ainda nao existir, pra que
        o chamador possa abrir o arquivo direto sem se preocupar com mkdir.
        """
        pasta = os.path.join(self.pasta_saidas, prefixo)
        os.makedirs(pasta, exist_ok=True)
        return os.path.join(pasta, nome_arquivo)


# Instancia unica usada em todo o codigo
config = Config()
