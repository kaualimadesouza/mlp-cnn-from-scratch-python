"""Value objects (dataclasses e enums) compartilhados pelo projeto.

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

from dataclasses import dataclass
from enum import Enum


@dataclass
class Amostra:
    """Amostra de treino, validacao ou teste: entrada e saida esperada."""

    entrada: list[float]
    esperado: list[float]


@dataclass
class Dataset:
    """Dataset de treino, validacao e teste, com numero de entradas e saidas."""

    treino: list[Amostra]
    validacao: list[Amostra]
    teste: list[Amostra]
    num_entradas: int
    num_saidas: int


@dataclass
class ResultadoTeste:
    """Resultado de uma amostra de teste depois do forward."""

    entrada: list[float]
    esperado_raw: list[float]
    saida_raw: list[float]
    classe_predita: int
    classe_esperada: int
    acerto: bool


class MetodoValidacaoEnum(Enum):
    HOLD_OUT = "HOLD_OUT"
    CROSS_VALIDATION = "CROSS_VALIDATION"


class DataChoiceEnum(Enum):
    # Conferencia com o exemplo numerico da professora (nao é um treino):
    # fica em primeiro pra validar a corretude do forward/backprop antes
    # de rodar os treinos de verdade.
    TESTE_DE_MESA = "TESTE_DE_MESA"
    OR = "OR"
    AND = "AND"
    XOR = "XOR"
    CARACTERES_REDUZIDO = "CARACTERES_REDUZIDO"
    CARACTERES_COMPLETO = "CARACTERES_COMPLETO"
    # Variacao AUTORAL: treina/valida nos dados originais e testa na versao
    # com ruido criado pelo grupo (10% dos pixels invertidos, seed fixa).
    CARACTERES_COMPLETO_AUTORAL = "CARACTERES_COMPLETO_AUTORAL"
    IRIS = "IRIS"
