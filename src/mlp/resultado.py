"""Resultado completo de um experimento (treino + teste).

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

from dataclasses import dataclass

from entities import MLP, Camada
from value_objects import DataChoiceEnum, Dataset, ResultadoTeste


@dataclass
class ResultadoExperimento:
    """Resultado completo de um experimento (treino + teste).

    Usado pra desacoplar "rodar o experimento" de "salvar arquivos de saida".
    O `run()` produz isso e o `main()` consome pra gerar os arquivos.
    """

    data_choice: DataChoiceEnum
    dataset: Dataset
    arquitetura: list[int]
    taxa_aprendizado: float
    epocas: int
    num_neuronios_oculta: int
    mlp: MLP
    camadas_iniciais: list[Camada]
    historico_erro: list[float]
    historico_validacao: list[float]
    resultados_teste: list[ResultadoTeste]
    tempo_treino: float
    acuracias_cv: list[float] | None = None

    @property
    def prefixo_arquivo(self) -> str:
        """Prefixo pra nomear os arquivos de saida (ex: `caracteres_completo`)."""
        return self.data_choice.value.lower()

    @property
    def acertos(self) -> int:
        return sum(1 for r in self.resultados_teste if r.acerto)

    @property
    def total_teste(self) -> int:
        return len(self.resultados_teste)

    @property
    def acuracia(self) -> float:
        return self.acertos / self.total_teste if self.total_teste else 0.0
