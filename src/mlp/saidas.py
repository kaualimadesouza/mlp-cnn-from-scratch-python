"""Geracao dos arquivos de saida do experimento (hiperparametros, pesos, csv, graficos).

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
from config import config
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
    camadas_iniciais: list[Camada]  # snapshot dos pesos ANTES do treino
    historico_erro: list[float]  # MSE de treino por epoca
    historico_validacao: list[float]  # MSE de validacao por epoca
    resultados_teste: list[ResultadoTeste]
    tempo_treino: float

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


# Helpers de descricao usados no arquivo de hiperparametros


def _descrever_preprocessamento(data_choice: DataChoiceEnum) -> str:
    """Retorna uma descricao em uma linha do pre-processamento feito no dataset."""
    if data_choice == DataChoiceEnum.CARACTERES_COMPLETO:
        return "Entradas/alvos em {0, 1} unipolar (X convertido de bipolar; Y one-hot nativo)"
    return "Entradas/alvos convertidos de bipolar {-1,+1} para unipolar {0, 1} (compat. sigmoide)"


def _descrever_split(data_choice: DataChoiceEnum) -> str:
    """Retorna uma descricao em uma linha do split do dataset."""
    if data_choice in {DataChoiceEnum.OR, DataChoiceEnum.AND, DataChoiceEnum.XOR}:
        return "Mesmas amostras em treino/validacao/teste (tabela verdade completa)"
    if data_choice == DataChoiceEnum.CARACTERES_REDUZIDO:
        return "80/10/10 (treino/validacao/teste) com shuffle sobre pool de limpo+ruido+ruido20"
    if data_choice == DataChoiceEnum.CARACTERES_COMPLETO:
        return "80/10/10 (treino/validacao/teste) com shuffle previo aleatorio"
    return "(desconhecido)"


# Funcoes de salvamento dos arquivos de saida


def salvar_erro_por_epoca(historico: list[float], caminho: str) -> None:
    """Salva o MSE de cada epoca em CSV (requisito da especificacao)."""
    with open(caminho, "w") as f:
        f.write("epoca,mse\n")
        for i, mse in enumerate(historico, 1):
            f.write(f"{i},{mse:.6f}\n")


def salvar_hiperparametros(
    mlp: MLP,
    dataset: Dataset,
    data_choice: DataChoiceEnum,
    taxa_aprendizado: float,
    epocas: int,
    caminho: str,
) -> None:
    """Salva hiperparametros da arquitetura + inicializacao (requisito da especificacao)."""
    num_entradas = mlp.tamanhos_camadas[0]
    num_ocultas = mlp.tamanhos_camadas[1]
    num_saidas = mlp.tamanhos_camadas[-1]

    # Contagem de parametros treinaveis
    pesos_entrada_oculta = num_entradas * num_ocultas
    bias_oculta = num_ocultas
    pesos_oculta_saida = num_ocultas * num_saidas
    bias_saida = num_saidas
    total_params = pesos_entrada_oculta + bias_oculta + pesos_oculta_saida + bias_saida

    total_amostras = len(dataset.treino) + len(dataset.validacao) + len(dataset.teste)
    pct_treino = len(dataset.treino) / total_amostras * 100
    pct_validacao = len(dataset.validacao) / total_amostras * 100
    pct_teste = len(dataset.teste) / total_amostras * 100

    with open(caminho, "w") as f:
        f.write("=== Hiperparametros do Experimento ===\n\n")
        f.write(
            f"Data do experimento:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        f.write("Grupo:\n")
        for integrante in config.integrantes:
            f.write(f"  - {integrante}\n")
        f.write("\n")

        f.write(f"Dataset:                    {data_choice.name}\n")
        f.write(
            f"Amostras treino:            {len(dataset.treino)} ({pct_treino:.0f}%)\n"
        )
        f.write(
            f"Amostras validacao:         {len(dataset.validacao)} ({pct_validacao:.0f}%)\n"
        )
        f.write(
            f"Amostras teste:             {len(dataset.teste)} ({pct_teste:.0f}%)\n"
        )
        f.write(f"Split:                      {_descrever_split(data_choice)}\n")
        f.write(
            f"Pre-processamento:          {_descrever_preprocessamento(data_choice)}\n"
        )

        f.write("\n--- Arquitetura (final) ---\n")
        f.write(f"Entradas:                   {num_entradas}\n")
        f.write(f"Camada oculta:              {num_ocultas} neuronios\n")
        f.write(f"Saidas:                     {num_saidas}\n")
        f.write("Ativacao:                   sigmoide (logistica)\n")
        f.write("Derivada:                   a(1 - a)  (a = saida do neuronio)\n")
        f.write(
            f"Total de parametros:        {total_params:,} "
            f"({pesos_entrada_oculta:,} + {bias_oculta} + "
            f"{pesos_oculta_saida:,} + {bias_saida})\n"
        )

        f.write("\n--- Inicializacao ---\n")
        f.write(f"Taxa aprendizado (alpha):   {taxa_aprendizado}\n")
        f.write(f"Epocas:                     {epocas}\n")
        f.write(
            "Inicializacao dos pesos:    Xavier/Glorot - Uniforme [-L, +L], L=sqrt(6/(fan_in+fan_out))\n"
        )
        f.write("Inicializacao do bias:      0.0 (todos os neuronios)\n")
        f.write("Condicao de parada:         Nenhuma (treina as N epocas completas)\n")
        f.write("Funcao de erro:             MSE (Erro Quadratico Medio)\n")
        f.write(
            "Algoritmo:                  Backpropagation com Gradiente Descendente\n"
        )


def acrescentar_resultados_finais(
    resultado: ResultadoExperimento, caminho: str
) -> None:
    """Acrescenta os resultados do treino/teste no final do arquivo de hiperparametros."""
    with open(caminho, "a") as f:
        f.write("\n--- Resultados Finais ---\n")
        f.write(f"MSE inicial:   {resultado.historico_erro[0]:.6f}\n")
        f.write(f"MSE final:     {resultado.historico_erro[-1]:.6f}\n")
        f.write(
            f"Acuracia:      {resultado.acuracia:.2%} "
            f"({resultado.acertos}/{resultado.total_teste})\n"
        )
        f.write(f"Tempo treino:  {resultado.tempo_treino:.1f}s\n")


def salvar_pesos(camadas: list[Camada], titulo: str, caminho: str) -> None:
    """Salva os pesos e bias de cada neuronio em texto, camada por camada.

    Funciona tanto pra pesos iniciais quanto finais - é so passar o snapshot
    certo (resultado.camadas_iniciais ou resultado.mlp.camadas) e o titulo.
    """
    with open(caminho, "w") as f:
        f.write(f"=== {titulo} ===\n\n")
        for i, camada in enumerate(camadas, 1):
            tipo = "saida" if i == len(camadas) else "oculta"
            f.write(f"--- Camada {i} ({tipo}, {len(camada.neuronios)} neuronios) ---\n")
            for j, neuronio in enumerate(camada.neuronios, 1):
                pesos_str = ", ".join(f"{p:.6f}" for p in neuronio.pesos)
                f.write(
                    f"Neuronio {j}: bias={neuronio.bias:.6f} | pesos=[{pesos_str}]\n"
                )
            f.write("\n")


def salvar_saidas_teste(resultados: list[ResultadoTeste], caminho: str) -> None:
    """Salva as saidas da MLP para cada amostra de teste em CSV."""
    with open(caminho, "w") as f:
        f.write(
            "amostra,classe_esperada,classe_predita,acerto,esperado_raw,saida_raw\n"
        )
        for i, r in enumerate(resultados, 1):
            esperado_str = ", ".join(f"{v:.0f}" for v in r.esperado_raw)
            saida_str = ", ".join(f"{v:.4f}" for v in r.saida_raw)
            acerto_str = "sim" if r.acerto else "nao"
            f.write(
                f"{i},{r.classe_esperada},{r.classe_predita},{acerto_str},"
                f'"[{esperado_str}]","[{saida_str}]"\n'
            )


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
