"""Objetivo 1: MLP from scratch with Backpropagation.

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

import os
import time

from config import HIPERPARAMETROS, config
from datasets import carregar_dados
from entities import MLP, Camada, Neuronio
from graficos import (
    salvar_grafico_mse,
    salvar_grafico_validacao_cruzada,
    salvar_matriz_confusao,
)
from resultado import ResultadoExperimento
from teste_de_mesa import rodar_teste_de_mesa
from validacao import validacao_cruzada
from value_objects import DataChoiceEnum, Dataset, MetodoValidacaoEnum

from saidas import (
    acrescentar_resultados_finais,
    salvar_erro_por_epoca,
    salvar_hiperparametros,
    salvar_pesos,
    salvar_saidas_teste,
)


def run(
    data_choice: DataChoiceEnum = DataChoiceEnum.CARACTERES_COMPLETO,
    taxa_aprendizado: float = 0.1,
    epocas: int = 1000,
    num_neuronios_oculta: int = 10,
    paciencia: int | None = None,
    metodo_validacao: MetodoValidacaoEnum = MetodoValidacaoEnum.HOLD_OUT,
    k_folds: int | None = None,
) -> ResultadoExperimento:
    """Carrega os dados, treina e testa a MLP. Retorna um ResultadoExperimento."""
    # 1. Carregar os dados, de forma que ja fiquem no formato certo pra treinar a MLP.
    dataset: Dataset = carregar_dados(data_choice)

    # 2. Definir arquitetura e criar a MLP com inicializacao de pesos.
    # Sao 3 camadas: entrada, oculta e saida.
    # Entrada = numero de atributos do dataset
    # Ocultas = hiperparametro definido pelo grupo
    # Saida = numero de classes/rotulos
    arquitetura = [dataset.num_entradas, num_neuronios_oculta, dataset.num_saidas]
    mlp = MLP(arquitetura)

    # Salva os hiperparametros (arquitetura + inicializacao) no comeco do experimento.
    caminho_hp = config.caminho_saida(
        data_choice.value.lower(), config.nome_arquivo_hiperparametros
    )
    salvar_hiperparametros(
        mlp=mlp,
        dataset=dataset,
        data_choice=data_choice,
        taxa_aprendizado=taxa_aprendizado,
        epocas=epocas,
        paciencia=paciencia,
        caminho=caminho_hp,
    )

    # Snapshot dos pesos ANTES do treino (pra salvar como pesos iniciais depois).
    # Copiamos manualmente pra evitar guardar referencias aos mesmos neuronios
    # (senao os "pesos iniciais" mudariam junto com o treino).
    camadas_iniciais: list[Camada] = []
    for camada in mlp.camadas:
        # Cria uma nova camada (o construtor sorteia novos pesos, mas vamos substituir).
        nova_camada = Camada(len(camada.neuronios), len(camada.neuronios[0].pesos))
        nova_camada.neuronios = []
        for neuronio in camada.neuronios:
            # Copia os pesos (list() cria uma nova lista) e o bias, neuronio por neuronio.
            copia = Neuronio(list(neuronio.pesos), neuronio.bias)
            nova_camada.neuronios.append(copia)
        camadas_iniciais.append(nova_camada)

    print(f"Dataset: {data_choice.name}")
    print(f"Amostras: {len(dataset.treino)} treino / {len(dataset.teste)} teste")
    print(f"Arquitetura: {' -> '.join(str(n) for n in arquitetura)} (sigmoide)")
    print(f"Taxa de aprendizado: {taxa_aprendizado}")
    print(f"Epocas: {epocas}")

    # K-fold sobre treino+validacao (teste fica de fora); o modelo final
    # continua sendo o do treino hold-out abaixo.
    acuracias_cv = None
    if metodo_validacao == MetodoValidacaoEnum.CROSS_VALIDATION:
        if not k_folds:
            raise ValueError("CROSS_VALIDATION exige k_folds definido no config")
        print(f"\n--- Validacao cruzada ({k_folds}-fold) ---")
        acuracias_cv = validacao_cruzada(
            dataset.treino + dataset.validacao,
            arquitetura,
            taxa_aprendizado,
            epocas,
            k_folds,
            paciencia,
        )
        media = sum(acuracias_cv) / len(acuracias_cv)
        desvio = (
            sum((a - media) ** 2 for a in acuracias_cv) / len(acuracias_cv)
        ) ** 0.5
        print(f"Acuracia media dos folds: {media:.2%} +- {desvio:.2%}")

    print("\n--- Treinamento ---")

    # 3. Treinar a MLP. Passa a validacao junto pra receber os dois historicos (treino+val).
    t0 = time.time()
    historico, historico_validacao = mlp.treinar(
        dataset.treino,
        taxa_aprendizado,
        epocas,
        dados_validacao=dataset.validacao,
        paciencia=paciencia,
    )
    tempo_treino = time.time() - t0

    # 4. Testar a MLP (depois de treinada, com os pesos finais).
    print("\n--- Teste ---")
    resultados = mlp.testar(dataset.teste)
    acertos = sum(1 for r in resultados if r.acerto)
    total = len(resultados)
    print(f"Acuracia: {acertos}/{total} ({acertos / total:.2%})")

    return ResultadoExperimento(
        data_choice=data_choice,
        dataset=dataset,
        arquitetura=arquitetura,
        taxa_aprendizado=taxa_aprendizado,
        epocas=epocas,
        num_neuronios_oculta=num_neuronios_oculta,
        mlp=mlp,
        camadas_iniciais=camadas_iniciais,
        historico_erro=historico,
        historico_validacao=historico_validacao,
        resultados_teste=resultados,
        tempo_treino=tempo_treino,
        acuracias_cv=acuracias_cv,
    )


def main(
    data_choice: DataChoiceEnum = DataChoiceEnum.CARACTERES_COMPLETO,
    taxa_aprendizado: float = 0.1,
    epocas: int = 1000,
    num_neuronios_oculta: int = 10,
    paciencia: int | None = None,
    metodo_validacao: MetodoValidacaoEnum = MetodoValidacaoEnum.HOLD_OUT,
    k_folds: int | None = None,
):
    """Executa o experimento e salva todos os arquivos de saida pedidos."""
    # TESTE_DE_MESA nao é um treino: roda a conferencia com o exemplo numerico
    # da professora (forward + backprop unicos) e nao gera arquivos de saida.
    if data_choice == DataChoiceEnum.TESTE_DE_MESA:
        print("--- Teste de mesa (conferencia com docs/exemplo+numérico+MLP.pdf) ---")
        rodar_teste_de_mesa()
        return

    os.makedirs(config.pasta_saidas, exist_ok=True)

    # Executa o experimento completo (carregar dados, treinar, testar).
    resultado = run(
        data_choice=data_choice,
        taxa_aprendizado=taxa_aprendizado,
        epocas=epocas,
        num_neuronios_oculta=num_neuronios_oculta,
        paciencia=paciencia,
        metodo_validacao=metodo_validacao,
        k_folds=k_folds,
    )

    prefixo = resultado.prefixo_arquivo
    print("\n--- Arquivos salvos ---")

    # Erro por epoca
    caminho_erro = config.caminho_saida(prefixo, config.nome_arquivo_erro_por_epoca)
    salvar_erro_por_epoca(resultado.historico_erro, caminho_erro)
    print(f"Erro por epoca:    {caminho_erro}")

    # Hiperparametros: o arquivo ja foi criado no `run()` antes do treino.
    # Agora so acrescentamos os resultados finais no mesmo arquivo.
    caminho_hp = config.caminho_saida(prefixo, config.nome_arquivo_hiperparametros)
    acrescentar_resultados_finais(resultado, caminho_hp)
    print(f"Hiperparametros:   {caminho_hp}")

    caminho_pi = config.caminho_saida(prefixo, config.nome_arquivo_pesos_iniciais)
    salvar_pesos(resultado.camadas_iniciais, "Pesos Iniciais", caminho_pi)
    print(f"Pesos iniciais:    {caminho_pi}")

    caminho_pf = config.caminho_saida(prefixo, config.nome_arquivo_pesos_finais)
    salvar_pesos(resultado.mlp.camadas, "Pesos Finais", caminho_pf)
    print(f"Pesos finais:      {caminho_pf}")

    caminho_st = config.caminho_saida(prefixo, config.nome_arquivo_saidas_teste)
    salvar_saidas_teste(resultado.resultados_teste, caminho_st)
    print(f"Saidas de teste:   {caminho_st}")

    # Grafico de evolucao do MSE (treino + validacao)
    caminho_grafico = config.caminho_saida(prefixo, "mse.png")
    salvar_grafico_mse(resultado, caminho_grafico)
    print(f"Grafico MSE:       {caminho_grafico}")

    # Matriz de confusao (heatmap PNG)
    caminho_matriz = config.caminho_saida(prefixo, "matriz_confusao.png")
    salvar_matriz_confusao(resultado.resultados_teste, caminho_matriz)
    print(f"Matriz confusao:   {caminho_matriz}")

    if resultado.acuracias_cv:
        caminho_cv = config.caminho_saida(prefixo, "validacao_cruzada.png")
        salvar_grafico_validacao_cruzada(resultado.acuracias_cv, caminho_cv)
        print(f"Validacao cruzada: {caminho_cv}")


if __name__ == "__main__":
    # Teste de mesa primeiro: valida a corretude do forward/backprop
    # main(data_choice=DataChoiceEnum.TESTE_DE_MESA)
    hp = HIPERPARAMETROS[DataChoiceEnum.CARACTERES_COMPLETO]

    main(
        data_choice=DataChoiceEnum.CARACTERES_COMPLETO,
        taxa_aprendizado=hp.taxa_aprendizado,
        epocas=hp.epocas,
        num_neuronios_oculta=hp.num_neuronios_oculta,
        paciencia=hp.paciencia,
        metodo_validacao=hp.metodo_validacao,
        k_folds=hp.k_folds,
    )
