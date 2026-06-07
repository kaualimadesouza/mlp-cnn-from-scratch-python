"""MLP com uma camada escondida, seguindo a nomenclatura do Fausett (secao 6.1.2):
v_ij, w_jk, z_in_j, z_j, y_in_k, y_k, t_k, alpha, delta. Bias ficou separado de
`pesos` por clareza (equivalente ao v_0j do Fausett, so muda a representacao).

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

import math
import random

from value_objects import Amostra, ResultadoTeste


def sigmoide(x: float) -> float:
    """f(x) = 1 / (1 + e^(-x)). Saida em (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def sigmoide_derivada_de_saida(a: float) -> float:
    """f'(z) = a(1 - a), onde a = f(z) ja esta calculado."""
    return a * (1 - a)


class Neuronio:
    """Um neuronio: calcula z_in = Soma(w*x) + bias e aplica sigmoide."""

    def __init__(self, pesos: list[float], bias: float):
        self.pesos: list[float] = pesos
        self.bias: float = bias
        # Preenchidos a cada forward e consumidos no backprop da mesma amostra.
        self.entradas: list[float] = []
        self.soma_ponderada: float = 0.0
        self.saida: float = 0.0
        # delta = info de erro do backprop (delta_k na saida, delta_j^h na oculta)
        self.delta: float = 0.0

    def calcular_saida(self, entradas: list[float]) -> float:
        """z_in = Soma(w*x) + bias; saida = sigmoide(z_in). Guarda tudo pra usar no backprop."""
        self.entradas = entradas

        soma = 0.0
        for entrada, peso in zip(entradas, self.pesos):
            soma += entrada * peso

        # z_in -> pre-ativacao usada no backprop pra f'(z_in).
        self.soma_ponderada = soma + self.bias
        # saida = f(z_in) -> pos-ativacao no intervalo (0, 1).
        self.saida = sigmoide(self.soma_ponderada)
        return self.saida


class Camada:
    """Conjunto de neuronios que operam sobre a mesma entrada."""

    def __init__(self, num_neuronios: int, num_entradas: int):
        self.neuronios: list[Neuronio] = []

        # Init uniforme com faixa adaptada ao tamanho da camada. Faixa maior em camadas
        # pequenas (ajuda XOR a sair do platao), menor em camadas grandes (evita saturar
        # a sigmoide). Ex: XOR (2->4) -> L=1.00; CARACTERES (120->55) -> L~=0.19.
        limite = math.sqrt(6.0 / (num_entradas + num_neuronios))
        for _ in range(num_neuronios):
            pesos = [random.uniform(-limite, limite) for _ in range(num_entradas)]
            bias = 0.0
            self.neuronios.append(Neuronio(pesos, bias))


class MLP:
    """MLP com uma camada escondida: y = f(W * f(V * x))."""

    def __init__(self, tamanhos_camadas: list[int]):
        """Ex: tamanhos_camadas = [120, 55, 26] -> 120 entradas, 55 ocultos, 26 saidas."""
        self.tamanhos_camadas = tamanhos_camadas
        self.camadas: list[Camada] = []

        # Camada de entrada nao tem neuronios "proprios" - ela so alimenta a oculta.
        # Por isso comecamos em i=1: a camada i recebe n_{i-1} entradas.
        for i in range(1, len(tamanhos_camadas)):
            camada = Camada(tamanhos_camadas[i], tamanhos_camadas[i - 1])
            self.camadas.append(camada)

    def forward(self, entrada: list[float]) -> list[float]:
        """Propaga x pela rede e devolve o vetor de saida y."""

        ativacao = entrada

        # Propaga camada por camada: a saida de cada camada vira a entrada da proxima.
        for camada in self.camadas:
            novas_saidas: list[float] = []
            # Itera sobre os neuronios da camada atual
            for neuronio in camada.neuronios:
                # Cada neuronio calcula sua saida a partir da ativacao (entrada) atual e armazena
                neuronio.calcular_saida(ativacao)
                novas_saidas.append(neuronio.saida)
            ativacao = novas_saidas
        return ativacao

    def backpropagation(self, esperado: list[float], taxa_aprendizado: float) -> None:
        """Propaga o erro de volta e atualiza pesos/bias pela regra delta."""

        # Para cada neuronio da camada de saida, calcula delta_k = (t_k - y_k) * f'(y_in_k).
        for k, neuronio in enumerate(self.camadas[-1].neuronios):
            erro = esperado[k] - neuronio.saida
            neuronio.delta = erro * sigmoide_derivada_de_saida(neuronio.saida)

        # Percorre as camadas ocultas do final pro inicio
        for j in range(len(self.camadas) - 2, -1, -1):
            camada_atual = self.camadas[j]
            camada_acima = self.camadas[j + 1]

            # Para cada neuronio da camada atual, calcula delta_j^h = (Soma delta_k * w_jk) * f'(z_in_j).
            for i, neuronio in enumerate(camada_atual.neuronios):
                erro = 0.0
                for neuronio_acima in camada_acima.neuronios:
                    erro += neuronio_acima.delta * neuronio_acima.pesos[i]
                neuronio.delta = erro * sigmoide_derivada_de_saida(neuronio.saida)

        # Com o delta de cada neuronio calculado, atualiza os pesos e bias da rede
        for camada in self.camadas:
            for neuronio in camada.neuronios:
                # Para cada entrada, atualiza o peso respectivo
                for i, entrada in enumerate(neuronio.entradas):
                    # Novo peso = peso antigo + ajuste, onde ajuste = taxa_aprendizado * delta * entrada
                    neuronio.pesos[i] += taxa_aprendizado * neuronio.delta * entrada
                # Novo bias = bias antigo + ajuste, onde ajuste = taxa_aprendizado * delta * 1 (x_0 = 1)
                neuronio.bias += taxa_aprendizado * neuronio.delta

    def treinar(
        self,
        dados_treino: list[Amostra],
        taxa_aprendizado: float,
        epocas: int,
        dados_validacao: list[Amostra] | None = None,
        paciencia: int | None = None,
    ) -> tuple[list[float], list[float]]:
        """Treina por backpropagation e retorna (historico_treino, historico_validacao)."""
        historico_treino: list[float] = []
        historico_validacao: list[float] = []
        # padding pra alinhar numero da epoca no print
        largura = len(str(epocas))
        epocas_sem_melhora = 0
        melhor_mse_val = float("inf")

        for epoca in range(epocas):
            # Shuffle por epoca: evita decorar a ordem das amostras e ajuda generalizacao.
            random.shuffle(dados_treino)
            erro_treino = 0.0
            # Treino: pra cada amostra, faz forward, backprop e acumula o erro (t_k - y_k)^2.
            for amostra in dados_treino:
                # Forward: propaga a entrada e calcula a saida da rede ate a camada de saida.
                saida = self.forward(amostra.entrada)
                # Backpropagation: propaga o erro de volta e atualiza os pesos da rede.
                self.backpropagation(amostra.esperado, taxa_aprendizado)
                # Acumula somatorio de (t_k - y_k)^2 pra depois virar MSE da epoca.
                for y, t in zip(saida, amostra.esperado):
                    erro_treino += (t - y) ** 2
            mse_treino = erro_treino / len(dados_treino)
            historico_treino.append(mse_treino)

            # MSE de validacao: so forward, sem backpropagation.
            msg_validacao = ""
            if dados_validacao:
                erro_val = 0.0
                for amostra in dados_validacao:
                    # Apenas fazemos forward nas amostras de validacao, sem atualizar pesos, e acumulamos o erro.
                    saida = self.forward(amostra.entrada)
                    for y, t in zip(saida, amostra.esperado):
                        erro_val += (t - y) ** 2
                mse_val = erro_val / len(dados_validacao)
                historico_validacao.append(mse_val)
                msg_validacao = f"  |  Val: {mse_val:.6f}"

                if mse_val < melhor_mse_val:
                    melhor_mse_val = mse_val
                    epocas_sem_melhora = 0
                else:
                    epocas_sem_melhora += 1
                    if paciencia and epocas_sem_melhora >= paciencia:
                        print(
                            f"Parando por paciencia: {epocas_sem_melhora} epocas sem melhora no MSE de validacao."
                        )
                        break

            print(
                f"Epoca {epoca + 1:>{largura}}/{epocas} - MSE: {mse_treino:.6f}{msg_validacao}"
            )

        # Retornamos historicos de MSE pra plotar grafico depois. Se nao tiver validacao, o segundo vetor fica vazio.
        return historico_treino, historico_validacao

    def testar(self, dados_teste: list[Amostra]) -> list[ResultadoTeste]:
        """Avalia a rede em cada amostra de teste e devolve lista de `ResultadoTeste`."""
        resultados: list[ResultadoTeste] = []
        for amostra in dados_teste:
            y = self.forward(amostra.entrada)
            if len(amostra.esperado) == 1:
                # Binario (portas logicas): threshold 0.5 (ponto medio da sigmoide) decide 0 ou 1.
                predita = 1 if y[0] > 0.5 else 0
                esperada = 1 if amostra.esperado[0] > 0.5 else 0
            else:
                # Multiclasse (caracteres): classe = indice do neuronio com maior ativacao.
                predita = y.index(max(y))
                esperada = amostra.esperado.index(max(amostra.esperado))
            resultados.append(
                ResultadoTeste(
                    entrada=amostra.entrada,
                    esperado_raw=amostra.esperado,
                    saida_raw=y,
                    classe_predita=predita,
                    classe_esperada=esperada,
                    acerto=(predita == esperada),
                )
            )
        return resultados
