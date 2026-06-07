"""Conferencia com o teste de mesa da professora (docs/exemplo+numérico+MLP.pdf):
rede 2 -> 3 -> 2 com pesos fixos, x=[1,1], t=[1,0] e alpha=0.5. Roda UM forward +
UM backpropagation com a nossa MLP e compara z_j, y_k, delta_k, delta_j^h e os
pesos atualizados com os valores do PDF (tolerancia 2e-4, que absorve o
arredondamento de 4 casas que o PDF carrega entre os passos).

Executado pelo main.py via data_choice TESTE_DE_MESA.

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

from entities import MLP


def _montar_rede_do_pdf() -> MLP:
    """Cria a MLP 2 -> 3 -> 2 e troca os pesos sorteados pelos fixos do PDF."""
    mlp = MLP([2, 3, 2])
    # Cada tupla é (bias, [pesos]); V = camada escondida, W = camada de saida.
    pesos_pdf = [
        [(-0.1, [0.1, -0.1]), (-0.1, [0.1, 0.1]), (0.1, [-0.1, -0.1])],  # V
        [(-0.1, [0.1, 0.0, 0.1]), (0.1, [-0.1, 0.1, -0.1])],  # W
    ]
    for camada, pesos_camada in zip(mlp.camadas, pesos_pdf):
        for neuronio, (bias, pesos) in zip(camada.neuronios, pesos_camada):
            neuronio.bias, neuronio.pesos = bias, pesos
    return mlp


def _conferencias(mlp: MLP) -> list[tuple[str, list[float], list[float]]]:
    """Lista (nome, valores obtidos pela nossa MLP, valores esperados no PDF)."""
    oculta, saida = mlp.camadas
    return [
        ("z_j", [n.saida for n in oculta.neuronios], [0.4750, 0.5250, 0.4750]),
        ("y_k", [n.saida for n in saida.neuronios], [0.4988, 0.5144]),
        ("delta_k", [n.delta for n in saida.neuronios], [0.1253, -0.1285]),
        ("delta_j^h", [n.delta for n in oculta.neuronios], [0.0063, -0.0032, 0.0063]),
        (
            "pesos W",  # por neuronio: w_0k (bias), w_1k, w_2k, w_3k
            [p for n in saida.neuronios for p in [n.bias, *n.pesos]],
            [-0.0373, 0.1298, 0.0329, 0.1298, 0.0358, -0.1305, 0.0663, -0.1305],
        ),
        (
            "pesos V",  # por neuronio: v_0j (bias), v_1j, v_2j
            [p for n in oculta.neuronios for p in [n.bias, *n.pesos]],
            [
                -0.0968,
                0.1032,
                -0.0968,
                -0.1016,
                0.0984,
                0.0984,
                0.1032,
                -0.0968,
                -0.0968,
            ],
        ),
    ]


def rodar_teste_de_mesa() -> None:
    """Confere a MLP contra o exemplo numerico do PDF; encerra com erro se divergir."""
    mlp = _montar_rede_do_pdf()

    # UM forward (passos 3-5 do algoritmo da pagina 294 do Fausett) +
    # UM backpropagation (passos 6-8) com a amostra do PDF.
    mlp.forward([1.0, 1.0])
    mlp.backpropagation([1.0, 0.0], taxa_aprendizado=0.5)

    falhas = 0
    for nome, obtidos, esperados in _conferencias(mlp):
        for obtido, esperado in zip(obtidos, esperados, strict=True):
            ok = abs(obtido - esperado) <= 2e-4
            falhas += not ok
            status = "OK " if ok else "ERRO"
            print(
                f"  [{status}] {nome:<9} obtido={obtido:+.4f}  esperado={esperado:+.4f}"
            )

    if falhas:
        raise SystemExit(f"FALHA: {falhas} valores divergem do teste de mesa do PDF.")
    print("\nSUCESSO: todos os valores conferem com o teste de mesa do PDF.")
