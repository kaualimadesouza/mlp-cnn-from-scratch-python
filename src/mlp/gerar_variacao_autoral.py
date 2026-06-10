"""Gera a variacao AUTORAL do conjunto CARACTERES COMPLETO (quesito do checklist).

Pega o X.npy original e inverte 10% dos pixels de cada imagem, escolhidos
aleatoriamente (seed fixa = reproduzivel). O resultado vai para
data/caracteres_completo_autoral/X_ruido.npy - os rotulos sao os mesmos do
original, entao so os pixels mudam.

Esse dado é usado APENAS como conjunto de teste (a rede treina nos dados
originais), pra medir a robustez da MLP a um ruido que ela nunca viu.

Rodar: python src/mlp/gerar_variacao_autoral.py

Integrantes:
- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
"""

import os
import random

import numpy as np

TAXA_RUIDO = 0.10

random.seed(42)

# X.npy original: (N, 10, 12, 1) em bipolar {-1, +1}.
X = np.load("data/caracteres_completo/X.npy")
X_ruido = X.copy().reshape(len(X), -1)

num_invertidos = int(X_ruido.shape[1] * TAXA_RUIDO)
for imagem in X_ruido:
    # Sorteia quais pixels inverter (sem repeticao) e troca o sinal (-1 <-> +1).
    for i in random.sample(range(imagem.size), num_invertidos):
        imagem[i] *= -1

os.makedirs("data/caracteres_completo_autoral", exist_ok=True)
np.save("data/caracteres_completo_autoral/X_ruido.npy", X_ruido.reshape(X.shape))
print(
    f"Gerado: data/caracteres_completo_autoral/X_ruido.npy "
    f"({len(X)} imagens, {num_invertidos}/{X_ruido.shape[1]} pixels invertidos cada)"
)
