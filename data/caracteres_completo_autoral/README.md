# Variação AUTORAL do CARACTERES COMPLETO

Dado criado **pelo grupo** (não fornecido pela professora), para o quesito
"Os testes contaram com variações (autorais) do conjunto de teste".

- **`X_ruido.npy`**: cópia do `data/caracteres_completo/X.npy` com **10% dos
  pixels (12 de 120) invertidos** em cada imagem, escolhidos aleatoriamente
  com seed fixa (42).
- **Rótulos**: os mesmos do original (`data/caracteres_completo/Y_classe.npy`),
  na mesma ordem.
- **Uso**: somente como **conjunto de teste** — a rede treina/valida nos dados
  originais e é testada nesta variação, medindo robustez a ruído nunca visto.
- **Reprodução**: `python src/mlp/gerar_variacao_autoral.py`

## Integrantes

- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
