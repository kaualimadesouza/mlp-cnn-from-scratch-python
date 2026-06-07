# Pendências da entrega — Trabalho de IA (MLP)

> **Prazo: 9 de junho, 23h55** (e-Disciplinas). A spec recomenda upload com ~1 dia
> de antecedência — ideal subir tudo até **8 de junho**.

## Slots de entrega no e-Disciplinas

| # | Slot | Status | O que falta |
|---|------|--------|-------------|
| 1 | **Códigos MLP** | 🟡 Quase pronto | Código completo e commitado. Falta só zipar/exportar pra upload (ver detalhes abaixo). |
| 2 | **Checklist preenchido — quesitos obrigatórios e opcionais** | 🔴 Não começado | Baixar o modelo do e-Disciplinas e preencher: pra cada conceito, indicar **arquivo + linhas principais** do código. |
| 3 | **Slides (usado no vídeo)** | 🔴 Não começado | Baixar o **modelo de slides** do e-Disciplinas (NÃO pode ser alterado; slides extras são permitidos) e preencher seguindo o roteiro. |
| 4 | **Vídeo — MLP** | 🔴 Não começado | Gravar vídeo de **10 a 20 min** (ver requisitos abaixo). |
| 5 | **Minutagem — MLP** | ⚪ Condicional | **Só** se o vídeo for hospedado fora do e-Disciplinas (ex: YouTube). Nesse caso, PDF com minuto de início/fim de cada trecho. Se o vídeo for postado no e-Disciplinas, **não** enviar. |
| 6 | **Outros arquivos** | ⚪ Opcional | Só se houver algo complementar. A spec pede pra **não** postar arquivos não solicitados. |

## Detalhes por item

### 1. Códigos MLP
- [x] MLP from scratch (forward, backprop, gradiente descendente) — `src/mlp/`
- [x] 5 datasets rodando (OR, AND, XOR, CARACTERES_REDUZIDO, CARACTERES_COMPLETO)
- [x] Arquivos de saída gerados e commitados (`saidas/mlp/`)
- [x] Nome + NUSP de todos os integrantes nos arquivos de código
- [ ] Preparar o pacote pra upload (zip com `src/`, `saidas/`, `README.md` — confirmar formato pedido)

### 2. Checklist (obrigatórios + opcionais)
- [x] Baixar o modelo de checklist no e-Disciplinas (`docs/Checklist.docx`)
- [x] Preencher itens já implementados com arquivo + linhas → **`docs/Checklist_preenchido.docx`**
- [ ] Marcar os itens de vídeo/upload depois que forem concluídos
- [ ] Decidir sobre os quesitos ainda não cobertos (lista abaixo)

#### ⚠️ Quesitos do checklist que o código ainda NÃO cobre

Deixados **em branco** no `Checklist_preenchido.docx`. Se implementarmos, atualizar o docx:

- [x] **Parada antecipada** (seção Implementação) — implementada em
      `entities.py` (paciência sobre o MSE de validação) e configurada por
      dataset em `config.py`. Testada: CARACTERES_COMPLETO parou na época
      208/300 e a acurácia subiu de 88.06% para **94.03%**.
- [x] **Busca de parâmetros** (seção Testes) — grid search em
      `src/mlp/busca_parametros.py` (taxa × neurônios × paciência), resultados em
      `saidas/mlp/caracteres_completo/busca_parametros.csv`. Confirma a config
      atual (taxa 0.04, 55 neurônios, paciência 15) como a melhor faixa de
      acurácia (91.04%) com o menor MSE que ainda aproveita a parada antecipada.
- [x] **Variações (autorais) do conjunto de teste** (seção Testes) — ruído
      autoral (10% dos pixels invertidos, seed fixa) gerado por
      `src/mlp/gerar_variacao_autoral.py` em `data/caracteres_completo_autoral/`;
      treina/valida nos dados limpos e testa só na versão ruidosa (data_choice
      `CARACTERES_COMPLETO_AUTORAL`).
- [x] **Hold-out × validação cruzada** (seção opcional) — `MetodoValidacaoEnum`
      escolhe por dataset; k-fold em `src/mlp/validacao.py`, diagrama em
      `saidas/mlp/<dataset>/validacao_cruzada.png` (REDUZIDO k=7, COMPLETO k=5).
- [x] **Teste com outro conjunto de dados** (seção opcional) — Iris (dataset
      externo) em `data/iris/`, data_choice `IRIS`, 100% de acurácia no teste.
- [x] **Conferência com o "Apoio a testes de mesa (MLP)"** (seção Procedimentos) —
      feita em `src/mlp/teste_de_mesa.py`: reproduz o exemplo numérico do
      `docs/exemplo+numérico+MLP.pdf` (rede 2→3→2) e confere os 27 valores
      (z, y, deltas e pesos atualizados). Resultado: **SUCESSO**.
      Roda automaticamente no início do `make run-mlp` (data_choice `TESTE_DE_MESA`).

### 3. Slides
- [ ] Baixar o **modelo oficial** (não alterar o modelo; ordem do roteiro não pode mudar)
- [ ] Incluir nome + NUSP de todos os integrantes
- [ ] Preencher conteúdo seguindo o roteiro do e-Disciplinas

### 4. Vídeo (10–20 min)
Requisitos da spec:
- [ ] **Todos os 4 integrantes** participam, cada um explicando algum aspecto, **com janela de webcam**
- [ ] Seguir o **roteiro** do e-Disciplinas (ordem dos itens não pode ser alterada)
- [ ] Voz em velocidade normal (pode acelerar só a tela durante treino longo)
- [ ] Mostrar **treinamento** da MLP no CARACTERES COMPLETO (portas lógicas e Fausett não precisam aparecer)
- [ ] Mostrar **teste** da MLP no CARACTERES COMPLETO
- [ ] Comentar qual parte dos dados é **treino** (80% = 1060 amostras)
- [ ] Comentar qual parte é **validação** (10% = 132 amostras)
- [ ] Comentar qual parte é **teste** (10% = 134 amostras)
- [ ] Mostrar o **console** acompanhando o treinamento (MSE por época)
- [ ] Mostrar **gráfico de comportamento de erros** (`saidas/mlp/caracteres_completo/mse.png`)
- [ ] Mostrar **resultados no console** para os dados de teste (acurácia 88.06%)
- [ ] Mostrar a **matriz de confusão** (`saidas/mlp/caracteres_completo/matriz_confusao.png`)

### 5. Minutagem (condicional)
- [ ] Decidir onde hospedar o vídeo (e-Disciplinas ou repositório público)
- [ ] Se for fora do e-Disciplinas: gerar PDF com minutagem (início/fim de cada trecho)

## Lembretes gerais da spec
- Nome + NUSP de **todos** os integrantes em **todos** os artefatos (código, vídeo, slides, PDFs)
- Boa estética em tudo (desleixo pode zerar o artefato)
- Não postar arquivos não solicitados
