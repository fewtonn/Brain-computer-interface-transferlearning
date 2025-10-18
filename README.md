# Brain-Computer-Interface — Transfer Learning

Repositório dedicado à construção de um sistema de BCI (Interface Cérebro-Computador) focado no reconhecimento de **intenções de movimento da mão esquerda (T1)** e **mão direita (T2)** de forma generalizada.

---

## Visão geral
Este repositório reúne códigos para o fluxo completo de um experimento BCI baseado em sinais EEG:  
validação da recepção de dados do OpenBCI, processamento/formatacao de datasets, testes offline com várias técnicas de aprendizado de máquina e execução do BCI online (OpenBCI → Python) com suporte a *transfer learning*.

Objetivo: testar e comparar técnicas de ML (incluindo fine-tuning / transfer learning a partir de modelos treinados em grandes datasets de EEG) para classificação das intenções motoras (mão esquerda vs mão direita) por indivíduo.

---

## Estrutura do repositório

/ (raiz)
├─ Código para verificar openbci/
├─ códigos teste offline/
├─ códigos processamento de dados/
├─ New_bci_codes/
└─ README.md

### `Código para verificar openbci/`
Scripts para **validar a comunicação entre o OpenBCI e o Python**. Execute estes códigos primeiro para garantir que os dados estão sendo recebidos corretamente antes de rodar os demais módulos.

O que fazem:
- Testam a leitura dos pacotes do OpenBCI;
- Geram logs/prints que confirmam a chegada das amostras e a integridade básica do stream.

### `códigos teste offline/`
Três scripts que realizam **testes offline** usando diferentes técnicas de aprendizado de máquina. Cada script avalia dados por indivíduo (per-subject) para comparar performance entre técnicas.

Fluxo típico:
1. Uso de um modelo pré-treinado (treinado em grande dataset de EEG) para predições iniciais;
2. Treino de um modelo personalizado por indivíduo;
3. Validação individual do modelo;
4. Aplicação de *transfer learning* partindo do modelo pré-treinado.

### `códigos processamento de dados/`
Scripts para **formatar datasets de EEG** (com marcações reconhecidas) no formato esperado pelos códigos do OpenBCI/Python.

Observações:
- Existem 3 scripts nessa pasta.
- Cada script recebe uma pasta com os dados de **um indivíduo** contendo marcações:
  - `T1` — movimento da mão esquerda
  - `T2` — movimento da mão direita
  - `T0` — estado de repouso (nem todos os scripts usam T0)

Comportamento dos scripts:
- Dois scripts dividem os dados em **2 fases** (cada fase tem o tempo impresso no output):
  - Fase 1: transfer learning (treino/ajuste inicial);
  - Fase 2: teste do modelo em tempo real/streaming.
  - A diferença entre os dois está na **presença ou ausência de `T0`**.
- O terceiro script divide os dados em **3 fases**:
  - Fase para mão direita;
  - Fase para mão esquerda;
  - Terceira fase: todos os dados juntos para um teste final.
- Todos os scripts **informam no output o tempo de cada fase**, útil para sincronização e reprodutibilidade.

### `New_bci_codes/`
Códigos para execução do BCI online, conectando o OpenBCI ao pipeline Python.

Funcionalidades:
- Variáveis para definir tempos das fases e limites de execução;
- Integração com arquivos gerados pelos scripts de processamento e testes offline;
- Contém 3 scripts para rodar o OpenBCI em tempo real a partir dos arquivos formatados.

---

## Convenções de dados
- **Marcação de tarefas:** `T1` = mão esquerda, `T2` = mão direita, `T0` = repouso (quando presente).
- Cada pasta de indivíduo deve conter arquivos brutos com marcações temporais reconhecíveis para que os scripts de processamento possam convertê-los ao formato do OpenBCI.

---

## Uso sugerido (passo a passo)

1. **Verificar conexão OpenBCI**  
   Rode os scripts em `Código para verificar openbci` para garantir que o stream está chegando no Python.

2. **Formatar dados offline** (se aplicável)  
   Use os scripts em `códigos processamento de dados` para gerar os arquivos no formato esperado.

3. **Testar offline**  
   Rode os scripts em `códigos teste offline` para comparar técnicas, treinar modelos por indivíduo e aplicar transfer learning.

4. **Executar BCI online**  
   Use os scripts em `New_bci_codes` para conectar o OpenBCI ao pipeline Python e rodar o sistema em tempo real — o fluxo costuma ser: transfer learning primeiro, depois somente teste na segunda fase (conforme configurado nos scripts de processamento).

---

