import os
import mne
import numpy as np
import pandas as pd
import time
import fnmatch
from io import StringIO
# ESSE CÓDIGO TEM COMO FUNÇÃO PEGAR TAREFAS ESPECIFICAS DO DATASET BRUTO E DIVIDIR EM2 FASES COM APENAS T1 E T2
# ============================================================================== #
# CONFIGURAÇÃO
# ============================================================================== #
DIRETORIO_ARQUIVOS = r"C:\Users\batis\OneDrive\Área de Trabalho\Códigos diversos\Faculdade\BCI\Dados BRUTOS\files\S086"
ARQUIVO_SAIDA = r"C:\Users\batis\OneDrive\Área de Trabalho\Brain Computer Interface\TESTE_TUDOJUNTO_SEMT0.csv"
PADROES_ARQUIVOS = ["*R04.edf", "*R08.edf", "*R12.edf"]
LISTA_EPOCAS = ['T1', 'T2']
CANAIS_DESEJADOS = ['Fp1.', 'F7..', 'F3..', 'T7..', 'C3..', 'P7..', 'P3..', 'O1..',
                    'Fp2.', 'F4..', 'F8..', 'C4..', 'T8..', 'P4..', 'P8..', 'O2..']
CASAS_DECIMAIS = 4
SAMPLE_RATE = 160  
RANDOM_SEED = None  # opcional: defina um inteiro pra reprodutibilidade

# ============================================================================== #
# FUNÇÕES
# ============================================================================== #
def carregar_edfs(diretorio, padroes):
    arquivos = []
    for padrao in padroes:
        arquivos.extend([os.path.join(diretorio, f) for f in os.listdir(diretorio)
                         if f.endswith(".edf") and fnmatch.fnmatch(f, padrao)])
    return arquivos

def extrair_epocas(arquivo, eventos_desejados):
    raw = mne.io.read_raw_edf(arquivo, preload=True, verbose=False)
    raw.pick_channels(CANAIS_DESEJADOS)
    eventos, ids = mne.events_from_annotations(raw)
    epocas_dict = {}
    for evento in eventos_desejados:
        if evento in ids:
            epochs = mne.Epochs(raw, eventos, event_id={evento: ids[evento]},
                                 tmin=-0.5, tmax=4, baseline=(-0.5, 0), preload=True, verbose=False)
            epocas = epochs.get_data() * 1e6  # para microvolts, como estava antes
            epocas = np.round(epocas, decimals=CASAS_DECIMAIS)
            epocas_dict[evento] = epocas
    return epocas_dict, raw.info

def dividir_fases(epocas_dict, prop=0.8,):
   
    t1 = np.array(epocas_dict.get('T1', []))
    t2 = np.array(epocas_dict.get('T2', []))
    #t0 = np.array(epocas_dict.get('T0', []))

    n1, n2 = len(t1), len(t2)
    n1_treino, n2_treino = int(n1 * prop), int(n2 * prop)

    fase1 = t1[:n1_treino]
    fase2 = t2[:n2_treino]
    t1_teste = t1[n1_treino:]
    t2_teste = t2[n2_treino:]

    n0_max = int((len(t1_teste) + len(t2_teste)) )
    #t0 = t0[:n0_max]

    fase3 = []
    ordem_f3 = []
    for i in range(max(len(t1_teste), len(t2_teste))):
        if i < len(t1_teste):
            fase3.append(t1_teste[i])
            ordem_f3.append('T1')
       # if i < len(t0):
        #    fase3.append(t0[i])
        #    ordem_f3.append('T0')
        if i < len(t2_teste):
            fase3.append(t2_teste[i])
            ordem_f3.append('T2')

    return fase1, fase2, fase3, ordem_f3

def salvar_csv_completo_embaralhado(epochs_array, labels_list, caminho, sample_rate=SAMPLE_RATE):
    """
    Salva o CSV sem adicionar colunas Epoch, Label ou Transition.
    Retorna info sobre o ponto de transição (início da 2ª metade).
    """
    n_epocas, n_canais, n_tempos = epochs_array.shape

    # Transpor e achatar (mesmo formato que já vinha sendo usado)
    dados_ajustados = epochs_array.transpose(0, 2, 1).reshape(n_epocas * n_tempos, n_canais)
    df = pd.DataFrame(dados_ajustados, columns=[f'EXG Channel {i}' for i in range(n_canais)])

    # Colunas padrão (mantendo o formato estilo OpenBCI que você usava)
    df.insert(0, 'Sample Index', range(1, len(df) + 1))
    df['Accel Channel 0'] = 0
    df['Accel Channel 1'] = 0
    df['Accel Channel 2'] = 0
    for i in range(7):
        df[f'Other_{i}'] = 0
    df['Analog Channel 0'] = 0
    df['Analog Channel 1'] = 0
    df['Analog Channel 2'] = 0

    # Timestamp (opcional — mantive porque já estava presente nas versões anteriores)
    total_samples = n_epocas * n_tempos
    timestamps = np.arange(total_samples) / sample_rate
    df['Timestamp'] = np.round(timestamps, 6)
    df['Other'] = 0
    df['Timestamp (Formatted)'] = pd.to_timedelta(df['Timestamp'], unit='s').astype(str)

    # NÃO adicionar: Epoch, Label, Transition (nem qualquer coluna extra)
    header = [
        '%OpenBCI Raw EXG Data\n',
        f'%Number of channels = {n_canais}\n',
        f'%Sample Rate = {sample_rate} Hz\n',
        '%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n'
    ]

    with open(caminho, 'w', newline='') as f:
        f.writelines(header)
        f.write(','.join(df.columns) + '\n')
        zero_line = ','.join(['0'] * df.shape[1]) + '\n'
        f.write(zero_line * 1000)
        buffer = StringIO()
        df.to_csv(buffer, index=False, header=False, lineterminator='\n')
        f.write(buffer.getvalue())

    # Calcular internamente o ponto de transição (sem escrever no arquivo)
    metade_epocas = n_epocas // 2
    transicao_sample_index = metade_epocas * n_tempos
    trans_time = float(timestamps[transicao_sample_index]) if transicao_sample_index < total_samples else None
    trans_sample_1based = transicao_sample_index + 1

    return {
        'n_epochs': n_epocas,
        'n_times': n_tempos,
        'epoch_duration_s': n_tempos / sample_rate,
        'total_duration_s': total_samples / sample_rate,
        'transition_epoch_index': metade_epocas,
        'transition_sample_index_0based': transicao_sample_index,
        'transition_sample_index_1based': trans_sample_1based,
        'transition_time_s': trans_time
    }

# ============================================================================== #
# EXECUÇÃO
# ============================================================================== #
inicio_total = time.time()

arquivos = carregar_edfs(DIRETORIO_ARQUIVOS, PADROES_ARQUIVOS)

# coletar todas as épocas em listas para depois concatenar
epocas_geral = {'T1': [], 'T2': []}
info = None
for arq in arquivos:
    epocas_dict, info = extrair_epocas(arq, LISTA_EPOCAS)
    for k in LISTA_EPOCAS:
        epocas_geral[k].extend(epocas_dict.get(k, []))

# Montar uma única lista de epochs e labels
all_epochs = []
all_labels = []
for label in LISTA_EPOCAS:
    arr = np.array(epocas_geral.get(label, []))
    if arr.size:
        for ep in arr:
            all_epochs.append(ep)
            all_labels.append(label)

if len(all_epochs) == 0:
    raise RuntimeError("Nenhuma época encontrada com os padrões e canais fornecidos.")

all_epochs = np.stack(all_epochs, axis=0)  # shape (N, channels, times)
all_labels = np.array(all_labels)

# Embaralhar as épocas mantendo labels sincronizados
if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)
perm = np.random.permutation(len(all_epochs))
shuffled_epochs = all_epochs[perm]
shuffled_labels = all_labels[perm].tolist()

# Salvar CSV embaralhado e obter info de transição
salvar_info = salvar_csv_completo_embaralhado(shuffled_epochs, shuffled_labels, ARQUIVO_SAIDA, sample_rate=SAMPLE_RATE)

# salvar_info já foi retornado pela função salvar_csv_completo_embaralhado(...)
n_epochs = salvar_info['n_epochs']
n_times = salvar_info['n_times']
sample_rate = SAMPLE_RATE  # ou usar salvar_info se quiser

trans_sample_idx0 = salvar_info.get('transition_sample_index_0based', None)
trans_epoch_idx = salvar_info.get('transition_epoch_index', None)

print("\nRESUMO FINAL:")
print("="*60)
print(f"Total de épocas salvas: {n_epochs}")
print(f"Duração por época (s): {salvar_info['epoch_duration_s']:.3f}")
print(f"Duração total do arquivo (s): {salvar_info['total_duration_s']:.3f}")
print(f"Índice da época onde começa a segunda metade (0-based): {trans_epoch_idx}")

# ---------------------------------------------------------------------------------------------
# CALCULA E IMPRIME O MOMENTO QUE TERMINA A PRIMEIRA FASE
# (último sample da época imediatamente anterior à época com '*')
# ---------------------------------------------------------------------------------------------
if trans_sample_idx0 is None:
    print("Ponto de transição não disponível.")
else:
    # última amostra da primeira fase (0-based)
    end_first_phase_sample0 = trans_sample_idx0 - 1

    if end_first_phase_sample0 < 0:
        print("A transição está na 1ª amostra do arquivo — não há 'época anterior' para marcar.")
    else:
        end_first_phase_time_s = end_first_phase_sample0 / sample_rate
        # formata como HH:MM:SS.mmm usando pandas (pd já importado no script original)
        end_first_phase_time_fmt = pd.to_timedelta(end_first_phase_time_s, unit='s')
        print(f"\nFim da Fase 1 -> último sample da fase 1 (0-based): {end_first_phase_sample0}")
        print(f"Sample (1-based): {end_first_phase_sample0 + 1}")
        print(f"Tempo (s): {end_first_phase_time_s:.6f} s")
        print(f"Formato legível: {end_first_phase_time_fmt}")

# ----- Imprime a ordem das labels no terminal com '*' na época que inicia a 2ª metade -----
trans_idx = trans_epoch_idx
labels_marcadas = []
for i, lbl in enumerate(shuffled_labels):
    if (trans_idx is not None) and (i == trans_idx):
        labels_marcadas.append(f"{lbl}*")  # apenas no terminal
    else:
        labels_marcadas.append(lbl)

print("\nOrdem das labels no arquivo (uma entrada por época, ordem embaralhada):")
print(" | ".join(labels_marcadas))

print("="*60)
print(f"CSV salvo em: {ARQUIVO_SAIDA}")
print(f"Tempo total de execução do script: {time.time() - inicio_total:.2f} segundos")