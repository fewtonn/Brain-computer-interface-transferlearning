import os
import mne
import numpy as np
import pandas as pd
import time
import fnmatch
from io import StringIO
 # ESSE CÓDIGO TEM A FUNÇÃO DE PEGAR TAREFAS DO DATASET BRUTO E TRANSFORMAR EM UMA MANEIRA Q O SOFTWARE OPENBCI CONSIGA LER. 
 # ELE DIVIDE O CÓDIGO EM 3 FASES FASE 1 E 2 TREINOS PARA T1 E T2 E A FASE 3 É Q JUNTA TUDO PARA TESTAR COM TRASNFERLEARNING
# ============================================================================== #
# CONFIGURAÇÃO
# ============================================================================== #
DIRETORIO_ARQUIVOS = r"C:\Users\batis\OneDrive\Área de Trabalho\Códigos diversos\Faculdade\BCI\Dados BRUTOS\files\S086"
ARQUIVO_SAIDA = r"C:\Users\batis\OneDrive\Área de Trabalho\Brain Computer Interface\teste.csv"
PADROES_ARQUIVOS = ["*R04.edf", "*R08.edf", "*R12.edf"]
LISTA_EPOCAS = ['T1', 'T2', 'T0']
CANAIS_DESEJADOS = ['Fp1.', 'F7..', 'F3..', 'T7..', 'C3..', 'P7..', 'P3..', 'O1..',
                    'Fp2.', 'F4..', 'F8..', 'C4..', 'T8..', 'P4..', 'P8..', 'O2..']
CASAS_DECIMAIS = 4

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
    raw = mne.io.read_raw_edf(arquivo, preload=True)
    raw.pick_channels(CANAIS_DESEJADOS)
    eventos, ids = mne.events_from_annotations(raw)
    epocas_dict = {}
    for evento in eventos_desejados:
        if evento in ids:
            epochs = mne.Epochs(raw, eventos, event_id={evento: ids[evento]},
                                 tmin=-0.5, tmax=4, baseline=(-0.5, 0), preload=True)
            epocas = epochs.get_data() * 1e6
            epocas = np.round(epocas, decimals=CASAS_DECIMAIS)
            epocas_dict[evento] = epocas
    return epocas_dict, raw.info

def dividir_fases(epocas_dict, prop=0.8, max_t0=0.5):
    t1 = np.array(epocas_dict.get('T1', []))
    t2 = np.array(epocas_dict.get('T2', []))
    t0 = np.array(epocas_dict.get('T0', []))

    n1, n2 = len(t1), len(t2)
    n1_treino, n2_treino = int(n1 * prop), int(n2 * prop)

    fase1 = t1[:n1_treino]
    fase2 = t2[:n2_treino]
    t1_teste = t1[n1_treino:]
    t2_teste = t2[n2_treino:]

    n0_max = int((len(t1_teste) + len(t2_teste)) * max_t0)
    t0 = t0[:n0_max]

    fase3 = []
    ordem_f3 = []
    for i in range(max(len(t1_teste), len(t2_teste))):
        if i < len(t1_teste):
            fase3.append(t1_teste[i])
            ordem_f3.append('T1')
        if i < len(t0):
            fase3.append(t0[i])
            ordem_f3.append('T0')
        if i < len(t2_teste):
            fase3.append(t2_teste[i])
            ordem_f3.append('T2')

    return fase1, fase2, fase3, ordem_f3

def salvar_csv_completo(fase1, fase2, fase3, caminho):
    dados = np.concatenate([fase1, fase2, fase3], axis=0)
    n_epocas, n_canais, n_tempos = dados.shape

    dados_ajustados = dados.transpose(0, 2, 1).reshape(n_epocas * n_tempos, n_canais)
    df = pd.DataFrame(dados_ajustados, columns=[f'EXG Channel {i}' for i in range(n_canais)])

    df.insert(0, 'Sample Index', range(1, len(df) + 1))
    df['Accel Channel 0'] = 0
    df['Accel Channel 1'] = 0
    df['Accel Channel 2'] = 0
    for i in range(7):
        df[f'Other_{i}'] = 0
    df['Analog Channel 0'] = 0
    df['Analog Channel 1'] = 0
    df['Analog Channel 2'] = 0
    df['Timestamp'] = 0
    df['Other'] = 0
    df['Timestamp (Formatted)'] = 0

    header = [
        '%OpenBCI Raw EXG Data\n',
        '%Number of channels = 16\n',
        '%Sample Rate = 160 Hz\n',
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

# ============================================================================== #
# EXECUÇÃO
# ============================================================================== #
inicio_total = time.time()

arquivos = carregar_edfs(DIRETORIO_ARQUIVOS, PADROES_ARQUIVOS)

epocas_geral = {'T1': [], 'T2': [], 'T0': []}
info = None
for arq in arquivos:
    epocas_dict, info = extrair_epocas(arq, LISTA_EPOCAS)
    for k in LISTA_EPOCAS:
        epocas_geral[k].extend(epocas_dict.get(k, []))

fase1, fase2, fase3, ordem_f3 = dividir_fases(epocas_geral)

dur_f1 = len(fase1) * 4.5
dur_f2 = len(fase2) * 4.5
dur_f3 = len(fase3) * 4.5

salvar_csv_completo(fase1, fase2, fase3, ARQUIVO_SAIDA)

print("\nRESUMO FINAL:")
print("="*40)
print(f"Fase 1 (T1): {len(fase1)} épocas | Duração estimada: {dur_f1:.2f} s")
print(f"Fase 2 (T2): {len(fase2)} épocas | Duração estimada: {dur_f2:.2f} s")
print(f"Fase 3 (T1+T2+T0): {len(fase3)} entradas | Duração estimada: {dur_f3:.2f} s")
print("\nOrdem das tarefas na Fase 3:")
print(" ➔ ".join(ordem_f3))
print("="*40)
print(f"CSV salvo em: {ARQUIVO_SAIDA}")
print(f"Tempo total: {time.time() - inicio_total:.2f} segundos")
