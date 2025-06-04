import mne
import numpy as np
import pandas as pd
import os

# ============================================================================== #
# FUNÇÕES DE PROCESSAMENTO
# ============================================================================== #

def processar_edf(edf_path, tarefas, decimal):
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    try:
        raw.pick_channels(CANAIS_DESEJADOS, ordered=True)
    except ValueError as e:
        print(f"Erro nos canais: {e}")
        raise

    if tarefas:
        eventos, event_dict = mne.events_from_annotations(raw)
        dados, duracoes = [], {}

        for tarefa in tarefas:
            epochs = mne.Epochs(raw, eventos, event_id=event_dict[tarefa],
                                tmin=0.5, tmax=4, baseline=None, preload=True)
            dados_tarefa = np.concatenate(epochs.get_data(), axis=-1) * 1e6
            dados_tarefa = (dados_tarefa * 10**decimal).astype(int) / 10**decimal
            dados.append(dados_tarefa)
            duracoes[tarefa] = len(epochs) * 3.5

        return np.hstack(dados).T, duracoes, raw.info
    else:
        dados = raw.get_data() * 1e6
        dados = (dados * 10**decimal).astype(int) / 10**decimal
        return dados.T, raw.times[-1], raw.info

def criar_csv(dados_fase1, dados_fase2, info, duracoes, caminho):
    dados_completos = np.vstack([dados_fase1, dados_fase2])
    df = pd.DataFrame(dados_completos, columns=[f'EXG Channel {i}' for i in range(16)])

    estrutura = [
        'Sample Index',
        *[f'EXG Channel {i}' for i in range(16)],
        'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2',
        *['Other']*7,
        'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2',
        'Timestamp', 'Other', 'Timestamp (Formatted)'
    ]

    df.insert(0, 'Sample Index', range(1, len(df)+1))
    for col in estrutura[17:]:
        df[col] = 0
    df = df[estrutura]

    with open(caminho, 'w', newline='') as f:
        f.write(
            "%OpenBCI Raw EXG Data\n"
            f"%Number of channels = {len(CANAIS_DESEJADOS)}\n"
            f"%Sample Rate = {int(info['sfreq'])} Hz\n"
            "%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n"
        )
        df.to_csv(f, index=False, header=True, lineterminator='\n')

    return df

def insert_zeros_and_header(path, n_zeros, header):
    with open(path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("Arquivo deve conter ao menos um cabeçalho (%) e o cabeçalho de colunas.")

    header_count = sum(1 for line in lines if line.startswith('%'))

    header_section = lines[:header_count]
    column_header  = lines[header_count]
    data_lines     = lines[header_count + 1:]

    num_cols  = len(column_header.strip().split(','))
    zero_line = ','.join(['0'] * num_cols) + '\n'
    zero_block = [zero_line] * n_zeros

    new_lines = header + [column_header] + zero_block + data_lines

    with open(path, 'w') as f:
        f.writelines(new_lines)

    print('Arquivo modificado com sucesso na ordem correta!')

def gerar_sequencia_fase3(qtd_repeticoes=10):
    # Sequência alternada: T0, T1, T0, T2, T0, T1, ...
    sequencia = []
    for i in range(qtd_repeticoes):
        sequencia.extend(['T0', 'T1', 'T0', 'T2'])
    return sequencia

# ============================================================================== #
# CONFIGURAÇÃO
# ============================================================================== #

CANAIS_DESEJADOS = [
    'C3..', 'C4..', 'Fp1.', 'Fp2.', 'F7..', 'F3..',
    'F4..', 'F8..', 'T7..', 'T8..', 'P7..', 'P3..',
    'P4..', 'P8..', 'O1..', 'O2..'
]

ARQUIVO_FASE1 = r"C:\Users\batis\OneDrive\Área de Trabalho\S003R04.edf"
ARQUIVO_FASE2 = r"C:\Users\batis\OneDrive\Área de Trabalho\S003R08.edf"

CASAS_DECIMAIS = 4
CAMINHO_SAIDA = r"C:\Users\batis\OneDrive\Área de Trabalho\teste.csv"

n_zero_lines = 1000
header_lines = [
    '%OpenBCI Raw EXG Data\n',
    '%Number of channels = 16\n',
    '%Sample Rate = 160 Hz\n',
    '%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n'
]

# ============================================================================== #
# EXECUÇÃO
# ============================================================================== #

dados_fase1, duracoes, info = processar_edf(ARQUIVO_FASE1, ['T1', 'T2'], CASAS_DECIMAIS)
dados_fase2, duracao_fase2, _ = processar_edf(ARQUIVO_FASE2, [], CASAS_DECIMAIS)

criar_csv(dados_fase1, dados_fase2, info, duracoes, CAMINHO_SAIDA)
insert_zeros_and_header(CAMINHO_SAIDA, n_zero_lines, header_lines)

# RELATÓRIO FINAL SIMPLES NO TERMINAL
print("\nPROCESSAMENTO CONCLUÍDO!")
print("="*40)
print(f"Arquivo gerado: {CAMINHO_SAIDA}")
print("\nDurações das Fases:")
for fase, (tarefa, duracao) in enumerate(duracoes.items(), 1):
    print(f"Fase {fase} ({tarefa}): {duracao:.2f} segundos")
print(f"Fase {len(duracoes)+1} (Dados Completos): {duracao_fase2:.2f} segundos")

# Sequência de tarefas esperada para Fase 3
print("\nSequência esperada de tarefas na FASE 3:")
sequencia_fase3 = gerar_sequencia_fase3(qtd_repeticoes=10)
print(" -> ".join(sequencia_fase3))
print("="*40)
