import mne
import numpy as np
import pandas as pd
import os

# ==============================================================================
# FUNÇÕES DE PROCESSAMENTO (NÃO EDITAR)
# ==============================================================================

def processar_edf(edf_path, tarefas, decimal):
    """Processa arquivo EDF mantendo apenas os canais desejados"""
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    
    # Selecionar canais obrigatórios
    try:
        raw.pick_channels(CANAIS_DESEJADOS, ordered=True)
    except ValueError as e:
        print(f"Erro nos canais: {e}")
        raise

    # Processar épocas se houver tarefas
    if tarefas:
        eventos, event_dict = mne.events_from_annotations(raw)
        dados, duracoes = [], {}
        
        for tarefa in tarefas:
            epochs = mne.Epochs(raw, eventos, event_id=event_dict[tarefa],
                              tmin=0.5, tmax=4, baseline=None, preload=True)
            
            # Converter e truncar
            dados_tarefa = np.concatenate(epochs.get_data(), axis=-1) * 1e6
            dados_tarefa = (dados_tarefa * 10**decimal).astype(int) / 10**decimal
            dados.append(dados_tarefa)
            duracoes[tarefa] = len(epochs) * 3.5
            
        return np.hstack(dados).T, duracoes, raw.info
    else:
        # Processar arquivo inteiro
        dados = raw.get_data() * 1e6
        dados = (dados * 10**decimal).astype(int) / 10**decimal
        return dados.T, raw.times[-1], raw.info

def criar_csv(dados_fase1, dados_fase2, info, duracoes, caminho):
    """Cria arquivo CSV com cabeçalho exato do OpenBCI"""
    dados_completos = np.vstack([dados_fase1, dados_fase2])
    
    # Criar DataFrame
    df = pd.DataFrame(dados_completos, columns=[f'EXG Channel {i}' for i in range(16)])
    
    # Adicionar estrutura OpenBCI completa
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
    
    # Escrever arquivo com cabeçalho correto
    with open(caminho, 'w', newline='') as f:
        # Cabeçalho obrigatório
        f.write(
            "%OpenBCI Raw EXG Data\n"
            f"%Number of channels = {len(CANAIS_DESEJADOS)}\n"
            f"%Sample Rate = {int(info['sfreq'])} Hz\n"
            "%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n"
        )
        
        # Metadados adicionais das fases
        #for idx, (tarefa, duracao) in enumerate(duracoes.items(), 1):
       #     f.write(f"%{tarefa} Duration = {duracao:.2f} seconds\n")
       # f.write(f"%Full File Duration = {dados_fase2.shape[0]/info['sfreq']:.2f} seconds\n\n")
        
        # Dados sem espaços
        df.to_csv(f, index=False, header=True, lineterminator='\n')

    return df

# ==============================================================================
# CONFIGURAÇÃO DE VARIAVEIS OQ EU FIZ AQ NA PESSOA S01 COLOQUEI O R04 E R08, R04 E R08 SÃO DADOS Q USARAM O MSM PROTOCOLO PARA ADQUIRIR Fase 1 (T1): 28.00 segundos ESSAS FORAM AS INFOS Q CONSEGUI
#Fase 2 (T2): 24.50 segundos
#Fase 3 (Dados Completos): 124.99 segundos
# ==============================================================================
CANAIS_DESEJADOS = [
    'C3..', 'C4..', 'Fp1.', 'Fp2.', 'F7..', 'F3..', 
    'F4..', 'F8..', 'T7..', 'T8..', 'P7..', 'P3..', 
    'P4..', 'P8..', 'O1..', 'O2..'
]

# Primeiro arquivo EDF (T1 e T2)
ARQUIVO_FASE1 = r"C:\Users\batis\OneDrive\Área de Trabalho\Códigos diversos\Faculdade\BCI\Dados BRUTOS\files\S001\S001R04.edf" # <--- Alterar
TAREFAS_FASE1 = ['T1', 'T2']  # Tarefas para extrair do primeiro arquivo

# Segundo arquivo EDF (dados completos)
ARQUIVO_FASE2 = r"C:\Users\batis\OneDrive\Área de Trabalho\Códigos diversos\Faculdade\BCI\Dados BRUTOS\files\S001\S001R08.edf"  # <--- Alterar 

# Configurações de saída
CASAS_DECIMAIS = 4  # Número de casas decimais a manter
CAMINHO_SAIDA = r"C:\Users\batis\OneDrive\Área de Trabalho\Códigos diversos\dados_cobinados.csv" # <--- Alterar

# ==============================================================================
# EXECUÇÃO PRINCIPAL (NÃO EDITAR)
# ==============================================================================

# Processar Fase 1 (T1 e T2)
dados_fase1, duracoes, info = processar_edf(
    edf_path = ARQUIVO_FASE1,
    tarefas = ['T1', 'T2'],
    decimal = CASAS_DECIMAIS
)

# Processar Fase 2 (Arquivo completo)
dados_fase2, duracao_fase2, _ = processar_edf(
    edf_path = ARQUIVO_FASE2,
    tarefas = [],
    decimal = CASAS_DECIMAIS
)

# Gerar CSV final
criar_csv(
    dados_fase1 = dados_fase1,
    dados_fase2 = dados_fase2,
    info = info,
    duracoes = duracoes,
    caminho = CAMINHO_SAIDA
)
# Relatório final
print("\nPROCESSAMENTO CONCLUÍDO!")
print("="*40)
print(f"Arquivo gerado: {CAMINHO_SAIDA}")
print("\nDurações das Fases:")
for fase, (tarefa, duracao) in enumerate(duracoes.items(), 1):
    print(f"Fase {fase} ({tarefa}): {duracao:.2f} segundos")
print(f"Fase {len(duracoes)+1} (Dados Completos): {duracao_fase2:.2f} segundos")
print("="*40)