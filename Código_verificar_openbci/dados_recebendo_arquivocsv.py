#%% Imports e configurações iniciais
import keyboard   # Para detectar tecla Esc
import numpy as np
import time
from pylsl import StreamInlet, resolve_stream
import csv

# Conecta ao stream EEG via LSL
def connect_stream():
    streams = resolve_stream('type', 'EEG')
    return StreamInlet(streams[0])

print("Procurando por uma stream EEG...")
inlet = connect_stream()
print("Stream encontrada! Pressione Esc para encerrar a gravação.")

# Configurações de gravação
SCALE_FACTOR = 1e6  # de Volts para microvolts
MAX_CHANNELS = 16   # eletrodos 1-16

def infer_channels(inlet):
    chunk, _ = inlet.pull_chunk(timeout=10.0)
    if not chunk:
        raise RuntimeError("Não recebeu dados do stream EEG")
    return min(len(chunk[0]), MAX_CHANNELS)

n_channels = infer_channels(inlet)
print(f"Gravando dados de {n_channels} canais.")

# Prepara arquivo CSV
csv_path = 'eeg_stream_data.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['timestamp'] + [f'Eletrodo{i+1}' for i in range(n_channels)]
    writer.writerow(header)

    # Loop de gravação
    while not keyboard.is_pressed('Esc'):
        chunk, timestamps = inlet.pull_chunk(timeout=0.0)
        if chunk and timestamps:
            for sample, ts in zip(chunk, timestamps):
                row = [ts] + list((np.array(sample[:n_channels]) * SCALE_FACTOR))
                writer.writerow(row)
        time.sleep(0.01)

print(f"Gravação encerrada. Dados salvos em {csv_path}")
