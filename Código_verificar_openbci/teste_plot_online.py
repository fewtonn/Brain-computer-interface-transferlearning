#%% Imports e configurações iniciais
import keyboard   # obs: testar mudar o batch_size para 1 e epochs para 1, conforme solicitado
import numpy as np
import time
from random import choice
from pylsl import StreamInlet, resolve_stream, local_clock
from time import sleep
from sys import exit
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Importa o modelo e as funções do Keras
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as kutils

# Carrega o modelo (com saída Dense(1, activation='sigmoid'))
model = load_model(r"C:\Users\batis\Downloads\melhor_modelo_0.8871.h5")
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

limiar = 0.4
# 'task' será usado nas fases 1 e 2 (treinamento), na fase 3 será None para simular cenário real
task = None

def normalize_sample(input_sample):
    inp = np.array(input_sample)
    local_max = inp.max()
    local_min = inp.min()
    normalized = (inp - local_min) / (local_max - local_min + 1e-8)
    return np.array([normalized])

def predict(model, input_sample):
    m = normalize_sample(input_sample)
    return model(m, training=False)

epochsize = model.input_shape[1]

print("Procurando por uma stream EEG...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
sleep(1)
print("Stream encontrada!")

#%% Plot em tempo real (µV) para eletrodos 1-16
channel_names = [f"Eletrodo {i}" for i in range(1, 17)]
SCALE_FACTOR = 1e6    # Converte de Volts para microvolts
DATA_WINDOW = 5.0     # segundos de histórico na janela móvel (configurado para 5 segundos)
REFRESH_INTERVAL = 0.5  # segundos entre atualizações

# Inferir número de canais e limitar a 16
chunk, _ = inlet.pull_chunk(timeout=1.0)
if not chunk:
    raise RuntimeError("Não recebeu dados do stream EEG")
n_channels = min(len(chunk[0]), 16)

# Buffers para timestamps e sinais
times = []
buffers = [[] for _ in range(n_channels)]

# Configuração da figura e eixos
plt.ion()
fig, axes = plt.subplots(n_channels, 1, sharex=True, figsize=(12, n_channels * 1.0))
for ax, name in zip(axes, channel_names):
    ax.set_ylabel(name, rotation=0, labelpad=40, va='center')
    ax.set_yticks([])
# Label do eixo X no último subplot
axes[-1].set_xlabel('Tempo (s)')

last_refresh = time.time()

# Loop de coleta e plotagem
while not keyboard.is_pressed('Esc'):
    chunk, timestamps = inlet.pull_chunk(timeout=0.0)
    if chunk and timestamps:
        for sample, ts in zip(chunk, timestamps):
            times.append(ts)
            for i in range(n_channels):
                buffers[i].append(sample[i] * SCALE_FACTOR)
            # Remover dados fora da janela móvel
            while times and (times[-1] - times[0] > DATA_WINDOW):
                times.pop(0)
                for buf in buffers:
                    buf.pop(0)

    # Atualizar gráficos periodicamente
    if time.time() - last_refresh >= REFRESH_INTERVAL:
        last_refresh = time.time()
        if times:
            t0 = times[0]
            rel_times = [t - t0 for t in times]
            for idx, ax in enumerate(axes):
                ax.clear()
                ax.plot(rel_times, buffers[idx], linewidth=0.8)
                ax.set_ylabel(channel_names[idx], rotation=0, labelpad=40, va='center')
                ax.set_yticks([])
            axes[-1].set_xlabel('Tempo (s)')
        plt.pause(0.001)

plt.ioff()
print("Plot encerrado.")
