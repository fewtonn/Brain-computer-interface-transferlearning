import keyboard
import numpy as np
from random import choice
import time
from pylsl import StreamInlet, resolve_stream, local_clock
from time import sleep
from sys import exit
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# PARA O ARQUIVO DE FASES A FASE 1 É T1 28 SEGUNDOS, T2 24,5 SEGUNDOS E OS DADOS COMPLETOS 124,99 S 
# Importa o modelo e as funções do Keras
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as kutils

# Carrega o modelo (com entrada (721, 16) e saída Dense(1, activation='sigmoid'))
model = load_model(r"C:\Users\batis\Downloads\melhor_modelo_0.8871.h5")
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Fases e tempos
tempo_fase1 = 28.0
tempo_fase2 = 24.5
tempo_fase3 = 124.99

contador = 0
limiar = 0.4
task = ''
current_data = []
MIN_GLOBAL, MAX_GLOBAL = -1, 1
norm_dinamica = True

fase = 1
label_fase1 = None
inicio_fase = None

# Normalização dinâmica

def normalize_sample(input_sample):
    inp = np.array(input_sample)
    if norm_dinamica:
        local_max = inp.max()
        local_min = inp.min()
    else:
        local_max, local_min = MAX_GLOBAL, MIN_GLOBAL
    normalized = (inp - local_min) / (local_max - local_min + 1e-8)
    return normalized

def predict(model, input_sample):
    m = normalize_sample(input_sample)
    m = np.expand_dims(m, axis=0)
    return model(m, training=False)

epochsize = model.input_shape[1]  # 721
input_channels = model.input_shape[2]  # 16

print("Procurando por uma stream EEG...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
sleep(1)
print("Stream encontrada!")

class Sistema:
    def __init__(self):
        self.dt = 4.506  # tempo entre predições

sistema = Sistema()

buffer_x = []
buffer_y = []
batch_size = 1
task2label = {'T1': 0, 'T2': 1}
tgraf = []
started = False

plt.ion()
fig, ax = plt.subplots()
plot_interval = sistema.dt
last_plot_update = time.time()

print('Pronto!')
print('Input shape:', model.input_shape)
print('Output shape:', model.output_shape)
print(f'Aguarde {round(epochsize/160,3)} segundos após começar o streaming.')

# Loop principal
while not keyboard.is_pressed('Esc'):
    if keyboard.is_pressed('1'):
        task = 'T1'
        print("Tarefa atual: Esquerda (T1)")
    elif keyboard.is_pressed('2'):
        task = 'T2'
        print("Tarefa atual: Direita (T2)")
    elif keyboard.is_pressed('0'):
        task = ''
        print("Tarefa atual: Parado")

    chunk, timestamp = inlet.pull_chunk()
    if chunk:
        for ind, sample in enumerate(chunk):
            if not started and np.any(np.array(sample) != 0):
                started = True
                inicio_fase = time.time()
                print("Dados não zerados detectados. Iniciando processamento.")

            if not started:
                continue

            tempo_fase_atual = time.time() - inicio_fase
            if fase == 1 and tempo_fase_atual > tempo_fase1:
                fase = 2
                inicio_fase = time.time()
                print("\n>> Iniciando Fase 2")
            elif fase == 2 and tempo_fase_atual > tempo_fase2:
                fase = 3
                inicio_fase = time.time()
                print("\n>> Iniciando Fase 3")
            elif fase == 3 and tempo_fase_atual > tempo_fase3:
                print("\n>> Encerrando fases.")
                break

            current_data.append(sample)
            if len(current_data) < epochsize:
                continue

            pred = predict(model, current_data).numpy()[0][0]
            if pred < limiar:
                let = 'T1'
            elif pred > 1 - limiar:
                let = 'T2'
            else:
                let = 'T0'

            # Definir label da fase 1
            if fase == 1 and label_fase1 is None and let in ['T1', 'T2']:
                label_fase1 = let
                print(f"Label da Fase 1 definida: {label_fase1}")

            # Fase 1: só aceita o label inicial
            if fase == 1 and let != label_fase1:
                current_data = []
                continue

            # Fase 2: só aceita label oposto ao da fase 1
            if fase == 2:
                label_oposto = 'T2' if label_fase1 == 'T1' else 'T1'
                if let != label_oposto:
                    current_data = []
                    continue

            # Fase 3: aceita qualquer coisa, sem fine-tuning
            do_finetune = False
            if fase == 2 and task == let:
                do_finetune = True

            tgraf.append(let)
            print(f'===> Fase {fase} | Output: {let} | Prob: {pred:.5f} | Tarefa: {task} | Tempo: {time.time()}')

            if do_finetune and task in task2label:
                norm_sample = normalize_sample(current_data)
                buffer_x.append(norm_sample)
                buffer_y.append(task2label[task])
                if len(buffer_x) >= batch_size:
                    x = np.array(buffer_x)
                    y = np.array(buffer_y).reshape(-1, 1)
                    print("Treinando com fine-tuning online...")
                    model.fit(x, y, epochs=1, verbose=1)
                    buffer_x.clear()
                    buffer_y.clear()

            current_data = []

            if time.time() - last_plot_update > plot_interval:
                last_plot_update = time.time()
                ax.cla()
                ax.plot(tgraf, marker='o', linestyle='-', color='red')
                ax.set_title('Predições ao Vivo')
                ax.set_xlabel('Amostras')
                ax.set_ylabel('Classe (T1, T2, T0)')
                plt.draw()
                plt.pause(0.001)

print("Fim da aquisição")
plt.ioff()
fig2, ax2 = plt.subplots()
ax2.plot(tgraf, marker='o', linestyle='-', color='blue')
ax2.set_title('Resultado Final das Predições')
ax2.set_xlabel('Índice')
ax2.set_ylabel('Classe (T1, T2, T0)')
plt.show()
