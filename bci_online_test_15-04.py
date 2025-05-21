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

class Sistema:
    def __init__(self):
        self.t0 = time.time()
        self.dt = 2  # Intervalo entre predições (2 segundos)

sistema = Sistema()

totalNumSamples = 0
numChunks = 0
data = []
np.set_printoptions(precision=4, suppress=True)

print('Pronto!')
print('Input shape:', model.input_shape)
print('Output shape:', model.output_shape)
print(f'Aguarde {round(model.input_shape[1]/160,3)} segundos após começar o streaming.')

buffer_x = []
buffer_y = []
batch_size = 1
# Mapeia a tarefa para rótulo para treinamento (válido apenas para fases 1 e 2)
task2label = {'T1': 0, 'T2': 1}

# Listas para armazenar as predições e as probabilidades por fase
phase_predictions = []
phase_probabilities = []

# Configuração do gráfico (opcional)
plt.ion()
fig, ax = plt.subplots()
plot_interval = 5.0
last_plot_update = time.time()
tgraf = []  # Armazena as predições para o gráfico (globalmente)

#%% Configuração das fases
# Durações de cada fase (em segundos)
fase1_duration = 28   # Fase 1: apenas T1 (treinamento)
fase2_duration = 24.5   # Fase 2: apenas T2 (treinamento)
fase3_duration = 128.99  # Fase 3: simulação real (nenhuma indicação de label)
fase3_end = fase1_duration + fase2_duration + fase3_duration

start_time = time.time()
prev_phase = 1  # Inicia na fase 1

#%% Loop principal
while not keyboard.is_pressed('Esc'):
    current_time = time.time() - start_time

    # Define a fase atual baseado no tempo decorrido
    if current_time < fase1_duration:
        current_phase = 1
        task_auto = 'T1'
    elif current_time < (fase1_duration + fase2_duration):
        current_phase = 2
        task_auto = 'T2'
    elif current_time < fase3_end:
        current_phase = 3
        # Na fase 3, não há rótulo: simula uso real, portanto task é None.
        task_auto = None
    else:
        # Ao finalizar a fase 3, exibe o resumo da última fase e encerra o experimento.
        print("Experimento finalizado! Fase 3 concluída.")
        if phase_predictions:
            mean_prob = np.mean(phase_probabilities)
            dist = {label: phase_predictions.count(label) for label in set(phase_predictions)}
            print(f"Fase {prev_phase} finalizada - Nº predições: {len(phase_predictions)} | Média Prob: {round(mean_prob, 3)} | Distribuição: {dist}")
        break

    # Detecta mudança de fase e exibe resumo da fase anterior
    if current_phase != prev_phase:
        if phase_predictions:
            mean_prob = np.mean(phase_probabilities)
            dist = {label: phase_predictions.count(label) for label in set(phase_predictions)}
            print(f"Fase {prev_phase} finalizada - Nº predições: {len(phase_predictions)} | Média Prob: {round(mean_prob, 3)} | Distribuição: {dist}")
        phase_predictions.clear()
        phase_probabilities.clear()
        prev_phase = current_phase

    # Para fases 1 e 2, possibilita override manual, mas na fase 3 o sistema simula o cenário real (sem rótulo)
    if current_phase in [1, 2]:
        if keyboard.is_pressed('1'):
            task = 'T1'
        elif keyboard.is_pressed('2'):
            task = 'T2'
        elif keyboard.is_pressed('0'):
            task = ''
        else:
            task = task_auto
    else:
        task = None  # Na fase 3, o modelo não recebe indicação de label

    # Coleta o chunk do LSL
    chunk, timestamp = inlet.pull_chunk()
    if chunk:
        numChunks += 1
        totalNumSamples += len(chunk)
        for ind, sample in enumerate(chunk):
            data.append(sample)

            # Quando há amostras suficientes para formar um "epoch"
            if len(data) >= epochsize:
                if time.time() - sistema.t0 > sistema.dt:
                    sistema.t0 = time.time()
                    if len(data) == epochsize:
                        pred = predict(model, data).numpy()[0][0]
                        # Interpreta a predição baseada no limiar
                        if pred < limiar:
                            let = 'T1'
                        elif pred > 1 - limiar:
                            let = 'T2'
                        else:
                            let = 'T0'
                        tgraf.append(let)
                        # Armazena para o resumo da fase
                        phase_predictions.append(let)
                        phase_probabilities.append(pred)
                        # Exibe a saída, sempre mostrando a probabilidade
                        print('===> Output:', let, '| Probabilidade:', round(pred, 3), end='')
                        if task is not None:
                            print("| Tarefa:", task)
                        else:
                            print()  # Apenas nova linha se não houver indicação de tarefa

                        # Realiza fine-tuning somente nas fases 1 e 2 (quando há indicação de label)
                        if task in task2label and let == task:
                            norm_sample = normalize_sample(data)
                            buffer_x.append(norm_sample[0])
                            buffer_y.append(task2label[task])
                            if len(buffer_x) >= batch_size:
                                x = np.array(buffer_x)
                                y = np.array(buffer_y).reshape(-1, 1)
                                print("Treinamento (fine-tuning) online realizado!")
                                model.fit(
                                    x, y,
                                    epochs=1,
                                    verbose=1
                                )
                                buffer_x.clear()
                                buffer_y.clear()
                data.pop(0)

    # Atualiza o gráfico a cada 'plot_interval' (opcional)
    if time.time() - last_plot_update > plot_interval:
        last_plot_update = time.time()
        ax.cla()
        ax.plot(tgraf, marker='o', linestyle='-', color='red')
        ax.set_title('Predições ao Vivo')
        ax.set_xlabel('Amostras')
        ax.set_ylabel('Classe (T1, T2, T0)')
        plt.draw()
        plt.pause(0.001)

#%% Fim da aquisição e resumo geral
print("Número total de amostras:", len(data) + 1)
print("Número total de chunks e samples: {} , {}".format(numChunks, totalNumSamples))
if len(tgraf) == 0:
    exit('Não rodou o EEG')

print('Distribuição GLOBAL das classes preditas:')
print('T1:', round(tgraf.count('T1') / len(tgraf), 3))
print('T2:', round(tgraf.count('T2') / len(tgraf), 3))
print('T0:', round(tgraf.count('T0') / len(tgraf), 3))

plt.ioff()
fig2, ax2 = plt.subplots()
ax2.plot(tgraf, marker='o', linestyle='-', color='blue')
ax2.set_title('Resultado Final das Predições')
ax2.set_xlabel('Índice')
ax2.set_ylabel('Classe (T1, T2, T0)')
plt.show()
