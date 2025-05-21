import keyboard   # obs: testar mudar batch size p 1 e epochs 1 se precisar
import numpy as np
from random import choice
import time
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
model = load_model(r"C:\Users\LaBios - BCI\Downloads\melhor_modelo_0.8871 (1).h5")
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
contador=0
limiar = 0.4
task = ''
current_data=[]
# Normalização fixa para compatibilidade com o código original2
MIN_GLOBAL, MAX_GLOBAL = -1, 1
norm_dinamica = True

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
    m = np.expand_dims(m, axis=0)  # adiciona dimensão do batch: (1, 721, 16)
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
        self.dt = 4.506 # Saída a cada 2 segundos

sistema = Sistema()

totalNumSamples = 0
numChunks = 0

data = []
totdata = []
np.set_printoptions(precision=4, suppress=True)

buffer_x = []
buffer_y = []
batch_size = 1
task2label = {'T1': 0, 'T2': 1}

tgraf = []

# Flag para iniciar somente após encontrar amostra não-zero
started = False

plt.ion()
fig, ax = plt.subplots()
plot_interval = 5.0
last_plot_update = time.time()

print('Pronto!')
print('Input shape:', model.input_shape)
print('Output shape:', model.output_shape)
print(f'Aguarde {round(epochsize/160,3)} segundos após começar o streaming.')

#%% Loop principal
while not keyboard.is_pressed('Esc'):
    # Definir tarefa
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
        numChunks += 1
        totalNumSamples += len(chunk)
        # Sempre acumula dados, incluindo zeros iniciais
        for ind, sample in enumerate(chunk):
           # data.append(sample)
            # Sinal de início: quando detecta primeiro valor não-zero
            if not started and np.any(np.array(sample) != 0):
                started = True
                sistema.t0 = time.time()
                print("Dados não zerados detectados. Iniciando processamento.")
            # Só roda predição após started e com buffer cheio
            if started:
                #data.append
                current_data.append(sample)  # garante que input sempre seja do tamanho certo
                if len(current_data) >= epochsize:
                    sistema.t0 = time.time()
                    pred = predict(model, current_data).numpy()[0][0]
                    if pred < limiar:
                        let = 'T1'
                    elif pred > 1 - limiar:
                        let = 'T2'
                    else:
                        let = 'T0'
                    tgraf.append(let)
                    contador+=1
                    if contador == 3:
                        pass
                    current_data=[]
                    print(f'===> Output: {let} | Probabilidade: {pred:.5f} | Tarefa: {task}| Tempo: {time.time()}')
                    # Fine-tuning
                    if task in task2label:
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
            tempo_a = timestamp[ind]

    # Atualiza gráfico
    if time.time() - last_plot_update > plot_interval:
        last_plot_update = time.time()
        ax.cla()
        ax.plot(tgraf, marker='o', linestyle='-', color='red')
        ax.set_title('Predições ao Vivo')
        ax.set_xlabel('Amostras')
        ax.set_ylabel('Classe (E, D, P)')
        plt.draw()
        plt.pause(0.001)

#%% Fim da aquisição
print("Número total de amostras:", totalNumSamples)
print("Number of Chunks and Samples == {} , {}".format(numChunks, totalNumSamples))
if len(tgraf) == 0:
    exit('Não rodou o EEG')

print('Distribuição:')
print('E:', round(tgraf.count('E') / len(tgraf), 3))
print('D:', round(tgraf.count('D') / len(tgraf), 3))
print('P:', round(tgraf.count('P') / len(tgraf), 3))

plt.ioff()
fig2, ax2 = plt.subplots()
ax2.plot(tgraf, marker='o', linestyle='-', color='blue')
ax2.set_title('Resultado Final das Predições')
ax2.set_xlabel('Índice')
ax2.set_ylabel('Classe (E, D, P)')
plt.show()