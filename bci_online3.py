import numpy as np
import keyboard  # para detectar ESC

import time
from pylsl import StreamInlet, resolve_stream
from time import sleep
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Carrega o modelo (entrada esperada: batch x 721 x 16)
model = load_model(r"C:\Users\batis\Downloads\melhor_modelo_0.8871.h5")
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---------- CONFIGURAÇÕES ----------
fase1_forced_label = 'T1'  # 'T1' ou 'T2' para Fase 1
tempo_fase1 = 28.0
tempo_fase2 = 24.5
tempo_fase3 = 124.99
limiar = 0.4
batch_size = 1

def normalize_sample(input_sample):
    inp = np.array(input_sample)
    local_max, local_min = inp.max(), inp.min()
    return (inp - local_min) / (local_max - local_min + 1e-8)

# Inicializa stream e variáveis
epochsize = model.input_shape[1]  # 721
print(f"Aguardando stream EEG... input shape: {model.input_shape}")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
sleep(1)
print("Stream EEG encontrada!")

class Sistema:
    def __init__(self):
        self.dt = 4.506
sistema = Sistema()

tgraf = []
buffer_x, buffer_y = [], []
task2label = {'T1': 0, 'T2': 1}

# Controle de fases
tempo_inicio = None
fase_atual = 1
label_fase1 = fase1_forced_label
if label_fase1 not in ['T1','T2']:
    raise ValueError("fase1_forced_label deve ser 'T1' ou 'T2'.")
print(f"Fase 1 configurada para aceitar sempre: {label_fase1}")

# Aquisição de dados
current_data = []
started = False

plt.ion()
fig, ax = plt.subplots()
last_plot = time.time()

print('Pressione ESC para encerrar.')
while not keyboard.is_pressed('Esc'):
    if keyboard.is_pressed('Esc'):
        print('ESC detectado. Encerrando imediatamente.')
        break

    chunk, _ = inlet.pull_chunk()
    if not chunk:
        continue

    for sample in chunk:
        # Detecta início válido (amostra não-zero)
        if not started and np.any(np.array(sample) != 0):
            started = True
            tempo_inicio = time.time()
            print("Início da Fase 1: dados válidos detectados.")
        if not started:
            continue

        # Controle de transição entre fases
        elapsed = time.time() - tempo_inicio
        if fase_atual == 1 and elapsed > tempo_fase1:
            fase_atual = 2
            tempo_inicio = time.time()
            print("-> Fase 2 iniciada")
        elif fase_atual == 2 and elapsed > tempo_fase2:
            fase_atual = 3
            tempo_inicio = time.time()
            print("-> Fase 3 iniciada")
        elif fase_atual == 3 and elapsed > tempo_fase3:
            print("-> Todas as fases concluídas. Encerrando loop.")
            break
        if keyboard.is_pressed('Esc'):
            print('ESC detectado dentro do loop de amostras. Encerrando imediatamente.')
            break

        # Acumula cada sample até formar exatamente 721 amostras
        current_data.append(sample)

        if len(current_data) == epochsize:
            # Aqui temos uma janela completa de 721×16
            arr_norm = normalize_sample(current_data)
            pred = model(
                np.expand_dims(arr_norm, axis=0),
                training=False
            ).numpy()[0][0]

            if pred < limiar:
                label = 'T1'
            elif pred > 1 - limiar:
                label = 'T2'
            else:
                label = 'T0'

            # Verifica se faz Transfer Learning
            do_tl = False
            if fase_atual == 1:
                if label != label_fase1:
                    current_data = []  # descarta a janela e reinicia
                    continue
                do_tl = True
            elif fase_atual == 2:
                opp = 'T1' if label_fase1 == 'T2' else 'T2'
                if label != opp:
                    current_data = []
                    continue
                do_tl = True
            # Na Fase 3, do_tl permanece False

            tgraf.append(label)
            print(f"[Fase {fase_atual}] Previsão: {label} | Prob: {pred:.3f} | TL: {'Sim' if do_tl else 'Não'}")

            if do_tl and label in ('T1','T2'):
                buffer_x.append(arr_norm)
                buffer_y.append(task2label[label])
                if len(buffer_x) >= batch_size:
                    print("=== Iniciando transfer learning (1 época) ===")
                    model.fit(
                        np.array(buffer_x),
                        np.array(buffer_y).reshape(-1,1),
                        epochs=1,
                        verbose=1
                    )
                    print("=== Transfer learning concluído ===")
                    buffer_x, buffer_y = [], []

            # Zero o buffer para começar nova janela exata de 721
            current_data = []

        # Atualiza o gráfico periodicamente
        if time.time() - last_plot >= sistema.dt:
            last_plot = time.time()
            ax.cla()
            ax.plot(tgraf[-100:], marker='o', linestyle='-')
            ax.set_title(f"Live: Fase {fase_atual} | Label Fase1: {label_fase1}")
            plt.draw(); plt.pause(0.001)

    else:
        # Continua puxando próximo chunk se não houve break nas fases ou ESC
        continue
    break  # se houve break dentro do for, sai do while

plt.ioff()
plt.show()
