#FILTRAR OS DADOS 8-30
import numpy as np
import keyboard  # para detectar ESC
import time
from pylsl import StreamInlet, resolve_stream
from time import sleep
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd  # usado apenas para formatação de tempo legível

# ----------------------------- CONFIGURAÇÃO -----------------------------
# Modelo
model = load_model(r"C:\Users\batis\Downloads\melhor_modelo_0.8871.h5")
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---------- Defina AQUI o tempo da FASE 1 (em segundos) ----------
TIME_PHASE1 = 99.13125  # duração da fase 1 (com TL); após isso começa a Fase 2 (que dura o resto)

# ---------- Defina AQUI as labels esperadas para cada época de cada fase ----------
# Cada item corresponde à 1 época (janela) na ordem.
LABELS_PHASE1 = [
 'T1','T1','T1','T1','T2','T1','T1','T1','T1','T1','T1','T1','T1','T2','T2','T2',
 'T1','T1','T1','T2','T1','T2'
]   # rótulos que serão usados para TL na fase 1
LABELS_PHASE2 = [
 'T2','T2','T2','T1','T2','T1','T2','T2','T2','T1','T1','T2','T2','T2','T2','T1',
 'T2','T2','T1','T2','T2','T2','T2'
]  # rótulos que serão usados só para avaliação na fase 2

# ---------- Outros parâmetros ----------
LIMIAR = 0.4               # limiar para decidir T1/T2/T0
# -----------------------------------------------------------------------

# Classe de controle de atualização de gráfico
class Sistema:
    def __init__(self):
        self.dt = 4.506  # intervalo de plotagem (s)
sistema = Sistema()

# Auxiliares
label2num = {'T1': 1, 'T2': 2, 'T0': 0}

def normalize_sample(input_sample):
    inp = np.array(input_sample)  # shape esperada: (epochsize, n_channels)
    local_max, local_min = inp.max(), inp.min()
    return (inp - local_min) / (local_max - local_min + 1e-8)

def predict(model, input_sample):
    arr = normalize_sample(input_sample)
    arr = np.expand_dims(arr, axis=0)  # batch dimension
    return model(arr, training=False)

# Inicializa stream
epochsize = model.input_shape[1]
print(f"Aguardando stream EEG... input shape: {model.input_shape}")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
sleep(1)
print("Stream EEG encontrada!")

# Fases e métricas
tfase = 1
time_start = None
phase1_end_time = None  # será preenchido no momento da troca

# Contadores e índices para rótulos por época
idx_phase1 = 0
idx_phase2 = 0

phase1_total = phase1_correct = 0
phase2_total = phase2_correct = 0

# Para plotagem dinâmica
pred_values = []
pred_colors = []

plt.ion()
fig, ax = plt.subplots()
last_plot = time.time()

print('Pressione ESC para encerrar.')
current_data = []
started = False
while not keyboard.is_pressed('Esc'):
    # Inicia ao detectar dados não-zero
    if not started:
        chunk, _ = inlet.pull_chunk()
        if chunk and np.any(np.array(chunk) != 0):
            started = True
            time_start = time.time()
            print("Início da Fase 1: dados válidos detectados.")
        else:
            continue

    # Atualiza fase com base no tempo decorrido
    elapsed = time.time() - time_start
    if tfase == 1 and elapsed > TIME_PHASE1:
        tfase = 2
        # registra o momento exato de fim da Fase 1
        phase1_end_time = time.time()
        phase1_elapsed = phase1_end_time - time_start  # deveria ser ~TIME_PHASE1
        print("-> Fase 2 iniciada (sem Transfer Learning).")
        print(f"Fim da Fase 1 (última época processada antes da fase 2): {phase1_elapsed:.6f} s")
        print(f"Formato legível: {pd.to_timedelta(phase1_elapsed, unit='s')}")

    # Puxa chunk de amostras
    chunk, _ = inlet.pull_chunk()
    if not chunk:
        continue

    for sample in chunk:
        # janela deslizante
        current_data.append(sample)
        if len(current_data) > epochsize:
            current_data.pop(0)
        if len(current_data) < epochsize:
            continue

        # Predição
        pred = predict(model, current_data).numpy()[0][0]
        if pred < LIMIAR:
            label = 'T1'
        elif pred > 1 - LIMIAR:
            label = 'T2'
        else:
            label = 'T0'

        # Determina expected a partir das listas de rótulos (se houver)
        expected = None
        correct = False

        if tfase == 1:
            # pegar próximo rótulo da Fase 1, se disponível
            if idx_phase1 < len(LABELS_PHASE1):
                expected = LABELS_PHASE1[idx_phase1]
                idx_phase1 += 1
                phase1_total += 1
                correct = (label == expected)
                if correct:
                    phase1_correct += 1

                # Transfer Learning somente na FASE 1 e somente se tiver rótulo e acertou
                if correct:
                    bx = np.array([normalize_sample(current_data)])
                    by = np.array([label2num[label]]).reshape(-1,1)
                    print("=== Iniciando transfer learning (1 época) ===")
                    model.fit(bx, by, epochs=1, verbose=1)
                    print("=== Transfer learning concluído ===")
            else:
                # sem rótulo definido para essa época -> não contabiliza para acurácia
                expected = None

        else:  # tfase == 2 (dura o resto)
            if idx_phase2 < len(LABELS_PHASE2):
                expected = LABELS_PHASE2[idx_phase2]
                idx_phase2 += 1
                phase2_total += 1
                correct = (label == expected)
                if correct:
                    phase2_correct += 1
            else:
                expected = None

        # Salva para plot
        pred_values.append(label2num[label])
        pred_colors.append('g' if correct else 'r')

        # Impressão por época
        print(f"[Fase {tfase}] Prev: {label} | Exp: {expected} | Prob: {pred:.3f} | Acerto: {correct}")

        # limpa janela atual para começar próxima época
        current_data = []

        # Plot dinâmico a cada sistema.dt
        if time.time() - last_plot >= sistema.dt:
            last_plot = time.time()
            ax.cla()
            # plota últimos 100 pontos com cores
            for i, (y, c) in enumerate(zip(pred_values[-100:], pred_colors[-100:])):
                ax.scatter(i, y, c=c, marker='o')
            ax.set_title(f"Live: Fase {tfase}")
            ax.set_ylabel('Label numérico')
            plt.draw(); plt.pause(0.001)

# ----- Relatórios finais -----
print("\n=== Acurácias finais ===")
if phase1_total:
    acc1 = phase1_correct / phase1_total * 100
    print(f"Fase 1: {phase1_correct}/{phase1_total} = {acc1:.2f}% (TL aplicado somente na Fase 1)")
else:
    acc1 = None
    print("Fase 1: sem épocas rotuladas para calcular acurácia.")

if phase2_total:
    acc2 = phase2_correct / phase2_total * 100
    print(f"Fase 2: {phase2_correct}/{phase2_total} = {acc2:.2f}%")
else:
    acc2 = None
    print("Fase 2: sem épocas rotuladas para calcular acurácia.")

# acurácia geral combinando apenas épocas que tiveram rótulo
total_counted = phase1_total + phase2_total
total_correct = phase1_correct + phase2_correct
if total_counted:
    acc_total = total_correct / total_counted * 100
    print(f"\nAcurácia geral (Fase1+Fase2): {total_correct}/{total_counted} = {acc_total:.2f}%")
else:
    print("\nAcurácia geral: sem épocas rotuladas para calcular.")

plt.ioff()
plt.show()
