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

# ---------- Defina AQUI o tempo da FASE 1 (em segundos) ----------IMPORTANTE NÃO ESQUECER DEFINIR O TEMPO afhudpfadghfçadhfaÇÁHF ÓUFHASDPOUFHADPFUADHF IOMPORTANTE VEJA AUJDFNAJOFNADJFNADLFÇJANDFÇKLJANDFLÇKJADN FLKADJFBNADLJFNADLFJBNADFJLANFADÇKFNMADLFJN ADF
TIME_PHASE1 = 193.7625  # duração da fase 1 (com TL); após isso começa a Fase 2 (que dura o resto)

# ---------- Defina AQUI as labels esperadas para cada época de cada fase ----------
LABELS_PHASE1 = [
 'T0','T1','T1','T0','T0','T0','T0','T0','T0','T1','T1','T1','T0','T0',
 'T0','T0','T0','T2','T2','T0','T0','T1','T2','T2','T0','T0','T0','T2',
 'T0','T2','T1','T0','T0','T0','T2','T2','T0','T2','T0','T0','T0','T1','T0','T1'
]

LABELS_PHASE2 = [
 'T1','T0','T1','T2','T0','T2','T2','T1','T2','T0','T0','T1','T1','T2',
 'T0','T0','T0','T2','T0','T1','T0','T0','T0','T1','T1','T0','T2','T1',
 'T2','T1','T0','T0','T2','T1','T2','T2','T2','T1','T2','T2','T0','T1','T0'
]



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
phase1_end_time = None

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

    # Atualiza fase
    elapsed = time.time() - time_start
    if tfase == 1 and elapsed > TIME_PHASE1:
        tfase = 2
        phase1_end_time = time.time()
        phase1_elapsed = phase1_end_time - time_start
        print("-> Fase 2 iniciada (sem Transfer Learning).")
        print(f"Fim da Fase 1: {phase1_elapsed:.6f} s")
        print(f"Formato legível: {pd.to_timedelta(phase1_elapsed, unit='s')}")

    # Puxa chunk
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

# Determina expected a partir das listas
        expected = None
        correct = False

        if tfase == 1:
            if idx_phase1 < len(LABELS_PHASE1):
                expected = LABELS_PHASE1[idx_phase1]
                idx_phase1 += 1
                phase1_total += 1           # agora conta T0 também
                correct = (label == expected)
                if correct:
                    phase1_correct += 1

        # Transfer Learning só se acertou e não for T0
                if correct and expected != 'T0':
                    bx = np.array([normalize_sample(current_data)])
                    by = np.array([label2num[label]]).reshape(-1,1)
                    print("=== Iniciando transfer learning (1 época) ===")
                    model.fit(bx, by, epochs=1, verbose=1)
                    print("=== Transfer learning concluído ===")
        else:
            if idx_phase2 < len(LABELS_PHASE2):
                expected = LABELS_PHASE2[idx_phase2]
                idx_phase2 += 1
                phase2_total += 1           # agora conta T0 também
                correct = (label == expected)
                if correct:
                    phase2_correct += 1
# ---------------------------------------------------------------------------
        # Salva para plot
        pred_values.append(label2num[label])
        if label == 'T0':
            pred_colors.append('b')   # azul para T0
        else:
            pred_colors.append('g' if correct else 'r')

        # Impressão por época
        print(f"[Fase {tfase}] Prev: {label} | Exp: {expected} | Prob: {pred:.3f} | Acerto: {correct}")

        # limpa janela
        current_data = []

        # Plot dinâmico
        if time.time() - last_plot >= sistema.dt:
            last_plot = time.time()
            ax.cla()
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
    print("Fase 1: sem épocas rotuladas (ou só T0).")

if phase2_total:
    acc2 = phase2_correct / phase2_total * 100
    print(f"Fase 2: {phase2_correct}/{phase2_total} = {acc2:.2f}%")
else:
    print("Fase 2: sem épocas rotuladas (ou só T0).")

total_counted = phase1_total + phase2_total
total_correct = phase1_correct + phase2_correct
if total_counted:
    acc_total = total_correct / total_counted * 100
    print(f"\nAcurácia geral: {total_correct}/{total_counted} = {acc_total:.2f}%")
else:
    print("\nAcurácia geral: sem épocas rotuladas.")

plt.ioff()
plt.show()
