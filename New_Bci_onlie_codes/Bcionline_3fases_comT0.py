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
fase1_forced_label = 'T1'  # ← 'T1' ou 'T2'
time_fase1 = 76.5
time_fase2 = 81.0
time_fase3 = 67.5
limiar = 0.4
# Sequência de verdadeiros rótulos para Fase 3 (preencher manualmente)
truth_sequence3 = ['T1', 'T0', 'T2', 'T1', 'T0', 'T2', 'T1', 'T0', 'T2', 'T1', 'T0', 'T2', 'T1', 'T0', 'T2']  # exemplo: ['T1','T2','T0', ...]

# Classe de controle de atualização de gráfico
class Sistema:
    def __init__(self):
        self.dt = 4.506  # intervalo de plotagem (s)
sistema = Sistema()

# Auxiliares
label2num = {'T1': 1, 'T2': 2, 'T0': 0}

def normalize_sample(input_sample):
    inp = np.array(input_sample)
    local_max, local_min = inp.max(), inp.min()
    return (inp - local_min) / (local_max - local_min + 1e-8)

def predict(model, input_sample):
    arr = normalize_sample(input_sample)
    arr = np.expand_dims(arr, axis=0)
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
phase1_total = phase1_correct = 0
phase2_total = phase2_correct = 0
phase3_total = phase3_correct = 0
idx3 = 0

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
    if tfase == 1 and elapsed > time_fase1:
        tfase = 2; time_start = time.time(); print("-> Fase 2 iniciada")
    elif tfase == 2 and elapsed > time_fase2:
        tfase = 3; time_start = time.time(); print("-> Fase 3 iniciada")
    elif tfase == 3 and elapsed > time_fase3:
        print("-> Todas as fases concluídas.")
        break

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
        if pred < limiar:
            label = 'T1'
        elif pred > 1 - limiar:
            label = 'T2'
        else:
            label = 'T0'

        # Avalia acerto
        if tfase == 1:
            expected = fase1_forced_label
            phase1_total += 1
            correct = (label == expected)
            if correct: phase1_correct += 1
        elif tfase == 2:
            expected = 'T1' if fase1_forced_label == 'T2' else 'T2'
            phase2_total += 1
            correct = (label == expected)
            if correct: phase2_correct += 1
        else:
            expected = truth_sequence3[idx3] if idx3 < len(truth_sequence3) else None
            idx3 += 1
            phase3_total += 1
            correct = (label == expected)
            if correct: phase3_correct += 1

        # Salva para plot
        pred_values.append(label2num[label])
        pred_colors.append('g' if correct else 'r')

        print(f"[Fase {tfase}] Prev: {label} | Exp: {expected} | Prob: {pred:.3f} | Acerto: {correct}")

        # Transfer Learning se acerto nas fases 1 ou 2
        if tfase in [1,2] and correct:
            bx = np.array([normalize_sample(current_data)])
            by = np.array([label2num[label]]).reshape(-1,1)
            print("=== Iniciando transfer learning (1 época) ===")
            model.fit(bx, by, epochs=1, verbose=1)
            print("=== Transfer learning concluído ===")

        current_data = []

        # Plot dinâmico a cada sistema.dt
        if time.time() - last_plot >= sistema.dt:
            last_plot = time.time()
            ax.cla()
            # plota últimos 100 pontos com cores
            for i, (y, c) in enumerate(zip(pred_values[-100:], pred_colors[-100:])):
                ax.scatter(i, y, c=c, marker='o')
            ax.set_title(f"Live: Fase {tfase} | Label Fase1: {fase1_forced_label}")
            ax.set_ylabel('Label numérico')
            plt.draw(); plt.pause(0.001)

print("\n=== Acurácias finais ===")
if phase1_total:
    print(f"Fase 1: {phase1_correct}/{phase1_total} = {phase1_correct/phase1_total*100:.2f}%")
if phase2_total:
    print(f"Fase 2: {phase2_correct}/{phase2_total} = {phase2_correct/phase2_total*100:.2f}%")
if phase3_total:
    print(f"Fase 3: {phase3_correct}/{phase3_total} = {phase3_correct/phase3_total*100:.2f}%")

plt.ioff()
plt.show()
