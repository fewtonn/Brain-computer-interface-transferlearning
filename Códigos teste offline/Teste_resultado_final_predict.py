import os
import gc
import fnmatch
import numpy as np
import tensorflow as tf
import mne
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# ... (mantenha as funções processar_arquivos e separar_dados inalteradas) ...
def processar_arquivos(diretorio_raiz, lista_epocas, tarefas, pessoas_escolhidas, limite_por_evento=None):
    """
    Processa os arquivos EEG das pessoas selecionadas.
    """
    if not os.path.exists(diretorio_raiz):
        raise FileNotFoundError(f"Diretório raiz '{diretorio_raiz}' não encontrado.")

    canais_desejados = ['Fp1.', 'F7..', 'F3..', 'T7..', 'C3..', 'P7..', 'P3..', 'O1..',
                        'Fp2.', 'F4..', 'F8..', 'C4..', 'T8..', 'P4..', 'P8..', 'O2..']

    epocas_por_evento = {evento: [] for evento in lista_epocas}
    total_arquivos_eventos = {evento: 0 for evento in lista_epocas}

    # Listar todas as pastas no diretório raiz e filtrar pelas pessoas escolhidas
    todas_pastas = [p for p in os.listdir(diretorio_raiz) if os.path.isdir(os.path.join(diretorio_raiz, p))]
    pastas_selecionadas = []
    
    for pasta in todas_pastas:
        try:
            # Extrair número da pasta (ex: S001 -> 1, S085 -> 85)
            numero_pasta = int(pasta[1:4])
            if numero_pasta in pessoas_escolhidas:
                pastas_selecionadas.append(pasta)
        except:
            continue

    print("Pastas selecionadas:", pastas_selecionadas)

    for pasta_pessoa in pastas_selecionadas:
        caminho_pasta = os.path.join(diretorio_raiz, pasta_pessoa)
        arquivos_edf = []
        
        for tarefa in tarefas:
            arquivos_edf.extend(fnmatch.filter(os.listdir(caminho_pasta), tarefa))
        
        print(f'Processando {pasta_pessoa}...')
        
        raws = []
        for arquivo_edf in arquivos_edf:
            caminho_arquivo = os.path.join(caminho_pasta, arquivo_edf)
            raw = mne.io.read_raw_edf(caminho_arquivo, preload=True)
            raw.pick_channels(canais_desejados)
            raws.append(raw)
        
        if raws:
            raw_concatenado = mne.concatenate_raws(raws)
            events, event_id = mne.events_from_annotations(raw_concatenado)

            for evento in lista_epocas:
                if evento in event_id:
                    epochs = mne.Epochs(raw_concatenado, events, event_id={evento: event_id[evento]}, 
                                        tmin=-0.5, tmax=4, baseline=(-0.5, 0))
                    epocas = epochs.get_data().astype(np.float32)
                    epocas_por_evento[evento].extend(epocas)
                    total_arquivos_eventos[evento] += len(epocas)

    print("\nQuantidade de arquivos por evento:")
    for evento, quantidade in total_arquivos_eventos.items():
        print(f"{evento}: {quantidade} arquivos")

    if limite_por_evento is None:
        recomendacao = int(np.mean([q for q in total_arquivos_eventos.values() if q > 0]))
        print(f"\nRecomendação: Use um limite de {recomendacao} arquivos por evento.")
        limite_por_evento = recomendacao
    
    for evento, epocas in epocas_por_evento.items():
        if len(epocas) > limite_por_evento:
            np.random.shuffle(epocas)  
            epocas_por_evento[evento] = epocas[:limite_por_evento]
            print(f'{evento} reduzido para {len(epocas_por_evento[evento])} arquivos')

    return epocas_por_evento

# --- Função separar_dados (incluída no código) ---
def separar_dados(epocas_por_evento):
    arrays = {}
    for evento, epocas in epocas_por_evento.items():
        arrays[evento] = np.array(epocas) if epocas else np.array([])
        print(f'{evento}_array shape:', arrays[evento].shape)

    del epocas_por_evento
    gc.collect()

    data = np.concatenate([array for array in arrays.values() if array.size > 0])
    print("Data shape:", np.shape(data))

    labels = []
    for i, (evento, array) in enumerate(arrays.items()):
        labels.extend([i] * len(array))

    del arrays
    gc.collect()

    x = np.nan_to_num(data)
    y = np.array(labels)

    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x = (x - x_min) / (x_max - x_min + 1e-8)

    print("Shape do x:", x.shape)
    print("Shape do y:", y.shape)
    print("Valores únicos em y:", np.unique(y))

    return x, y

# --- Código Principal ---
def calcular_acuracia_individuo(modelo, x, y):
    """Calcula a acurácia para um indivíduo usando o modelo pré-treinado."""
    # Fazer as predições
    y_pred = modelo.predict(x, verbose=0)
    
    # Converter probabilidades em classes (0 ou 1)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Calcular acurácia
    return accuracy_score(y, y_pred_classes)

if __name__ == "__main__":
    # 1. Carregar modelo pré-treinado
    modelo = load_model(r"C:\Users\batis\Downloads\checkpoints\melhor_modelo_fold_25_acc_0.8952.h5")
    
    # 2. Configurações
    diretorio_raiz = r"C:\Users\batis\OneDrive\Área de Trabalho\Códigos diversos\Faculdade\BCI\Dados BRUTOS\files"
    lista_eventos = ["T1", "T2"]
    tarefas = ["*R04.edf", "*R08.edf", "*R12.edf"]
    resultados = {}

    # 3. Processar pessoas de 85 a 109
    pessoas_escolhidas = list(range(85, 110))

    for pessoa_num in pessoas_escolhidas:
        print(f"\n=== Processando S{pessoa_num:03d} ===")
        
        try:
            # 3.1. Processar dados da pessoa
            epocas = processar_arquivos(
                diretorio_raiz=diretorio_raiz,
                lista_epocas=lista_eventos,
                tarefas=tarefas,
                pessoas_escolhidas=[pessoa_num],
                limite_por_evento=None
            )
            
            # 3.2. Separar e normalizar dados
            x, y = separar_dados(epocas)
            x = x.transpose(0, 2, 1)  # Ajustar formato para o modelo

            # 3.3. Verificar dados válidos
            if x.size == 0 or len(np.unique(y)) < 2:
                print(f"Dados insuficientes para S{pessoa_num:03d}")
                resultados[pessoa_num] = None
                continue

            # 3.4. Fazer predições e calcular acurácia
            acc = calcular_acuracia_individuo(modelo, x, y)
            resultados[pessoa_num] = acc
            print(f"Acurácia para S{pessoa_num:03d}: {acc * 100:.2f}%")

            # 3.5. Limpar memória
            del epocas, x, y
            gc.collect()

        except Exception as e:
            print(f"Erro em S{pessoa_num:03d}: {str(e)}")
            resultados[pessoa_num] = None
            continue

    # 4. Salvar resultados detalhados
    with open("resultados_individuais.csv", "w") as f:
        f.write("Pessoa,Acurácia\n")
        for pessoa, acc in resultados.items():
            if acc is not None:
                f.write(f"S{pessoa:03d},{acc:.4f}\n")
            else:
                f.write(f"S{pessoa:03d},N/A\n")

    print("\nProcessamento concluído! Resultados salvos em resultados_individuais.csv")