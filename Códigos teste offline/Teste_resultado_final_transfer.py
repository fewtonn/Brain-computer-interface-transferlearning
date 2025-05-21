import os
import gc
import fnmatch
import numpy as np
import tensorflow as tf
import mne
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Flatten,
                                   Dense, Dropout, Reshape, GlobalAveragePooling1D,
                                   BatchNormalization)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

def processar_arquivos(diretorio_raiz, lista_epocas, tarefas, pessoas_escolhidas, limite_por_evento=None):
    """Processa arquivos EDF e extrai épocas"""
    if not os.path.exists(diretorio_raiz):
        raise FileNotFoundError(f"Diretório raiz '{diretorio_raiz}' não encontrado.")

    canais_desejados = ['Fp1.', 'F7..', 'F3..', 'T7..', 'C3..', 'P7..', 'P3..', 'O1..',
                        'Fp2.', 'F4..', 'F8..', 'C4..', 'T8..', 'P4..', 'P8..', 'O2..']

    epocas_por_evento = {evento: [] for evento in lista_epocas}
    total_arquivos_eventos = {evento: 0 for evento in lista_epocas}

    todas_pastas = [p for p in os.listdir(diretorio_raiz) if os.path.isdir(os.path.join(diretorio_raiz, p))]
    pastas_selecionadas = []
    
    for pasta in todas_pastas:
        try:
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

def separar_dados(epocas_por_evento):
    """Prepara e normaliza os dados para treinamento"""
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
    x = x.transpose(0, 2, 1)  # (samples, time_steps, channels)

    print("Shape final do x:", x.shape)
    print("Shape do y:", y.shape)
    print("Valores únicos em y:", np.unique(y))

    return x, y

def criar_modelo_transfer(base_model):
    """Arquitetura otimizada para transfer learning"""
    input_tensor = Input(shape=base_model.input_shape[1:])  # (721, 16)
    
    # Congelamento adaptativo
    for layer in base_model.layers:
        layer.trainable = False  # Começa congelando tudo
    for layer in base_model.layers[-4:]:  # Descongela últimas 4 camadas
        if 'conv' in layer.name:
            layer.trainable = True

    # Pipeline de processamento
    x = base_model(input_tensor)
    
    # Adaptação dimensional com verificação
    if len(x.shape) == 2:
        x = Reshape((x.shape[1], 1))(x)
    
    # Bloco convolucional adicional
    x = Conv1D(16, 5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    
    # Camadas densas otimizadas
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(0.5)(x)
    
    saida = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=input_tensor, outputs=saida)

def scheduler(epoch, lr):
    """Agendador de learning rate adaptativo"""
    return lr * 0.95 if epoch > 15 else lr

def plotar_historico(history, sujeito_id):
    """Visualização melhorada do treinamento"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title(f'Acurácia - {sujeito_id}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Loss - {sujeito_id}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Configurações ajustadas
    modelo_base = load_model(r"C:\Users\batis\Downloads\checkpoints\melhor_modelo_fold_25_acc_0.8952.h5")
    diretorio_raiz = r"C:\Users\batis\OneDrive\Área de Trabalho\Códigos diversos\Faculdade\BCI\Dados BRUTOS\files"
    
    # Hiperparâmetros otimizados
    config = {
        'lista_eventos': ["T1", "T2"],
        'tarefas': ["*R04.edf", "*R08.edf", "*R12.edf"],
        'epochs': 60,
        'batch_size': 16,
        'lr_inicial': 1e-4,
        'patience': 15
    }

    todos_sujeitos = [f"S{p:03d}" for p in range(85, 110)]
    resultados = {sujeito: {'Pessoa': sujeito, 'Val_acc': np.nan} for sujeito in todos_sujeitos}

    for pessoa_num in range(85, 110):
        sujeito_id = f"S{pessoa_num:03d}"
        print(f"\n=== Processando {sujeito_id} ===")
        
        try:
            # Carregamento e preparação de dados
            epocas = processar_arquivos(
                diretorio_raiz=diretorio_raiz,
                lista_epocas=config['lista_eventos'],
                tarefas=config['tarefas'],
                pessoas_escolhidas=[pessoa_num],
                limite_por_evento=50  # Limite máximo por evento
            )
            
            x, y = separar_dados(epocas)
            
            # Verificação de qualidade dos dados
            if len(np.unique(y)) < 2 or x.shape[0] < 30:
                raise ValueError("Dados insuficientes ou desbalanceados")
                
            # Balanceamento dinâmico
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            class_weights = {i:w for i,w in enumerate(class_weights)}

            # Divisão estratificada
            X_train, X_test, Y_train, Y_test = train_test_split(
                x, y, 
                test_size=0.25,  # Mais dados para validação
                stratify=y,
                random_state=42
            )

            # Construção do modelo
            modelo_transfer = criar_modelo_transfer(modelo_base)
            modelo_transfer.compile(
                optimizer=Adam(learning_rate=config['lr_inicial']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # Callbacks aprimorados
            early_stop = EarlyStopping(
                monitor='val_accuracy',
                patience=config['patience'],
                mode='max',
                restore_best_weights=True
            )

            checkpoint = ModelCheckpoint(
                filepath=f"melhor_modelo_{sujeito_id}.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )

            # Treinamento monitorado
            history = modelo_transfer.fit(
                X_train, Y_train,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                validation_data=(X_test, Y_test),
                class_weight=class_weights,
                callbacks=[early_stop, checkpoint, LearningRateScheduler(scheduler)],
                verbose=1  # Mostra progresso
            )

            # Avaliação final
            melhor_val_acc = max(history.history['val_accuracy'])
            resultados[sujeito_id]['Val_acc'] = melhor_val_acc
            print(f"\nResultado final para {sujeito_id}: {melhor_val_acc:.4f}")

            # Diagnóstico visual
            plotar_historico(history, sujeito_id)

        except Exception as e:
            print(f"\nErro crítico em {sujeito_id}: {str(e)}")
            resultados[sujeito_id]['Val_acc'] = np.nan
        
        finally:
            gc.collect()

    # Salvamento robusto
    df = pd.DataFrame(resultados.values())
    df = df[['Pessoa', 'Val_acc']].sort_values('Pessoa').reset_index(drop=True)
    df.to_csv("resultados_finais_transfer_learning.csv", index=False)
    print("\nProcessamento concluído! Resultados salvos em 'resultados_finais_transfer_learning.csv'")