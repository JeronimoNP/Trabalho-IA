import os
import sys

# Passo 1: Forçar o Python a encontrar as DLLs do CUDA
# Adicionamos manualmente o caminho da pasta 'bin' do CUDA.
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"

print(f"--- Verificando o caminho do CUDA ---")
print(f"Caminho a ser adicionado: {cuda_path}")

if os.path.exists(cuda_path):
    print("O caminho do CUDA existe.")
    try:
        os.add_dll_directory(cuda_path)
        print("Caminho do CUDA adicionado com sucesso ao carregador de DLL do Python.")
    except Exception as e:
        print(f"Falha ao adicionar o caminho do CUDA: {e}")
else:
    print("ERRO CRÍTICO: O caminho do CUDA não foi encontrado. Verifique a instalação.")
    sys.exit()

print("\n--- Tentando importar o TensorFlow ---")
# Passo 2: Tentar importar o TensorFlow e capturar qualquer erro
try:
    import tensorflow as tf
    print("TensorFlow importado com sucesso!")
except ImportError as e:
    print(f"ERRO DE IMPORTAÇÃO: Não foi possível importar o TensorFlow.")
    print(f"Detalhe do erro: {e}")
    print("Isso geralmente indica que uma DLL essencial (como cudart64_12.dll) não foi encontrada.")
    sys.exit()
except Exception as e:
    print(f"UM ERRO INESPERADO OCORREU: {e}")
    sys.exit()

# Passo 3: Se a importação funcionou, verificar a GPU
print("\n--- Verificando dispositivos GPU ---")
try:
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Versão do TensorFlow: {tf.__version__}")
    print(f"Número de GPUs disponíveis: {len(gpus)}")

    if gpus:
      print("\n✅ SUCESSO! A GPU foi detectada!")
      for gpu in gpus:
        print(f"  - Dispositivo Encontrado: {gpu.name}")
    else:
      print("\n❌ FALHA: A GPU ainda não foi detectada, mesmo forçando o caminho da DLL.")
except Exception as e:
    print(f"Erro ao verificar os dispositivos: {e}")