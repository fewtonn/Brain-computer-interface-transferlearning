# Configuração:
file_path     = r"C:\Users\batis\OneDrive\Área de Trabalho\Códigos diversos\dados_cobinados.csv"         # Caminho para o seu CSV (input e output são o mesmo)
n_zero_lines  = 1000                       # Quantas linhas de zeros inserir
header_lines = [
    '%OpenBCI Raw EXG Data\n',
    '%Number of channels = 16\n',
    '%Sample Rate = 160 Hz\n',
    '%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n'
]

def insert_zeros_and_header(path, n_zeros, header):
    # Lê todas as linhas originais
    with open(path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 1:
        raise ValueError("Arquivo vazio ou sem linhas suficientes.")

    # Guarda a primeira linha original
    first_line = lines[0:1]
    rest_lines = lines[1:]

    # Cria bloco de zeros com o mesmo número de colunas da segunda linha original
    num_cols   = len(rest_lines[0].strip().split(',')) if rest_lines else len(first_line[0].strip().split(','))
    zero_line  = ','.join(['0'] * num_cols) + '\n'
    zero_block = [zero_line] * n_zeros

    # Monta: cabeçalho % na primeira posição, depois primeira linha original,
    # depois zeros e por fim o restante
    new_lines = header + first_line + zero_block + rest_lines

    # Grava de volta no mesmo arquivo
    with open(path, 'w') as f:
        f.writelines(new_lines)
    print('foi')

# Executa
insert_zeros_and_header(file_path, n_zero_lines, header_lines)