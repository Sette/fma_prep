
from fma_prep import run
import sys

if __name__=='__main__':
    
    # Defina os argumentos como se fossem passados pela linha de comando
    sys.argv = [
        'fma_prep.py',  # Nome do script (pode ser qualquer string)
        '--input_path', '/home/bruno/storage/data/fma',
        '--output_path', '/home/bruno/storage/data/fma',
        '--top_genres', 'Rock', 'Electronic', 
        '--sequence_size', '1280',
        '--train_id', 'rock_electronic',
        '--sample_size', '1'
    ]
    
    run()