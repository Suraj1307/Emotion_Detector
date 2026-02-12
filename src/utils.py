# src/utils.py
import numpy as np
import pickle

def load_embedding_matrix(tokenizer, embedding_path, embedding_dim=100):
    """Creates a weight matrix for the Embedding layer."""
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    print(f"Loading GloVe vectors from {embedding_path}...")
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in tokenizer.word_index:
                idx = tokenizer.word_index[word]
                embedding_matrix[idx] = np.asarray(values[1:], dtype='float32')
    
    return embedding_matrix