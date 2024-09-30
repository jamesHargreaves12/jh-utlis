import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np

torch.set_default_device("mps")

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')

def batch(lst, batch_size=100):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

def cosine_similarity_custom(X, Y):
    dot_product = np.dot(X, Y.T)
    norm_x = np.linalg.norm(X)
    norm_y = np.linalg.norm(Y)
    return dot_product / (norm_x * norm_y)

def _embed_prompts(input_texts):
    embeddings = []
    if len(input_texts) > 150:
        iterator = tqdm(list(batch(input_texts, 150)), desc="Embedding texts")
    else:
        iterator = input_texts
    for batch_texts in iterator:
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        batch_embs = torch.nn.functional.normalize(sentence_embeddings.cpu(), p=2, dim=1).numpy()
        embeddings.extend(batch_embs)
    return embeddings

def embed_prompts(input_texts):
    if isinstance(input_texts, str):
        return _embed_prompts([input_texts])[0]
    return _embed_prompts(input_texts)