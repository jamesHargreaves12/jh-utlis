from utils import cosine_similarity_custom, embed_prompts
import numpy as np
import json
import faiss

prompts = json.load(open("/Users/jameshargreaves/dev/tmp/all_prompts.json"))


# ## Build index 

embeddings = embed_prompts(prompts)
embeddings = np.array(embeddings).astype('float32')
index = faiss.IndexFlatL2(768)
index.add(np.array(embeddings))


def search(text, k=5):
    emb = embed_prompts(text)
    emb = emb.reshape(1, -1).astype('float32')
    # query_embedding = np.array([emb]).astype('float32')
    distances, indecies = index.search(emb , k)
    return indecies.reshape(k)


close_prompts = search("getting stronger")
for x in close_prompts:
    print(prompts[x])


