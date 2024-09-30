import numpy as np
import json
from sklearn.cluster import KMeans
from utils import cosine_similarity_custom, embed_prompts

def k_means_clustering(embeddings, n_clusters=50):
    embeddings_array = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings_array)
    return kmeans.labels_


if __name__ == "__main__":
    prompts = json.load(open("/Users/jameshargreaves/dev/tmp/all_prompts.json"))
    embeddings = embed_prompts(prompts)
    n_clusters = 70
    cluster_labels = k_means_clustering(embeddings, n_clusters)
    clusters = [[] for _ in range(n_clusters)]
    for prompt, lab in zip(prompts, cluster_labels):
        clusters[lab].append(prompt)

    clustered_prompts = sorted(clusters, key=lambda x: len(x), reverse=True)
    
    # Find most representative of the cluster by finding the one closest to the center
    top_prompt_to_prompts = {}
    for cluster_i in range(n_clusters):
        cluster_indexes = [emb_i for emb_i, lab in enumerate(cluster_labels) if lab == cluster_i]
        cluster_prompts = [prompts[i] for i in cluster_indexes]
        cluster_embeddings = [embeddings[i] for i in cluster_indexes]
        if len(cluster_embeddings) < 6:
            continue

        center = np.mean(np.array(cluster_embeddings), axis=0)

        max_cosine_sim = cosine_similarity_custom(center, cluster_embeddings[0])
        closest_index = 0
        for i, e in enumerate(cluster_embeddings):
            cosine_sim = cosine_similarity_custom(center, e)
            if cosine_sim > max_cosine_sim:
                max_cosine_sim = cosine_sim
                closest_index = i
        representative_prompt = cluster_prompts[closest_index]
        top_prompt_to_prompts[representative_prompt] = cluster_prompts

        json.dump(top_prompt_to_prompts, open("./clustered_prompts.json", "w"), indent=2)
