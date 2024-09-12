import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances 

from collections import Counter

def align_sequence(sequence, length):
    if len(sequence) > length:
        return sequence[:length]
    elif len(sequence) < length:
        return sequence + [0] * (length - len(sequence))
    else:
        return sequence

if __name__ == '__main__':

    payloads = [
        ([-186, -198, -198, -198, -198, -186, -186, 216, 209], 'tag1'),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'tag2'),
        ([216, 209], 'tag3')
    ]

    sample_query = [216, 209]
    length = 10  


    aligned_payloads = [(align_sequence(seq, length), tag) for seq, tag in payloads]

    aligned_query = align_sequence(sample_query, length)


    dimension = length
    index = faiss.IndexFlatIP(dimension)  
    faiss.normalize_L2(np.array([seq for seq, tag in aligned_payloads])) 
    index.add(np.array([seq for seq, tag in aligned_payloads], dtype='float32'))


    k = 5
    faiss.normalize_L2(np.array([aligned_query]))
    D, I = index.search(np.array([aligned_query], dtype='float32'), k)

   
    results = [(aligned_payloads[i][0], D[0][j], aligned_payloads[i][1]) for j, i in enumerate(I[0])]

   
    manhattan_sim = manhattan_distances([aligned_query], [seq for seq, tag in aligned_payloads])[0]
    chebyshev_sim = chebyshev_distances([aligned_query], [seq for seq, tag in aligned_payloads])[0]

   
    results_with_other_distances = []
    for i, (seq, sim, tag) in enumerate(results):
        manhattan_score = 1 / (1 + manhattan_sim[I[0][i]])
        chebyshev_score = 1 / (1 + chebyshev_sim[I[0][i]])
        weighted_score = (sim + manhattan_score + chebyshev_score) / 3
        results_with_other_distances.append((seq, weighted_score, tag))


    sorted_results = sorted(results_with_other_distances, key=lambda x: x[1], reverse=True)

 
    tag_counter = Counter([tag for seq, score, tag in sorted_results])


    for seq, score, tag in sorted_results:
        print(f"seq: {seq}, score: {score:.4f}, tag: {tag}")

   
    print(tag_counter)
