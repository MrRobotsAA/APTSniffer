import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import requests
import json
from collections import Counter
import numpy as np
import faiss
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances 
from scipy.spatial.distance import chebyshev
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from a2_rag2_sim import align_sequence
from a3_Corrs_2_ok import complete_tags_graph
from main2_2_predict import query_llm_function
import torch


from a1_sam3_ok_label_distribution_gaijin2 import SimpleStringMatcher


def submodel_1_train(train_data):
    sa = SimpleStringMatcher()
    payloads = train_data[['payload_lengths', 'label']].values.tolist()

    
    for payload, tag in payloads:
        sa.insert_payload(payload, tag)

    return sa

def submodel_1_test(sa,sample_query_list):
    # sa = submodel_1(test_data)
    sample_query = [216, 209]
    count_s, tag_s = sa.count_occurrences(sample_query)
    print("Occurrences of [216, 209]:", count_s)
    print("Tags distribution of [216, 209]:", tag_s)

    tags = []
    counts = []
    for sample_query in sample_query_list:
        count, tag = sa.count_occurrences(sample_query)
        tags.append(tag)
        counts.append(count)
    return counts,tags


def submodel_2_train(aligned_payloads, length):
    
    aligned_sequences = np.array([seq for seq, _ in aligned_payloads if isinstance(seq, list)], dtype='float32')

   
    if aligned_sequences.shape[1] != length:
        raise ValueError(f"Expected sequence length {length}, but got {aligned_sequences.shape[1]}")
    print(aligned_sequences.shape)


    dimension = length
    index = faiss.IndexFlatIP(dimension) 
    faiss.normalize_L2(aligned_sequences)
    index.add(aligned_sequences)
    # return index, aligned_payloads
    return index

from collections import Counter
from tqdm import tqdm
import faiss


def submodel_2_test_man(index, aligned_payloads,sample_query, k=5):
   
    results_list = []
    from tqdm import tqdm
  
    for single_query in tqdm(range(len(sample_query)), desc="Processing queries"):

        aligned_query = np.array([sample_query[single_query]], dtype='float32')
       
        faiss.normalize_L2(aligned_query)
        D, I = index.search(aligned_query, k)
        results = [(aligned_payloads[i][0], D[0][j], aligned_payloads[i][1]) for j, i in enumerate(I[0])]

  
        aligned_payload_sequences = np.array([seq for seq, _ in aligned_payloads], dtype='float32')
        manhattan_sim = manhattan_distances(aligned_query, aligned_payload_sequences)[0]
        chebyshev_sim = euclidean_distances(aligned_query, aligned_payload_sequences)[0]

  
        results_with_other_distances = []
        for i, (seq, sim, tag) in enumerate(results):
            manhattan_score = 1 / (1 + manhattan_sim[I[0][i]])
            chebyshev_score = 1 / (1 + chebyshev_sim[I[0][i]])
            weighted_score = (sim + manhattan_score + chebyshev_score) / 3
            results_with_other_distances.append((seq, weighted_score, tag))

  
        sorted_results = sorted(results_with_other_distances, key=lambda x: x[1], reverse=True)

  
        tag_counter = Counter([tag for seq, score, tag in sorted_results])


        results_list.append(tag_counter)
       


    return results_list


def submodel_2_test_fast1(index, aligned_payloads, sample_query, k=5):
    results_list = []
   
    sample_query_array = np.array(sample_query, dtype='float32')
    faiss.normalize_L2(sample_query_array)
   
    D, I = index.search(sample_query_array, k)
    aligned_payload_sequences = np.array([seq for seq, _ in aligned_payloads], dtype='float32')


    for single_query_idx in tqdm(range(len(sample_query)), desc="Processing queries"):
        aligned_query = sample_query_array[single_query_idx:single_query_idx + 1]
        results = [(aligned_payloads[i][0], D[single_query_idx][j], aligned_payloads[i][1]) for j, i in
                   enumerate(I[single_query_idx])]

       
        manhattan_sim = manhattan_distances(aligned_query, aligned_payload_sequences)[0]
        chebyshev_sim = euclidean_distances(aligned_query, aligned_payload_sequences)[0]

    
        results_with_other_distances = []
        for i, (seq, sim, tag) in enumerate(results):
            manhattan_score = 1 / (1 + manhattan_sim[I[single_query_idx][i]])
            chebyshev_score = 1 / (1 + chebyshev_sim[I[single_query_idx][i]])
            weighted_score = (sim + manhattan_score + chebyshev_score) / 3
            results_with_other_distances.append((seq, weighted_score, tag))

       
        sorted_results = sorted(results_with_other_distances, key=lambda x: x[1], reverse=True)


        tag_counter = Counter([tag for seq, score, tag in sorted_results])
        results_list.append(tag_counter)

    return results_list

import numpy as np
import faiss
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from collections import Counter
from tqdm import tqdm
def submodel_2_test(index, aligned_payloads, sample_query, k=5):
    results_list = []
   
    sample_query_array = np.array(sample_query, dtype='float32')
    faiss.normalize_L2(sample_query_array)


    D, I = index.search(sample_query_array, k)

    for single_query_idx in tqdm(range(len(sample_query)), desc="Processing queries"):
        
        aligned_query = sample_query_array[single_query_idx:single_query_idx+1]

        topk_indices = I[single_query_idx]

        
        topk_payloads = [aligned_payloads[i] for i in topk_indices]  # 修改这里


        topk_payload_sequences = np.array([seq for seq, _ in topk_payloads], dtype='float32')
        manhattan_sim = manhattan_distances(aligned_query, topk_payload_sequences)[0]
        chebyshev_sim = euclidean_distances(aligned_query, topk_payload_sequences)[0]

      
        results_with_other_distances = []
        for i, payload in enumerate(topk_payloads):
            seq, tag = payload
            faiss_score = D[single_query_idx][i]
            manhattan_score = 1 / (1 + manhattan_sim[i])
            chebyshev_score = 1 / (1 + chebyshev_sim[i])
            weighted_score = (faiss_score + manhattan_score + chebyshev_score) / 3
            results_with_other_distances.append((seq, weighted_score, tag))

     
        sorted_results = sorted(results_with_other_distances, key=lambda x: x[1], reverse=True)


        tag_counter = Counter(tag for _, _, tag in sorted_results)
        results_list.append(tag_counter)

    return results_list



from a3_Corrs_2_ok import query_graph,build_graph_and_distributions



def submodel_3_test(G, value_node_map, label_distribution,train_data,test_data,data_label):


    debug_top = 0
    from tqdm import tqdm
   
    traverse_results = []
    for i, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Processing rows"):
        query_sample = {
            'server_ip': row['server_ip'],
            'payload_lengths': row['payload_lengths'],
            'payload_timestamps': row['payload_timestamps'],
            'ja4': row['ja4'],
            'ja4s': row['ja4s'],
            'requested_server_name': row['requested_server_name'],
            'application_category_name': row['application_category_name'],
        }

        label_counts = query_graph(G, value_node_map, label_distribution, query_sample, train_data)
      
        all_tags = set(data_label)
        completed_tag_counts = complete_tags_graph(all_tags, label_counts)
        traverse_results.append(completed_tag_counts)


    return traverse_results

def transform_data(train_data_samples, normalize=True):
    def normalize_value(value, max_value=1.0, min_value=0.0):
        return (value - min_value) / (max_value - min_value)

    def get_max_min(data):
        all_values = [v for d in data for subdict in d[0].values() for v in subdict.values()]
        return max(all_values), min(all_values)

    max_value, min_value = get_max_min(train_data_samples) if normalize else (1.0, 0.0)

    transformed_data = []
    for sample in train_data_samples:
        transformed_sample = {}
        for key, value_dict in sample[0].items():
            transformed_value_list = []
            for k, v in value_dict.items():
                norm_v = normalize_value(v, max_value, min_value) if normalize else v
                transformed_value_list.append([int(k), norm_v])  
            transformed_value_list.sort(key=lambda x: x[0])  
            transformed_sample[key] = transformed_value_list
        transformed_data.append((transformed_sample, sample[1]))

    return transformed_data



from train.a4_reward2_ok import MultiTypeLabelPredictor
from train.a4_reward2_ok import train_model


def submodel_4_train(train_data):
  
    model = MultiTypeLabelPredictor()
    train_model(model, train_data)

    def predict(model, input_sample):
        model.eval()  
        with torch.no_grad():
            output = model(input_sample)
            predicted_label = torch.argmax(output).item()
        return predicted_label


    input_sample = {0: [[0, 0]], 1: [[1.0, 0]], 2: [[0, 0]]}
    predicted_label = predict(model, input_sample)
    print(f"Predicted Label: {predicted_label}")

    
    for i, weight_matrix in enumerate(model.weight_matrices):
        print(f"Weight matrix for type {i}:\n{weight_matrix.data.numpy()}")

    num_types = 3
    num_labels = 2
    
    average_matrix = np.zeros((num_types, num_labels))

    ans_list = []

    for i, weight_matrix in enumerate(model.weight_matrices):
        print(f"Weight matrix for type {i}:\n{weight_matrix}")
        weight_matrix = weight_matrix.data.numpy()
        weight_matrix = np.sum(weight_matrix, axis=0)
        weight_matrix = weight_matrix / np.sum(weight_matrix) if np.sum(weight_matrix) != 0 else weight_matrix

        ans_list.append(weight_matrix)


    ans = np.array(ans_list)
    ans = np.round(ans, 2)
    print("numpy-ans:", ans)

    
    ans = ans.tolist()
    ans = [[round(j, 4) for j in i] for i in ans]
    print("list-ans:", ans)

    return ans





# (fake_model.py)
def call_fake_model(data):
    url = "http://localhost:8000/v1/classify/completions"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    return response.json()



def label_distribution_statistics(train_data):
    label_distribution = Counter(train_data['label'])   
    print(label_distribution)


def align_sequence2(sequence, length):
    if not isinstance(sequence, list):
        try:
            sequence = eval(sequence)
        except Exception as e:
            raise ValueError(f"Invalid sequence format: {sequence}. Error: {e}")

    if not isinstance(sequence, list):
        raise ValueError(f"Sequence after eval is not a list: {sequence}")

    if len(sequence) > length:
        return sequence[:length]
    else:
        return sequence + [0] * (length - len(sequence))

def merge_samples(dict1, dict2, dict3, data_label):
    merged_data = {}
    all_ids = set(dict1.keys()).union(dict2.keys()).union(dict3.keys())

    for sample_id in all_ids:
        merged_sample = {0: {}, 1: {}, 2: {}}

        # Initialize all labels with 0
        for label in data_label:
            merged_sample[0][label] = 0
            merged_sample[1][label] = 0
            merged_sample[2][label] = 0

        if sample_id in dict1:
            for key, value in dict1[sample_id].items():
                merged_sample[0][key] = value
        if sample_id in dict2:
            for key, value in dict2[sample_id].items():
                merged_sample[1][key] = value
        if sample_id in dict3:
            # print("sample_id", sample_id)
            # print("dict3[sample_id]:", dict3[sample_id])
            for key, value in dict3[sample_id].items():
                merged_sample[2][key] = value

        merged_data[sample_id] = merged_sample

    return merged_data


def format_train_data(merged_samples, labels):
    train_data = []
    for sample_id, sample_data in merged_samples.items():
        if sample_id in labels:
            train_data.append((sample_data, labels[sample_id]))
    return train_data


def generate_prompts_en(merged_samples, list_ans):
    prompts = []
    for key, sample in merged_samples.items():
        prompt = f"Sample ID: {key}\n"
        prompt += "Exact Match:\n"
        for k, v in sample[0].items():
            prompt += f"  Label {k} count distribution: {v}\n"
            prompt += f"  Recommended training weight: {list_ans[0][k]}\n"
        prompt += "Fuzzy Match:\n"
        for k, v in sample[1].items():
            prompt += f"  Label {k} count distribution: {v}\n"
            prompt += f"  Recommended training weight: {list_ans[1][k]}\n"
        prompt += "Relational Match:\n"
        for k, v in sample[2].items():
            prompt += f"  Label {k} count distribution: {v}\n"
            prompt += f"  Recommended training weight: {list_ans[2][k]}\n"
        prompts.append(prompt)
    return prompts


def main():
  

    data = pd.read_csv('anyrun2024_flowcontainer_and_nfstream.csv')
    data_label = set(data['label'])
    
    label_distribution = data['label'].value_counts()
    

    csv_features = ['label', 'pcapname', 'src_ip', 'dst_ip', 'sport', 'dport', 'protocol', 'payload_lengths',
                    'ip_lengths',
                    'payload_timestamps', 'ip_timestamps', 'ja3', 'ja3s', 'ja4', 'ja4s', 'requested_server_name',
                    'application_category_name', 'id']

    data['id'] = range(1, len(data) + 1)


    data = data.rename(columns={'Label': 'label'})
    
    data = data.rename(columns={'src_ip_x': 'src_ip', 'dst_ip_x': 'dst_ip', 'protocol_x': 'protocol'})
   
    data = data[csv_features]


    data['label'] = data['label'].apply(lambda x: 0 if x == 2 else 1)
    print(data.columns.values.tolist())
    data_label = set(data['label'])
    label_distribution = data['label'].value_counts()


    data['aligned_payloads'] = data['payload_lengths'].apply(lambda x: align_sequence2(x, 50))
    print(data['aligned_payloads'][0:3])

    data['server_ip'] = data.apply(lambda row: row['src_ip'] if row['sport'] < row['dport'] else row['dst_ip'], axis=1)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
   
    label_distribution = Counter(train_data['label'])
    print(label_distribution)
    print("train_data.columns:", train_data.columns.values.tolist())


    # 2. 
    sa_model = submodel_1_train(train_data)
    tmp_lengths = train_data[['aligned_payloads', 'label']].values.tolist()
    aligned_payloads = [(seq, tag) for seq, tag in tmp_lengths]
    rag2_sim_model = submodel_2_train(aligned_payloads, 50)

    # sub_model3_G = build_graph(train_data)
    sub_model3_G, value_node_map, label_distribution = build_graph_and_distributions(train_data)
    

    # 3. 

    sample_train_data = train_data.sample(n=100, random_state=42)
    bak_train_data = train_data
    train_data = sample_train_data

    # 4.
    sample_query_list = train_data['payload_lengths'].values.tolist()
    count,tags = submodel_1_test(sa_model,sample_query_list)

    id_tags = dict(zip(train_data['id'],tags))
    
    with open('./data2/debug_id_tags.json', 'w') as f:
        json.dump(id_tags, f)
    print("id_tags build success!")

    sample_query_list = train_data['aligned_payloads'].values.tolist()
    tags = submodel_2_test(rag2_sim_model,aligned_payloads,sample_query_list)

    id_tags2 = dict(zip(train_data['id'],tags))
    with open('./data2/debug_id_tags2.json', 'w') as f:
        json.dump(id_tags, f)
    print("debug_id_tags2 create success!")

  
    traverse_results = submodel_3_test(sub_model3_G, value_node_map, label_distribution,bak_train_data,train_data,data_label)

    id_tags3 = dict(zip(train_data['id'],traverse_results))
    
    with open('./data2/debug_id_tags3.json', 'w') as f:
        json.dump(id_tags, f)
    print("debug_id_tags3 create success!")


    # 4. 
    merged_sampless = merge_samples(id_tags, id_tags2, id_tags3, data_label)
 
    label_lengths = {0: 2, 1: 2, 2: 2}
    normalized_samples = merged_sampless
   

  
    labels = dict(zip(train_data['id'], train_data['label']))
    print("len(labels),len(merged_sampless),train:",len(labels),len(merged_sampless))

    train_data_samples = format_train_data(normalized_samples, labels)
    

    
    with open('./data2/train_data_samples_anyrun2024.txt', 'w') as f:
        f.write(str(train_data_samples))


    transformed_data = transform_data(train_data_samples, normalize=True)
    reward_weighted_values = submodel_4_train(transformed_data)
    

    # 5.
    #submodel_1
    sample_query_list = test_data['payload_lengths'].values.tolist()
    count,tags = submodel_1_test(sa_model,sample_query_list)
    id_tags = dict(zip(test_data['id'],tags))
    with open('./data2/debug_id_tags_test.json', 'w') as f:
        json.dump(id_tags, f)
    print("id_tags test build success!")

    #submodel_2
    sample_query_list = test_data['aligned_payloads'].values.tolist()
    tags = submodel_2_test(rag2_sim_model,aligned_payloads,sample_query_list)
    id_tags2 = dict(zip(test_data['id'],tags))
    with open('./data2/debug_id_tags2_test.json', 'w') as f:
        json.dump(id_tags, f)
    print("id_tags2 test build success!")

    #submodel_3
    traverse_results = submodel_3_test(sub_model3_G, value_node_map, label_distribution,train_data,test_data,data_label)
    id_tags3 = dict(zip(test_data['id'],traverse_results))
    with open('./data2/debug_id_tags3_test.json', 'w') as f:
        json.dump(id_tags, f)
    print("id_tags3 test build success!")

    # 6. 
    merged_samples_test = merge_samples(id_tags, id_tags2, id_tags3, data_label)
    normalized_samples = merged_samples_test
   
    labels_test = dict(zip(test_data['id'], test_data['label']))
    test_data_label =  test_data['label']
    print("len(labels),len(merged_samples_test):",len(test_data_label),len(merged_samples_test))


    with open('./data2/test_labels_anyrun2024.txt', 'w') as f:
        f.write(str(labels_test))

    with open('./data2/merged_samples_test_anyrun2024.txt', 'w') as f:
        f.write(str(merged_samples_test))


    with open('./data2/list_ans_anyrun2024.txt', 'w') as f:
        f.write(str(reward_weighted_values))



    return 0


    with open('./data/test_labels.txt', 'r') as f:
        test_labels = eval(f.read())

    with open('./data/merged_samples_test.txt', 'r') as f:
        merged_samples_test = eval(f.read())

    with open('./data/list_ans.txt', 'r') as f:
        list_ans = eval(f.read())

   

    # 6. (fake_model.py)

    data = {
        "sample1": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        "sample2": [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]
    }


    prompts = generate_prompts(merged_samples, list_ans)
    for prompt in prompts:
        print(prompt)

    result = call_fake_model(data)
    print(result)


    #从test_labels中
    test_labels = list(test_labels.values())
    7.：f1, acc, precision, recall
    y_true = test_labels  #test_data['label']
    y_pred = [1, 0, 1, 1, 0, 0]  

    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    print(f"F1 Score: {f1}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


if __name__ == "__main__":
    main()
