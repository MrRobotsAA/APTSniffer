import pandas as pd
import networkx as nx
from collections import defaultdict, Counter


def build_label_distribution(df, col1, col2=None):
    label_distribution = defaultdict(Counter)
    if col2:
        col_combined = df[col1].astype(str) + '+' + df[col2].astype(str)
        for value in col_combined.unique():
            indices = df[col_combined == value].index
            labels = df.loc[indices, 'label']
            label_distribution[value].update(labels)
    else:
        for value in df[col1].unique():
            indices = df[df[col1] == value].index
            labels = df.loc[indices, 'label']
            label_distribution[value].update(labels)
    return label_distribution


import pandas as pd
import networkx as nx
from collections import defaultdict, deque
from collections import Counter
from collections import defaultdict, Counter

import pandas as pd
import networkx as nx
from collections import defaultdict, Counter

def build_graph_and_distributions(df):
    G = nx.Graph()
    value_node_map = defaultdict(dict)
    label_distribution = defaultdict(lambda: defaultdict(Counter))

   
    relations = [
        ('ja4', 'ja4s'),
        ('ja4', 'application_category_name'),
        ('requested_server_name', None)
    ]

    for cols in relations:
        if cols[1]:  
            combined_col = df[cols[0]].astype(str) + '&' + df[cols[1]].astype(str)
            column_name = f"{cols[0]}&{cols[1]}"
        else:  
            combined_col = df[cols[0]].astype(str)
            column_name = cols[0]

        
        for value in combined_col.unique():
            if 'nan' not in value.split('&'): 
                center_node = f"{column_name}_{value}"
                indices = df[combined_col == value].index
                G.add_node(center_node)
                value_node_map[column_name][value] = center_node
                for index in indices:
                    G.add_edge(center_node, index)
                    label = df.at[index, 'label']
                    label_distribution[column_name][value][label] += 1

    return G, value_node_map, label_distribution


def bfs_search(G, start_node, max_depth=3):
    visited = set()
    queue = deque([(start_node, 0)])
    results = set()

    while queue:
        current_node, depth = queue.popleft()
        if depth > max_depth:
            break
        results.add(current_node)
        visited.add(current_node)

        for neighbor in G.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return results

from collections import Counter

def query_graph(G, value_node_map, label_distribution, query_sample, df):
    label_counter = Counter()
    relevant_nodes = set()

    query_relations = {
        'ja4&ja4s': str(query_sample['ja4']) + '&' + str(query_sample['ja4s']),
        'ja4&application_category_name': str(query_sample['ja4']) + '&' + str(query_sample['application_category_name']),
        'requested_server_name': str(query_sample['requested_server_name'])
    }


    for rel, value in query_sample.items():
        if value in value_node_map[rel]:
            center_node = value_node_map[rel][value]
            related_nodes = bfs_search(G, center_node)
            relevant_nodes.update(related_nodes)

    for node in relevant_nodes:
        if isinstance(node, str) and (node.startswith('ja4&') or node.startswith('requested_server_name')):
         
            rel, value = node.split('_', 1)  
            if value in label_distribution[rel]:
                for label, count in label_distribution[rel][value].items():
                    label_counter[label] += count
        elif node in df.index:
           
            label = df.at[node, 'label']
            label_counter[label] += 1

    return label_counter


import networkx as nx
from collections import Counter

def complete_tags_old(all_tags, tags_count):
    for tag in all_tags:
        if tag not in tags_count:
            tags_count[tag] = 0
    return tags_count

def complete_tags_graph(all_tags, tags_count):
    
    tags_count_dict = {str(tag): tags_count.get(tag, 0) for tag in all_tags}
    return tags_count_dict



if __name__ == '__main__':
  
    df = pd.read_csv('LLm_CSV_all_features_fix.csv')
    
    df['server_ip'] = df.apply(lambda row: row['src_ip'] if row['sport'] < row['dport'] else row['dst_ip'], axis=1)
   
    G, value_node_map, label_distribution = build_graph_and_distributions(df)
    

    print("G:",G)
  
    query_sample = {
        'server_ip': '106.75.165.78',
        'payload_lengths': '[10, 10, 20]',
        'payload_timestamps': '[100, 100, 200]',
        'ja4': 'abcd',
        'ja4s': 'efgh',
        'requested_server_name': 'download.microsoft.com',
        'application_category_name': 'category1',
    }
   
    label_counts = query_graph(G, value_node_map, label_distribution, query_sample, df)
    print("label_counts:", label_counts)
    
    all_tags = set(df['label'])
    all_tags.add('0')  
    completed_tag_counts = complete_tags_graph(all_tags, label_counts)
    print("completed_tag_counts:", completed_tag_counts)

    