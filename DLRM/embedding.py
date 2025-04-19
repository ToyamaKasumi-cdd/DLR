from openai import OpenAI
import torch
import json
import os

def read_relations_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        relations = [line.strip().split('\t')[0] for line in file]
    return relations

def replace_underscore_with_space(input_string): 
    return input_string.replace("_", " ").replace("<", "").replace(">", "")

def get_embeddings(relations):
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # API Key
        base_url="base_url",  # DashScope base_url, for example "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    embeddings = []
    for relation in relations:
        relation = replace_underscore_with_space(relation)
        print(relation)
        completion = client.embeddings.create(
            model="MODEL_NAME", #select your model
            input=[relation],
            dimensions=512,
            encoding_format="float"
        )
        message_data = json.loads(completion.model_dump_json())
        embedding_data = message_data["data"][0]["embedding"]
        embeddings.append(embedding_data)
        embedding_tensor=torch.tensor(embeddings)
    print("Shape:", embedding_tensor.shape)
    print("Size:", embedding_tensor.size())
    return embedding_tensor

def save_embeddings(embeddings, output_file):
    torch.save(embeddings , output_file)

if __name__ == '__main__':
    file_path = r'\data\ICEWS14\all_relation.txt'  # txt
    output_file = r'\data\ICEWS14\relation&inverse_relation_tensor.pt'  # pt embedding
    
    relations = read_relations_from_file(file_path)
    embeddings = get_embeddings(relations)
    save_embeddings(embeddings, output_file)