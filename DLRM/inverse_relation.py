import os
from openai import OpenAI

def get_inverse_relation(relation):
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),  # API Key
        base_url="base_url",  # DashScope base_url, for example "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    completion = client.chat.completions.create(
        model="qwen-max-latest",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': f'A triple consists of a subject phrase, relation phrase, and object phrase. Please provide an inverse relation such that when the subject and object are swapped, the meaning of the triple remains unchanged for the word “{relation}”. The answer should not include any explanation. '}
        ]
    )
    return completion.choices[0].message.content.strip()

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    inverse_relations = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            relation ,_ = parts
            inverse_relation = get_inverse_relation(relation.replace("<", "").replace(">", ""))
            print(relation,inverse_relation)
            inverse_relations.append(inverse_relation)
    
    with open(output_file, 'a') as outfile:
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation ,_ = parts
                relation=relation.replace("<", "").replace(">", "")
            outfile.write(f"{relation}\n")
        for inverse_relation in inverse_relations:
            outfile.write(f"{inverse_relation}\n")

if __name__ == '__main__':
    input_file = r'\data\ICEWS14\relation2id.txt'
    output_file = r'\data\ICEWS14\all_relation.txt'
    process_file(input_file, output_file)