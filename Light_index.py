import argparse
import os
from lightrag import LightRAG
from lightrag.llm import gpt_4o_mini_complete

def document_loader(input_path:str):
    documents_path = []
    for file in os.listdir(input_path):
        if file.endswith('.txt') or file.endswith('.md'):
            file_path = os.path.join(input_path, file)
            documents_path.append(file_path)
    raw_content = ""
    for file_path in documents_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_content += file.read()
    return raw_content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--main_folder", type=str, help="The main folder of documents")
    args = parser.parse_args()
    
    WORKING_DIR = args.main_folder
    
    rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete
)
    documents = document_loader(os.path.join(WORKING_DIR,"input"))
    rag.insert(documents)
