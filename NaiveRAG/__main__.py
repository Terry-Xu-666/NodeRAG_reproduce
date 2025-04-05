import argparse
import requests
import yaml

parser = argparse.ArgumentParser(description='NaiveRAG search')
parser.add_argument('-q','--question', type=str, help='The question to ask the search engine')
parser.add_argument('-r','--retrieval', action='store_true', help='Whether to return the retrieval')
parser.add_argument('-c','--config', default='Nconfig.yaml', type=str, help='The configuration of models, embedding models and document')
args = parser.parse_args()

data = {'question':args.question}
with open(args.config, 'r') as f:
    args.config = yaml.safe_load(f)
document_config = args.config['config']

url = document_config.get('url','127.0.0.1')
port = document_config.get('port',5000)
url = f'http://{url}:{port}'
if args.retrieval:
    response = requests.post(url+'/retrieval', json=data)
    print(response.json()['retrieval'])
else:
    response = requests.post(url+'/answer', json=data)
    print(response.json()['answer'])
    
