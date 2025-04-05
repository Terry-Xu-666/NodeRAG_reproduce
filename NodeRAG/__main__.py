import argparse
import requests
import yaml

parser = argparse.ArgumentParser(description='TGRAG search')
parser.add_argument('-q','--question', type=str, help='The question to ask the search engine')
parser.add_argument('-f','--folder', type=str, help='The main folder of the project')
parser.add_argument('-r','--retrieval', action='store_true', help='Whether to return the retrieval')
parser.add_argument('-a','--answer', action='store_true', help='Whether to return the answer')

args = parser.parse_args()

data = {'question':args.question}

with open(args.folder+'/Node_config.yaml', 'r') as f:
    args.config = yaml.safe_load(f)
config = args.config['config']

url = config.get('url','127.0.0.1')
port = config.get('port',5000)

url = f'http://{url}:{port}'

if not args.answer and not args.retrieval:
    response = requests.post(url+'/answer', json=data)
    print(response.json()['answer'])

elif args.answer and not args.retrieval:
    response = requests.post(url+'/answer', json=data)
    print(response.json()['answer'])
    
elif not args.answer and args.retrieval:
    response = requests.post(url+'/retrieval', json=data)
    print(response.json()['retrieval'])

else:
    response = requests.post(url+'/answer_retrieval', json=data)
    print({'answer':response.json()['answer'], 'retrieval':response.json()['retrieval']})


    
