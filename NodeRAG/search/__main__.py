import argparse
from flask import Flask, request, jsonify
import yaml
from .search import NodeSearch
from ..config import NodeConfig
import os
parser = argparse.ArgumentParser(description='TGRAG search engine')
parser.add_argument('-f','--folder_path', type=str, help='The folder path of the document')
args = parser.parse_args()

config_path = os.path.join(args.folder_path, 'Node_config.yaml')

with open(config_path, 'r') as f:
    args.config = yaml.safe_load(f)

model_config = args.config['model_config']
embedding_config = args.config['embedding_config']


document_config = args.config['config']
path = args.folder_path
url = document_config.get('url','127.0.0.1')
port = document_config.get('port',5000)

Search_engine = NodeSearch(NodeConfig(args.config))
app = Flask(__name__)



@app.route('/answer', methods=['POST'])
def answer():
    question = request.json['question']
    answer = Search_engine.answer(question)
    return jsonify({'answer':answer.response})

@app.route('/answer_retrieval', methods=['POST'])
def answer_retrieval():
    question = request.json['question']
    answer = Search_engine.answer(question)
    return jsonify({'answer':answer.response, 'retrieval':answer.retrieval_info})

@app.route('/retrieval', methods=['POST'])
def search():
    question = request.json['question']
    retrieval = Search_engine.search(question)
    return jsonify({'retrieval':retrieval.retrieval_info})

if __name__ == '__main__':
    app.run(host=url, port=port,debug=False,threaded=True)

