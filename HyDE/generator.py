import time
from typing import TypeAlias,Any

from NaiveRAG.LLM.LLM_state import (
    set_api_client,
    get_api_client,
    set_embedding_client,
    get_embedding_client
)
from NaiveRAG.LLM.LLM_route import API_client

HyDECONFIG: TypeAlias = dict[str, Any]


class HyDEConfig():
    def __init__(self, config:HyDECONFIG):
        self.model_config = config['model_config']
        self.embedding_config = config['embedding_config']
        self.main_folder = config['config']['main_folder']
        self.n = config['config'].get('n',8)
        self.task = config['config'].get('task','web search')
        self.language = config['config'].get('language','en')
        self.top_k = config['config'].get('top_k',10)
        self._client_init()
        self.wait_till_success = config['config'].get('wait_till_success',False)
    
    def _client_init(self):
        set_api_client(API_client(self.model_config))
        set_embedding_client(API_client(self.embedding_config))
        self.client = get_api_client()
        self.embedding_client = get_embedding_client()

    def generate(self, prompt):
        get_results = False
        while not get_results:
            try:
                data = {'query':prompt}
                result = self.client.request(data)
                get_results = True
            except Exception as e:
                if self.wait_till_success:
                    time.sleep(1)
                else:
                    raise e
        return result


