from typing import Dict, Any, List
import pandas as pd
import json
import pickle
import os

class storage():
    
    def __init__(self,content:Dict[str,Any]|List[Dict[str,Any]]) -> None:
        self.content = content
        
    def save_json(self,path:str,append=False) -> None:
        if append:
            self.append_json(self.content,path)
        else:
            with open(path,'w') as f:
                json.dump(self.content,f,indent=4)
            
    def append_json(self,content:Dict[str,Any]|List[Dict[str,Any]],path:str) -> None:
        with open(path,'w') as f:
            exist_content = json.load(f)
            if isinstance(exist_content,dict):
                exist_content.update(content)
            elif isinstance(exist_content,list):
                exist_content.append(content)
            json.dump(content,f,indent=4)
            
    def save_parquet(self,path:str,append=False) -> None:
        if append:
            self.append_parquet(self.content,path)
        else:
            if isinstance(self.content,list):
                df = pd.DataFrame(self.content)
            elif isinstance(self.content,dict):
                df = pd.DataFrame(self.content)
            else:
                df =self.content
            df.to_parquet(path)
        
    def append_parquet(self,content,path:str) -> None:
        df = self.load_parquet(path)
        df = pd.concat([df,pd.DataFrame(content)],ignore_index=True)
        df.to_parquet(path)
        
    def save_pickle(self,path:str) -> None:
        with open(path,'wb') as f:
            pickle.dump(self.content,f)
    
    @staticmethod        
    def load_pickle(path:str) -> Any:
        with open(path,'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_parquet(path:str) -> pd.DataFrame:
        return pd.read_parquet(path)
    
    @staticmethod
    def load_json(path:str) -> Dict[str,Any]:
        with open(path) as f:
            return json.load(f)
    
    @staticmethod
    def load_jsonl(path:str) -> List[Dict[str,Any]]:
        with open(path) as f:
            return [json.loads(line) for line in f]
    
    @staticmethod
    def load_csv(path:str) -> pd.DataFrame:
        return pd.read_csv(path)
    
    @staticmethod
    def load_excel(path:str) -> pd.DataFrame:
        return pd.read_excel(path)
    
    @staticmethod
    def load_file(path:str) -> str:
        with open(path) as f:
            return f.read()
        
    @staticmethod
    def load_tsv(path:str) -> pd.DataFrame:
        return pd.read_csv(path,sep='\t')
    
    
    @staticmethod
    def load(path:str) -> str:
        if not os.path.exists(path):
            return None
        if path.endswith('.json'):
            return storage.load_json(path)
        elif path.endswith('.jsonl'):
            return storage.load_jsonl(path)
        elif path.endswith('.parquet'):
            return storage.load_parquet(path)
        elif path.endswith('.pkl'):
            return storage.load_pickle(path)
        elif path.endswith('.md') or path.endswith('.txt'):
            return storage.load_file(path)
        elif path.endswith('.csv'):
            return storage.load_csv(path)
        elif path.endswith('.tsv'):
            return storage.load_tsv(path)
        elif path.endswith('.xlsx'):
            return storage.load_excel(path)
    
    
    
        