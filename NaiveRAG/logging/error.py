from functools import wraps
from .logger import setup_logger
import json
import os

error_logger = setup_logger(__name__,os.path.join(os.getcwd(),'error.log'))

def error_handler(func): 
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return str(e)
    return wrapper
        
def error_handler_async(func): 
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return str(e)
    return wrapper

def cache_error(func): 
    @wraps(func)
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        
        if isinstance(response, list):
            return response
            
        if isinstance(response, str):
            if kwargs.get('cache_path'):
                if "'error':" in response.lower():
                    print(f'Error happened: {response}')
                    error_logger.error(response)
                    
                    meta_data = kwargs.get('meta_data',None)
                        
                    if meta_data is not None:
                        cache_path = kwargs.get('cache_path')
                            
                        input_data = args[1]
                        if input_data is None:
                            input_data = kwargs.get('input',None)
                        if isinstance(input_data,dict):
                            if input_data.get('response_format',None) is not None:
                                input_data.pop('response_format')
                        LLM_store = {'input':input_data,'meta_data':meta_data}
                        with open(cache_path,'a') as f:
                            f.write(json.dumps(LLM_store)+'\n')
                            response = 'Error cached'
                if response == 'Error cached':
                    return response
                else:
                    raise Exception(f'Error happened, please check the error log.')
        return response
            
    return wrapper

def cache_error_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        response = await func(*args, **kwargs)
        if isinstance(response, str):
            if kwargs.get('cache_path'):
                if "'error':" in response.lower():
                    #log errors
                    error_logger.error(response)
                
                
                    meta_data = kwargs.get('meta_data',None)
                            
                    if meta_data is not None:
                        if kwargs.get('cache_path',None) is not None:
                            cache_path = kwargs.get('cache_path')
                            
                            input_data = args[1]
                            if input_data is None:
                                input_data = kwargs.get('input',None)
                            if input_data.get('response_format',None) is not None:
                                input_data.pop('response_format')
                            LLM_store = {'input':input_data,'meta_data':meta_data}
                            with open(cache_path,'a') as f:
                                f.write(json.dumps(LLM_store)+'\n')
                                response = 'Error cached'
                    if response == 'Error cached':
                        return response
                    else:
                        raise Exception(f'Error happened, please check the error log.')
        return response
            
    return wrapper

def clear_cache(path:str) -> None:
    with open(path,'w') as f:
        f.write('')
    return 'cache cleared'