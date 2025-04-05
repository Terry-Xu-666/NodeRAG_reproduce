from functools import wraps

def info_timer(message:str):
    def decorator(func):
        @wraps(func)
        async def wrapper(self,*args,**kwargs):
            self.config.start_timer(f'{message} Started')
            result = await func(self,*args,**kwargs)
            self.config.record_message_with_time(f'{message} Finished')
            return result
        return wrapper
    return decorator
