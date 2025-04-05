from abc import ABC,abstractmethod

class Unit_base(ABC):
    
    @property
    @abstractmethod
    def hash_id(self):
        ...
    @property
    @abstractmethod
    def human_readable_id(self):
        ...
        
    def call_action(self,action:str,*args, **kwargs) -> None:
        method = getattr(self,action,None)
        
        if callable(method):
            method(*args, **kwargs)
        else:
            raise ValueError(f"Action {action} not found")
        

