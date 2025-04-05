from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console


class Observer(ABC):
    @abstractmethod
    def update(self, message: str):
        pass
    @abstractmethod
    def reset(self,total_tasks:int|List[str],desc:str=""):
        pass
    @abstractmethod
    def close(self):
        pass

class ProcessState():
    total_tasks:int = 0
    current_task:int = 0
    desc:str = ""
    
    def __init__(self):
        self.observers = []
        
    def add_observer(self, observer: Observer):
        self.observers.append(observer)
        
    def remove_observer(self, observer: Observer):
        self.observers.remove(observer)
        
    def notify(self):
        for observer in self.observers:
            observer.update(self)
            
    def reset(self,total_tasks:int,desc:str=""):
        self.total_tasks = total_tasks
        self._current_task = 0
        self.desc = desc
        for observer in self.observers:
            observer.reset(total_tasks,desc)
            
    def close(self):
        for observer in self.observers:
            observer.close()
        
    @property
    def current_task(self):
        return self._current_task
    
    @current_task.setter
    def current_task(self,value):
        self._current_task = value
        self.notify()
        

        
class Tracker():
    _instance = None    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self,use_tqdm:bool=True,use_rich:bool=False):
        self.process_state = ProcessState()
        if use_rich:
            self.process_state.add_observer(RichObserver())
        elif use_tqdm:
            self.process_state.add_observer(tqdm_observer()) 
        else:
            raise Exception("No observer selected")
        
    def set(self,total_task:int,desc:str=""):
        self.process_state.reset(total_task,desc)
        
    def update(self):
        self.process_state.current_task += 1
        
    def close(self):
        self.process_state.close()
        
        
class tqdm_observer(Observer):
    def __init__(self):
        self.tqdm_instance = None
        
    def reset(self,total_task:int,desc:str=""):
        if desc == "":
            self.tqdm_instance = tqdm(total=total_task,
                          bar_format="{l_bar}\033[92m{bar}\033[0m| \033[92m{n_fmt}/{total_fmt}\033[0m [\033[92m{elapsed}\033[0m<\033[92m{remaining}\033[0m]", 
                          ascii="░▒▓█",
                          ncols=80)
        else:
            self.tqdm_instance = tqdm(total=total_task,
                          desc='\033[92m' + desc + '\033[0m',  
                          bar_format="{l_bar}\033[92m{bar}\033[0m| \033[92m{n_fmt}/{total_fmt}\033[0m [\033[92m{elapsed}\033[0m<\033[92m{remaining}\033[0m]", 
                          ascii="░▒▓█",
                          ncols=80)
            
    def update(self,process_state:ProcessState):
        self.tqdm_instance.n = process_state.current_task
        self.tqdm_instance.refresh()
        
    def close(self):
        self.tqdm_instance.close()
            
    
class rich_console():
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.console = Console()
        
 
    

class RichObserver(Observer):
    def __init__(self):
        self.progress = None
        self.task = None
        self.console =  rich_console().console
        
    def reset(self, total_task: int, desc: str = "Processing"):
        # 创建带有多个列的进度条
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        self.progress.start()
        # 添加任务
        self.task = self.progress.add_task(desc, total=total_task)
            
    def update(self, process_state: ProcessState):
        if self.progress:
            self.progress.update(self.task, completed=process_state.current_task)
        
    def close(self):
        if self.progress:
            self.progress.stop()
            self.console.print('',end='\r')
