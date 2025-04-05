from .text_spliter import SemanticTextSplitter
from .genid import genid
from .readable_index import document_index
from .unit import Unit_base
from .text_unit import Text_unit


document_index_counter = document_index()


class document(Unit_base):
    def __init__(self, raw_context:str = None,path:str = None,splitter:SemanticTextSplitter = None):

        self.path = path
        self.raw_context = raw_context
        self._processed_context = False
        self._hash_id = None
        self._human_readable_id = None
        self.text_units = None
        self.text_hash_id = None
        self.text_human_readable_id = None
        self.splitter = splitter

    @property
    def hash_id(self):
        if not self._hash_id:
            self._hash_id = genid([self.raw_context],"sha256")
        return self._hash_id
    
    @property
    def human_readable_id(self):
        if not self._human_readable_id:
            self._human_readable_id = document_index_counter.increment()
        return self._human_readable_id
    
    def split(self) -> None:
        if not self._processed_context:
            self._processed_context = True
            texts = self.splitter.split(self.raw_context)
            self.text_units = [Text_unit(text) for text in texts]
            self.text_hash_id = [text.hash_id for text in self.text_units]
            self.text_human_readable_id = [text.human_readable_id for text in self.text_units]
            
    
        
  
    
    
