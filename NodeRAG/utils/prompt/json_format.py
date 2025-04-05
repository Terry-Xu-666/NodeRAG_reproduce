from pydantic import BaseModel

class semantic_group(BaseModel):
    semantic_unit:str
    entities:list[str]
    relationships:list[str]
    
class text_decomposition(BaseModel):
    Output:list[semantic_group]
    
class relationship_reconstraction(BaseModel):
    source:str
    relationship:str
    target:str
    

    
class elements(BaseModel):
    title:str
    description:str
    
    
class High_level_element(BaseModel):
    high_level_elements:list[elements]
    
class decomposed_text(BaseModel):
    elements:list[str]