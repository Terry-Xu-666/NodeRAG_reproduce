from .text_decomposition import text_decomposition_prompt, text_decomposition_prompt_Chinese
from .json_format import text_decomposition, relationship_reconstraction, High_level_element,decomposed_text
from .translation import translate_prompt
from .relationship_reconstraction import relationship_reconstraction_prompt, relationship_reconstraction_prompt_Chinese
from .attribute_generation_prompt import attribute_generation_prompt, attribute_generation_prompt_Chinese
from .community_summary import community_summary, community_summary_Chinese
from .decompose import decompos_query,decompos_query_Chinese
from .answer import answer_prompt, answer_prompt_Chinese
from ...LLM.LLM_state import get_api_client


API_request = get_api_client()

class prompt_manager():
    
    def __init__(self, language:str):
        self.language = language
        
    @property
    def text_decomposition(self):
        match self.language:
            case 'English':
                return text_decomposition_prompt
            case "Chinese":
                return text_decomposition_prompt_Chinese
            case _:
                return self.translate(text_decomposition_prompt)
    @property
    def relationship_reconstraction(self):
        match self.language:
            case 'English':
                return relationship_reconstraction_prompt
            case "Chinese":
                return relationship_reconstraction_prompt_Chinese
            case _:
                return self.translate(relationship_reconstraction_prompt)
            
    @property
    def attribute_generation(self):
        match self.language:
            case 'English':
                return attribute_generation_prompt
            case "Chinese":
                return attribute_generation_prompt_Chinese
            case _:
                return self.translate(attribute_generation_prompt)
            
    @property
    def community_summary(self):
        match self.language:
            case 'English':
                return community_summary
            case "Chinese":
                return community_summary_Chinese
            case _:
                return self.translate(community_summary)
    @property
    def decompose_query(self):
        match self.language:
            case 'English':
                return decompos_query
            case "Chinese":
                return decompos_query_Chinese
            case _:
                return self.translate(decompos_query)
    @property
    def answer(self):
        match self.language:
            case 'English':
                return answer_prompt
            case "Chinese":
                return answer_prompt_Chinese
            case _:
                return self.translate(answer_prompt)
    
            
    def translate(self,prompt:str):
        prompt = translate_prompt.format(language = self.language, prompt = prompt)
        input_dict = {'prompt':prompt}
        response = API_request.request(input_dict)
        return response
    
    @property
    def text_decomposition_json(self):
        return text_decomposition
    
    @property
    def relationship_reconstraction_json(self):
        return relationship_reconstraction
    
    @property
    def high_level_element_json(self):
        return High_level_element
    
    @property
    def decomposed_text_json(self):
        return decomposed_text
    
    
        
        
        