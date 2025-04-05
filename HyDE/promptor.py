Answer_prompt = """
---Role---

You are a thorough assistant responding to questions based on retrieved information.


---Goal---

Provide a clear and accurate response. Carefully review and verify the retrieved data, and integrate any relevant necessary knowledge to comprehensively address the user's question. 
If you are unsure of the answer, just say so. Do not fabricate information. 
Do not include details not supported by the provided evidence.


---Target response length and format---

Multiple Paragraphs

---Retrived Context---

{corpus}

---Query---

{question}
"""

WEB_SEARCH = """Please write a passage to answer the question.
Question: {}
Passage:"""


SCIFACT = """Please write a scientific paper passage to support/refute the claim.
Claim: {}
Passage:"""


ARGUANA = """Please write a counter argument for the passage.
Passage: {}
Counter Argument:"""


TREC_COVID = """Please write a scientific paper passage to answer the question.
Question: {}
Passage:"""


FIQA = """Please write a financial article passage to answer the question.
Question: {}
Passage:"""


DBPEDIA_ENTITY = """Please write a passage to answer the question.
Question: {}
Passage:"""


TREC_NEWS = """Please write a news passage about the topic.
Topic: {}
Passage:"""


MR_TYDI = """Please write a passage in {} to answer the question in detail.
Question: {}
Passage:"""

from HyDE.generator import HyDEConfig

class Promptor:
    def __init__(self, config: HyDEConfig):
        self.config = config
        self.task = config.task
        self.language = config.language
    
    def build_prompt(self, query: str):
        if self.task == 'web search':
            return WEB_SEARCH.format(query)
        elif self.task == 'scifact':
            return SCIFACT.format(query)
        elif self.task == 'arguana':
            return ARGUANA.format(query)
        elif self.task == 'trec-covid':
            return TREC_COVID.format(query)
        elif self.task == 'fiqa':
            return FIQA.format(query)
        elif self.task == 'dbpedia-entity':
            return DBPEDIA_ENTITY.format(query)
        elif self.task == 'trec-news':
            return TREC_NEWS.format(query)
        elif self.task == 'mr-tydi':
            return MR_TYDI.format(self.language, query)
        else:
            raise ValueError('Task not supported')
    
    def answer_prompt(self, corpus, question):
        return Answer_prompt.format(corpus=corpus, question=question)
