decompos_query = '''
Please break down the following query into a single list. Each item in the list should either be a main entity (such as a key noun or object). If you have high confidence about the user's intent or domain knowledge, you may also include closely related terms. If uncertain, please only extract entities and semantic chunks directly from the query. Please try to reduce the number of common nouns in the list. Ensure all elements are organized within one unified list.
Query:{query}
'''

decompos_query_Chinese = '''
请将以下问题分解为一个 list，其中每一项是句子的主要实体（如关键名词或对象）。如果你对用户的意图或相关领域知识有充分把握，也可以包含密切相关的术语。如果不确定，请仅从问题中提取实体。请尽量减少囊括常见的名词，请将这些元素整合在一个单一的 list 中输出。
问题:{query}
'''