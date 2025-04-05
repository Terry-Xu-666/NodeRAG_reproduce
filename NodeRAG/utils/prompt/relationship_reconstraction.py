relationship_reconstraction_prompt = """
You will be given a string containing tuples representing relationships between entities. The format of these relationships is incorrect and needs to be reconstructed. The correct format should be: 'ENTITY_A,RELATION_TYPE,ENTITY_B', where each tuple contains three elements: two entities and a relationship type. Your task is to reconstruct each relationship in the following format: {{'source': 'ENTITY_A', 'relation': 'RELATION_TYPE', 'target': 'ENTITY_B'}}. Please ensure the output follows this structure, accurately mapping the entities and relationships provided.
Incorrect relationships tuple string:{relationship}
"""

relationship_reconstraction_prompt_Chinese = """
你将获得一个包含实体之间关系的元组字符串。这些关系的格式是错误的，需要被重新构建。正确的格式应为：'实体A,关系类型,实体B'，每个元组应包含三个元素：两个实体和一个关系类型。你的任务是将每个关系重新构建为以下格式：{{'source': '实体A', 'relation': '关系类型', 'target': '实体B'}}。请确保输出遵循此结构，准确映射提供的实体和关系。
错误的关系元组:{relationship}
"""
