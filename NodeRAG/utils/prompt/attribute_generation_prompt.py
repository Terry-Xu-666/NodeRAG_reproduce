attribute_generation_prompt = """
Generate a concise summary of the given entity, capturing its essential attributes and important relevant relationships. The summary should read like a character sketch in a novel or a product description, providing an engaging yet precise overview. Ensure the output only includes the summary of the entity without any additional explanations or metadata. The length must not exceed 2000 words but can be shorter if the input material is limited. Focus on distilling the most important insights with a smooth narrative flow, highlighting the entity’s core traits and meaningful connections.
Entity: {entity}
Related Semantic Units: {semantic_units}
Related Relationships: {relationships}
"""
attribute_generation_prompt_Chinese = """
生成所给实体的简明总结，涵盖其基本属性和重要相关关系。该总结应像小说中的人物简介或产品描述一样，提供引人入胜且精准的概览。确保输出只包含该实体的总结，不包含任何额外的解释或元数据。字数不得超过2000字，但如果输入材料有限，可以少于2000字。重点在于通过流畅的叙述提炼出最重要的见解，突出实体的核心特征及重要关系。
实体: {entity}
相关语义单元: {semantic_units}
相关关系: {relationships}
"""
