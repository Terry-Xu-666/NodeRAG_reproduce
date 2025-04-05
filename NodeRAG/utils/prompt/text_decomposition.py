text_decomposition_prompt = """
Goal: Given a text, segment it into multiple semantic units, each containing detailed descriptions of specific events or activities. 
Perform the following tasks:
1. Provide a summary for each semantic unit while retaining all crucial details relevant to the original context.
2. Extract all entities directly from the original text of each semantic unit, not from the paraphrased summary. Format each entity name in UPPERCASE. You should extract all entities including times, locations, people, organizations and all kinds of entities.
3. From the entities extracted in Step 2, list all relationships within the semantic unit and the corresponding original context in the form of string seperated by comma : "ENTITY_A, RELATION_TYPE, ENTITY_B". The RELATION_TYPE could be a descriptive sentence, while the entities involved in the relationship must come from the entity names extracted in Step 2. Please make sure the string contains three elements representing two entities and the relationship type.

requirements:
1. Temporal Entities: Represent time entities based on the available details without filling in missing parts. Use specific formats based on what parts of the date or time are mentioned in the text.

Each semantic unit should be represented as a dictionary containing three keys: semantic_unit (a paraphrased summary of each semantic unit), entities (a list of entities extracted directly from the original text of each semantic unit, formatted in UPPERCASE), and relationships (a list of extracted relationship strings that contain three elements, where the relationship type is a descriptive sentence). All these dictionaries should be stored in a list to facilitate management and access.


Example:

Text:  In September 2024, Dr. Emily Roberts traveled to Paris to attend the International Conference on Renewable Energy. During her visit, she explored partnerships with several European companies and presented her latest research on solar panel efficiency improvements. Meanwhile, on the other side of the world, her colleague, Dr. John Miller, was conducting fieldwork in the Amazon Rainforest. He documented several new species and observed the effects of deforestation on the local wildlife. Both scholars' work is essential in their respective fields and contributes significantly to environmental conservation efforts.
Output:
[
  {{
    "semantic_unit": "In September 2024, Dr. Emily Roberts attended the International Conference on Renewable Energy in Paris, where she presented her research on solar panel efficiency improvements and explored partnerships with European companies.",
    "entities": ["DR. EMILY ROBERTS", "2024-09", "PARIS", "INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY", "EUROPEAN COMPANIES", "SOLAR PANEL EFFICIENCY"],
    "relationships": [
      "DR. EMILY ROBERTS, attended, INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY",
      "DR. EMILY ROBERTS, explored partnerships with, EUROPEAN COMPANIES",
      "DR. EMILY ROBERTS, presented research on, SOLAR PANEL EFFICIENCY"
    ]
  }},
  {{
    "semantic_unit": "Dr. John Miller conducted fieldwork in the Amazon Rainforest, documenting several new species and observing the effects of deforestation on local wildlife.",
    "entities": ["DR. JOHN MILLER", "AMAZON RAINFOREST", "NEW SPECIES", "DEFORESTATION", "LOCAL WILDLIFE"],
    "relationships": [
      "DR. JOHN MILLER, conducted fieldwork in, AMAZON RAINFOREST",
      "DR. JOHN MILLER, documented, NEW SPECIES",
      "DR. JOHN MILLER, observed the effects of, DEFORESTATION on LOCAL WILDLIFE"
    ]
  }},
  {{
    "semantic_unit": "The work of both Dr. Emily Roberts and Dr. John Miller is crucial in their respective fields and contributes significantly to environmental conservation efforts.",
    "entities": ["DR. EMILY ROBERTS", "DR. JOHN MILLER", "ENVIRONMENTAL CONSERVATION"],
    "relationships": [
      "DR. EMILY ROBERTS, contributes to, ENVIRONMENTAL CONSERVATION",
      "DR. JOHN MILLER, contributes to, ENVIRONMENTAL CONSERVATION"
    ]
  }}
]


#########
Real_Data:
#########
Text:{text}

"""

text_decomposition_prompt_Chinese = """
目标：给定一个文本，将该文本被划分为多个语义单元，每个单元包含对特定事件或活动的详细描述。 
执行以下任务：
1.为每个语义单元提供总结，同时保留与原始上下文相关的所有关键细节。
2.直接从每个语义单元的原始文本中提取所有实体，而不是从改写的总结中提取。
3.从第2步中提取的实体中列出语义单元内的所有关系,其中关系类型可以是描述性句子。使用格式"ENTITY_A,RELATION_TYPE,ENTITY_B"，请确保字符串中包含三个元素，分别表示两个实体和关系类型。

要求：

时间实体：根据文本中提到的日期或时间的具体部分来表示时间实体，不填补缺失部分。

每个语义单元应以一个字典表示,包含三个键:semantic_unit(每个语义单元的概括性总结)、entities(直接从每个语义单元的原始文本中提取的实体列表,实体名格式为大写)、relationships(描述性句子形式的提取关系字符串三元组列表）。所有这些字典应存储在一个列表中，以便管理和访问。

示例:

文本:2024年9月,艾米莉·罗伯茨博士前往巴黎参加国际可再生能源会议。在她的访问期间,她与几家欧洲公司探讨了合作并介绍了她在提高太阳能板效率方面的最新研究。与此同时,在世界的另一边,她的同事约翰·米勒博士在亚马逊雨林进行实地工作。他记录了几种新物种,并观察了森林砍伐对当地野生动物的影响。两位学者的工作在各自的领域内至关重要,对环境保护工作做出了重大贡献。
输出：
[
  {{
    "semantic_unit": "2024年9月,艾米莉·罗伯茨博士参加了在巴黎举行的国际可再生能源会议，她在会上介绍了她关于太阳能板效率提高的研究并探讨了与欧洲公司的合作。",
    "entities": ["艾米莉·罗伯茨博士", "2024-09", "巴黎", "国际可再生能源会议", "欧洲公司", "太阳能板效率"],
    "relationships": [
      "艾米莉·罗伯茨博士, 参加了, 国际可再生能源会议",
      "艾米莉·罗伯茨博士, 探讨了合作, 欧洲公司",
      "艾米莉·罗伯茨博士, 介绍了研究, 太阳能板效率"
    ]
  }},
  {{
    "semantic_unit": "约翰·米勒博士在亚马逊雨林进行实地工作，记录了几种新物种并观察了森林砍伐对当地野生动物的影响。",
    "entities": ["约翰·米勒博士", "亚马逊雨林", "新物种", "森林砍伐", "当地野生动物"],
    "relationships": [
      "约翰·米勒博士, 在, 亚马逊雨林进行实地工作",
      "约翰·米勒博士, 记录了, 新物种",
      "约翰·米勒博士, 观察了, 森林砍伐对当地野生动物的影响"
    ]
  }},
  {{
    "semantic_unit": "艾米莉·罗伯茨博士和约翰·米勒博士的工作在各自的领域内至关重要，对环境保护工作做出了重大贡献。",
    "entities": ["艾米莉·罗伯茨博士", "约翰·米勒博士", "环境保护"],
    "relationships": [
      "艾米莉·罗伯茨博士, 贡献于, 环境保护",
      "约翰·米勒博士, 贡献于, 环境保护"
    ]
  }}
]

##########
实际数据： 
########## 
文本:{text} 
"""

