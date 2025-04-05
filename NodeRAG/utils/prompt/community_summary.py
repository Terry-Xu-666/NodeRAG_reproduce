community_summary = """You will receive a set of text data from the same cluster. Your task is to extract distinct categories of high-level information, such as concepts, themes, relevant theories, potential impacts, and key insights. Each piece of information should include a concise title and a corresponding description, reflecting the unique perspectives within the text cluster.
Please do not attempt to include all possible information; instead, select the elements that have the most significance and diversity in this cluster. Avoid redundant information—if there are highly similar elements, combine them into a single, comprehensive entry. Ensure that the high-level information reflects the varied dimensions within the text, providing a well-rounded overview.
clustered text data:
{content}
"""

community_summary_Chinese = """你将收到来自同一聚类的一组文本数据。你的任务是从文本数据中提取不同类别的高层次信息，例如概念、主题、相关理论、潜在影响和关键见解。每条信息应包含一个简洁的标题和相应的描述，以反映该聚类文本中的独特视角。
请不要试图包含所有可能的信息；相反，选择在该聚类中最具重要性和多样性的元素。避免冗余信息——如果有高度相似的内容，请将它们合并为一个综合条目。确保提取的高层次信息反映文本中的多维度内容，提供全面的概览。
聚类文本数据：
{content}
"""