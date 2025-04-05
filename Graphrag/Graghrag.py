import os
from pathlib import Path
import pandas as pd
import tiktoken

from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_report_embeddings,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.drift_search.drift_context import (
    DRIFTSearchContextBuilder,
)
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text string using tiktoken
    
    Args:
        text (str): The text to count tokens for
        model (str): The model to use for tokenization (default: "gpt-4")
        
    Returns:
        int: Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        
    tokens = encoding.encode(text)
    return len(tokens)




def init_graphrag_search(
    input_dir: str,
    api_key: str = os.environ.get("OPENAI_API_KEY"),
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    
) -> DRIFTSearch:
    """
    Initialize and return a GraphRAG search instance.
    
    Args:
        api_key (str): OpenAI API key
        llm_model (str): Name of LLM model to use
        embedding_model (str): Name of embedding model to use
        input_dir (str): Directory containing input data files
        
    Returns:
        DRIFTSearch: Configured search instance
    """
    LANCEDB_URI = f"{input_dir}/lancedb"

    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes" 
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    RELATIONSHIP_TABLE = "create_final_relationships"
    COVARIATE_TABLE = "create_final_covariates"
    TEXT_UNIT_TABLE = "create_final_text_units"
    COMMUNITY_LEVEL = 2

    # read nodes table to get community and degree data
    entity_df = pd.read_parquet(f"{input_dir}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/{ENTITY_EMBEDDING_TABLE}.parquet")

    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

    # load description embeddings to an in-memory lancedb vectorstore
    description_embedding_store = LanceDBVectorStore(
        collection_name="default-entity-description",
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)

    full_content_embedding_store = LanceDBVectorStore(
        collection_name="default-community-full_content",
    )
    full_content_embedding_store.connect(db_uri=LANCEDB_URI)

    relationship_df = pd.read_parquet(f"{input_dir}/{RELATIONSHIP_TABLE}.parquet")
    relationships = read_indexer_relationships(relationship_df)

    text_unit_df = pd.read_parquet(f"{input_dir}/{TEXT_UNIT_TABLE}.parquet")
    text_units = read_indexer_text_units(text_unit_df)

    chat_llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    token_encoder = tiktoken.encoding_for_model(llm_model)

    text_embedder = OpenAIEmbedding(
        api_key=api_key,
        api_base=None,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )

    def read_community_reports(
        input_dir: str,
        community_report_table: str = COMMUNITY_REPORT_TABLE,
    ):
        """Embeds the full content of the community reports and saves the DataFrame with embeddings to the output path."""
        input_path = Path(input_dir) / f"{community_report_table}.parquet"
        return pd.read_parquet(input_path)

    report_df = read_community_reports(input_dir)
    reports = read_indexer_reports(
        report_df,
        entity_df,
        COMMUNITY_LEVEL,
        content_embedding_col="full_content_embeddings",
    )
    read_indexer_report_embeddings(reports, full_content_embedding_store)

    drift_params = DRIFTSearchConfig(
        temperature=0,
        max_tokens=6_000,
        primer_folds=1,
        drift_k_followups=0,
        n_depth=1,
        n=1,
    )

    GraphRAG_PROMPT = """
    ---Role---

    You are a thorough assistant responding to questions based on retrieved information.


    ---Goal---

    Provide a clear and accurate response. Carefully review and verify the retrieved data, and integrate any relevant necessary knowledge to comprehensively address the user's question. 
    If you are unsure of the answer, just say so. Do not fabricate information. 
    Do not include details not supported by the provided evidence.


    ---Target response length and format---

    {response_type}


    ---Retrived Context and Query---

    {context_data}


    Add sections and commentary to the response as appropriate for the length and format.

    Additionally provide a score between 0 and 100 representing how well the response addresses the overall research question: {global_query}. Based on your response, suggest up to five follow-up questions that could be asked to further explore the topic as it relates to the overall research question. Do not include scores or follow up questions in the 'response' field of the JSON, add them to the respective 'score' and 'follow_up_queries' keys of the JSON output. Format your response in JSON with the following keys and values:

    {{'response': str, Put your answer, formatted in markdown, here. Do not answer the global query in this section.
    'score': int,
    'follow_up_queries': List[str]}}
    """

    context_builder = DRIFTSearchContextBuilder(
        chat_llm=chat_llm,
        text_embedder=text_embedder,
        entities=entities,
        relationships=relationships,
        reports=reports,
        entity_text_embeddings=description_embedding_store,
        text_units=text_units,
        local_system_prompt=GraphRAG_PROMPT,
        token_encoder=token_encoder,
        config=drift_params,
    )

    search = DRIFTSearch(
        llm=chat_llm, context_builder=context_builder, token_encoder=token_encoder
    )
    
    return search



def init_local_search(input_dir: str, 
                      api_key: str = os.environ.get("OPENAI_API_KEY"),
                      llm_model: str = "gpt-4o-mini",
                      embedding_model: str = "text-embedding-3-small"):
    """Initialize local search with given input directory and API credentials.
    
    Args:
        input_dir: Path to input directory containing parquet files
        api_key: OpenAI API key
        llm_model: Name of LLM model to use
        embedding_model: Name of embedding model to use
        
    Returns:
        LocalSearch: Initialized search engine
    """
    LANCEDB_URI = f"{input_dir}/lancedb"

    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes" 
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    RELATIONSHIP_TABLE = "create_final_relationships"
    TEXT_UNIT_TABLE = "create_final_text_units"
    COMMUNITY_LEVEL = 2

    # read nodes table to get community and degree data
    entity_df = pd.read_parquet(f"{input_dir}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/{ENTITY_EMBEDDING_TABLE}.parquet")

    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

    # load description embeddings to an in-memory lancedb vectorstore
    description_embedding_store = LanceDBVectorStore(
        collection_name="default-entity-description",
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)

    relationship_df = pd.read_parquet(f"{input_dir}/{RELATIONSHIP_TABLE}.parquet")
    relationships = read_indexer_relationships(relationship_df)

    report_df = pd.read_parquet(f"{input_dir}/{COMMUNITY_REPORT_TABLE}.parquet")
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

    text_unit_df = pd.read_parquet(f"{input_dir}/{TEXT_UNIT_TABLE}.parquet")
    text_units = read_indexer_text_units(text_unit_df)

    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")

    text_embedder = OpenAIEmbedding(
        api_key=api_key,
        api_base=None,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=None,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )
    graphrag_prompt = """
    ---Role---

    You are a thorough assistant responding to questions based on retrieved information.


    ---Goal---

    Provide a clear and accurate response. Carefully review and verify the retrieved data, and integrate any relevant necessary knowledge to comprehensively address the user's question. 
    If you are unsure of the answer, just say so. Do not fabricate information. 
    Do not include details not supported by the provided evidence.


    ---Target response length and format---

    {response_type}


    ---Retrived Context and Query---

    {context_data}

    """
    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 9_000,
    }

    llm_params = {
        "max_tokens": 2_000,
        "temperature": 0.0,
    }

    search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        system_prompt=graphrag_prompt,
        context_builder_params=local_context_params,
        response_type="Multiple Paragraphs",
    )
    
    return search_engine



def init_global_search(input_dir: str, 
                       api_key: str = os.environ.get("OPENAI_API_KEY"),
                       llm_model: str = "gpt-4o-mini") -> GlobalSearch:
    """Initialize global search engine with given parameters.
    
    Args:
        input_dir: Directory containing parquet files from indexing pipeline
        api_key: API key for LLM service
        llm_model: Name of LLM model to use
        
    Returns:
        GlobalSearch: Configured search engine instance
    """
    # parquet files generated from indexing pipeline
    COMMUNITY_TABLE = "create_final_communities"
    COMMUNITY_REPORT_TABLE = "create_final_community_reports" 
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"

    # community level in the Leiden community hierarchy from which we will load the community reports
    # higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
    COMMUNITY_LEVEL = 2

    community_df = pd.read_parquet(f"{input_dir}/{COMMUNITY_TABLE}.parquet")
    entity_df = pd.read_parquet(f"{input_dir}/{ENTITY_TABLE}.parquet")
    report_df = pd.read_parquet(f"{input_dir}/{COMMUNITY_REPORT_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/{ENTITY_EMBEDDING_TABLE}.parquet")

    token_encoder = tiktoken.encoding_for_model(llm_model)

    communities = read_indexer_communities(community_df, entity_df, report_df)
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,
        entities=entities,  # default to None if you don't want to use community weights for ranking
        token_encoder=token_encoder,
    )

    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )

    context_builder_params = {
        "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
        "temperature": 0.0,
    }
    REDUCE_SYSTEM_PROMPT = """
    ---Role---

    You are a thorough assistant responding to questions based on retrieved information.


    ---Goal---

    Provide a clear and accurate response. Carefully review and verify the retrieved data, and integrate any relevant necessary knowledge to comprehensively address the user's question. 
    If you are unsure of the answer, just say so. Do not fabricate information. 
    Do not include details not supported by the provided evidence.


    ---Target response length and format---

    {response_type}


    ---Analyst Reports---

    {report_data}
    """

    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=9_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
        json_mode=True,  # set this to False if your LLM model does not support JSON mode.
        context_builder_params=context_builder_params,
        reduce_system_prompt=REDUCE_SYSTEM_PROMPT,
        concurrent_coroutines=1,
        response_type="Multiple Paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )
    
    return search_engine