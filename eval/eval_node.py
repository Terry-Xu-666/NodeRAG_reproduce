from NodeRAG import NodeConfig,NodeSearch
import argparse
import yaml
import pandas as pd
from tqdm import tqdm
import time
import tiktoken
from openai import OpenAI

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

def Response(search_engine,questions,path):
    for index, row in tqdm(questions.iterrows(),total=len(questions)):
        question = row["question"]
        start_time = time.time()
        answer = search_engine.answer(question)
        end_time = time.time()
        questions.at[index,"response"] = answer.response
        questions.at[index,"retrieval"] = answer.retrieval_info
        questions.at[index,"time"] = end_time - start_time
        questions.at[index,"tokens"] = count_tokens(answer.retrieval_info)
        questions.to_parquet(path.replace('.parquet', 'node_responsed.parquet'))
    return questions
    
def get_gpt4_response(client,answer, response):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please evaluate if the response matches the reference answer."},
                {"role": "user", "content": f"Instructions\nYou will receive a ground truth answer (referred to as Answer) and a model-generated answer (referred to as Response). Your task is to compare the two and determine whether they align.\n\nNote: The ground truth answer may sometimes be embedded within the model-generated answer. You need to carefully analyze and discern whether they align.\nYour Output:\nIf the two answers align, respond with yes.\nIf they do not align, respond with no.\nIf you are very uncertain, respond with unclear.\nYour response should first include yes, no, or unclear, followed by an explanation.\n\nExample 1\nAnswer: Houston Rockets\nResponse: The basketball player who was drafted 18th overall in 2001 is Jason Collins, who was selected by the Houston Rockets.\nExpeted output: yes\n\nExample 2\nAnswer: no\nResponse: Yes, both Variety and The Advocate are LGBT-interest magazines. The Advocate is explicitly identified as an American LGBT-interest magazine, while Variety, although primarily known for its coverage of the entertainment industry, also addresses topics relevant to the LGBT community.\n Expected output: no\n\nInput Data Format\nGround Truth Answer: {answer}\nModel Generated Answer: {response}\n\nExpected Output\nyes, no, or unclear\nAn explanation of your choice.\n\nOutput:"}
            ],
            temperature=0,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting GPT-4 response: {e}")
        return str(e)
    
def evaluate(responsed_questions,path):
    client = OpenAI()
    for index, row in tqdm(responsed_questions.iterrows(),total=len(responsed_questions)):
        answer = row["answer"]
        response = row["response"]
        evaluation = get_gpt4_response(client,answer,response)
        row["gpt_4o_evaluation"] = evaluation
        responsed_questions.at[index,"gpt4_evaluation"] = evaluation
    responsed_questions.to_parquet(path.replace('.parquet', '_evaluated.parquet'))
    return responsed_questions

def show_results(evaluated_questions):
    evaluated_questions["eval_result"] = evaluated_questions['gpt4_evaluation'].apply(lambda x: 1 if 'yes' in x.lower() else 0)
    # Calculate the percentage of 1s
    percentage_yes = (evaluated_questions['eval_result'].sum() / len(evaluated_questions)) * 100

    print(f"Percentage of evaluations starting with 'yes': {percentage_yes:.2f}%")
    print(f"average time: {evaluated_questions['time'].mean():.2f} seconds")
    print(f"average tokens: {evaluated_questions['tokens'].mean():.2f}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--main_folder", type=str, help="The main folder of documents")
    parser.add_argument("-q","--questions", type=str, help="the questions file")
    parser.add_argument("-a","--answer", default=True, type=bool, help="Whether evaluate the answer")
    args = parser.parse_args()
    
    config = NodeConfig.from_main_folder(args.main_folder)
    
        
    questions = pd.read_parquet(args.questions)
    
    search_engine = NodeSearch(config)
    questions = Response(search_engine,questions,args.questions)
    if args.answer:
        questions = evaluate(questions,args.questions)
        show_results(questions)


    
    




