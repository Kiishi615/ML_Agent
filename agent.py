import functools
import inspect
import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

import tools
from config import load_config
from database import (complete_session, create_or_get_dataset, create_tables,
                      create_version, generate_file_hash, log_event,
                      start_session)
from logging_setup import setup_logging

EXCLUDED = {"get_df","check_state"}

tool_functions = [
    obj for name, obj in inspect.getmembers(tools, inspect.isfunction)
    if name not in EXCLUDED
    and not name.startswith("_")
    and obj.__module__ == "tools"
]
agent_tools = []



def verify_and_inspect(filepath: str) -> dict:
    if not filepath:
        return {"error": "No filepath provided"}
    
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    if not filepath.endswith(".csv"):
        return {"error": "Only CSV files supported"}
    try:
        df= pd.read_csv(filepath)
    except Exception as e:
        return{"error": f"Pandas failed to read file : {str(e)}"}
    
    return {"status": "valid",
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns_json": json.dumps(df.columns.tolist())
    }

load_dotenv()
AppConfig= load_config()
level = AppConfig.logging.level
log_dir = AppConfig.logging.log_dir
log_file = setup_logging(level=level, log_dir=log_dir)
logger = logging.getLogger(__name__)
logger.info("Config loaded successfully")

model = init_chat_model(model='gpt-5-mini')

checkpoint = InMemorySaver()
config = {'configurable' : {'thread_id' : 1}}

#======================================================================= setup ends here
create_tables()
filepath = input("Enter CSV filepath: ")
file_path_report = verify_and_inspect(filepath)
if "error" in file_path_report:
    print(file_path_report["error"])
    exit()


filename = Path(filepath).name
file_hash = generate_file_hash(filepath)

row_count = file_path_report["row_count"]
column_count= file_path_report["column_count"]
columns= file_path_report["columns_json"]
logger.info(f"File verified: {filename}, {row_count} rows, {column_count} columns")

dataset = create_or_get_dataset(filename)
version = create_version(dataset_id=dataset.id, file_hash=file_hash, row_count=row_count, column_count=column_count, columns_json=columns)
ML_session = start_session(version_id=version.id)
logger.info(f"Dataset ID: {dataset.id}, Version ID: {version.id}, Session ID: {ML_session.id}")

def make_logged_tool(func, session_id):
    @functools.wraps(func)  
    def wrapper(**kwargs):
        logger.info(f"Tool called: {func.__name__} | Input: {kwargs}")
        result = func(**kwargs)
        if isinstance(result, dict) and "error" in result:
            logger.warning(f"Tool failed: {func.__name__} | Error: {result['error']}")    
        else:
            logger.info(f"Tool success: {func.__name__}")
        log_event(                          
            session_id= session_id,
            event_type="tool_call",
            content=str(kwargs),
            tool_name=func.__name__,
            result=str(result),
        )
        return result                      
    
    return wrapper

for func in tool_functions:
    logged_func = make_logged_tool(func, ML_session.id)
    tool = StructuredTool.from_function(
        func=logged_func,
        name=func.__name__,
        description=func.__doc__ or f"Run {func.__name__}",
    )
    agent_tools.append(tool)
#======================================================================= setup ends here


agent = create_agent(
    model=model,
    system_prompt=(
        f"""
        You are an ML pipeline agent. When a user asks you to analyze 
        or predict from a dataset, follow this EXACT sequence:

        1. load_dataset (always first)
        2. get_basic_info (check for problems)  
        3. identify_target_column (ask user to confirm)
        4. drop_missing_target_rows (remove target rows that are empty)
        5. drop_high_cardinality_columns (remove ID columns, names, etc)
        6. handle_missing_features (if missing > 0)
        7. encode_categorical (if non-numeric columns exist)
        8. separate_features_and_target
        9. split_data
        10. train_single_model
        11. generate_predictions

        DECISION RULES:
        - filepath is : {filepath}
        - dataframe name is : {filename}
        -  CRITICAL: You must pass session_id = {ML_session.id} to EVERY tool you call.
        - If get_basic_info shows missing values → call handle_missing_values
        - If get_basic_info shows object columns → call encode_categorical  
        - If identify_target_column returns not_found → ASK the user
        - NEVER call split_data before separate_features_and_target
        - NEVER call train_single_model if non-numeric columns exist
        """
    ),
    checkpointer=checkpoint,
    tools = agent_tools,
)

while True:
    user_input = input('Human: ')
    if user_input.lower() == "quit":
        complete_session(ML_session.id)
        logger.info(f"Session {ML_session.id} completed")
        break
    else:
        logger.info(f"User input: {user_input}")
        log_event(                          
            session_id= ML_session.id,
            event_type="message",
            content= user_input,
            
        )
        response = agent.invoke({
            'messages': [{'role': 'user', 'content': user_input}]
        }, 
        config= config # type: ignore
        )
        print(f"AI: {response['messages'][-1].content}\n")      
        logger.info(f"Agent response: {response['messages'][-1].content[:100]}")             