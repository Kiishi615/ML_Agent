import inspect
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

import tools
from config import load_config
from logging_setup import setup_logging

EXCLUDED = {"get_df"}

tool_functions = [
    obj for name, obj in inspect.getmembers(tools, inspect.isfunction)
    if name not in EXCLUDED
    and not name.startswith("_")
]
agent_tools = []

for func in tool_functions:
    tool = StructuredTool.from_function(
        func=func,
        name=func.__name__,
        description=func.__doc__ or f"Run {func.__name__}",
    )
    agent_tools.append(tool)

load_dotenv()
AppConfig= load_config()
level = AppConfig.logging.level
log_dir = AppConfig.logging.log_dir
setup_logging(level, log_dir)

model = init_chat_model(model='gpt-5-mini')

checkpoint = InMemorySaver()
config = {'configurable' : {'thread_id' : 1}}

session_id = 1

agent = create_agent(
    model=model,
    system_prompt=(
        f"""
        You're a Machine learning engineer, use the tools available to you to solve the user's problems
            Follow the following rules:
                Rule 1: Always load the dataset first.
                Rule 2: Never run split_data without running separate_features_and_target first.
                Rule 3: Ask the user to confirm the target column before separating.
                Rule 4: Unless specified use session_id = {session_id}


        """
    ),
    checkpointer=checkpoint,
    tools = agent_tools,
)

while True:
    user_input = input('Human: ')
    if user_input.lower() == "quit":
        break
    else:
        response = agent.invoke({
            'messages': [{'role': 'user', 'content': user_input}]
        }, 
        config= config # type: ignore
        )
        print(f"AI: {response['messages'][-1].content}\n") 