import functools
import inspect
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

import tools
from config import load_config
from database import (complete_session, create_or_get_dataset, create_tables,
                      create_version, generate_file_hash, log_event,
                      start_session)
from logging_setup import setup_logging

# ═══════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════

load_dotenv()
AppConfig = load_config()
log_file = setup_logging(AppConfig.logging.level, AppConfig.logging.log_dir)
logger = logging.getLogger(__name__)
logger.info("API starting up")

create_tables()

model = init_chat_model(model="gpt-5-mini")

EXCLUDED = {"get_df", "check_state"}
tool_functions = [
    obj for name, obj in inspect.getmembers(tools, inspect.isfunction)
    if name not in EXCLUDED
    and not name.startswith("_")
    and obj.__module__ == "tools"
]

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ML Intern Agent API")

# ═══════════════════════════════════════════════════════════
# SESSION CACHE
# ═══════════════════════════════════════════════════════════

_SESSIONS = {}  # session_id -> {"agent", "config", "filepath", "filename"}


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
            session_id=session_id,
            event_type="tool_call",
            content=str(kwargs),
            tool_name=func.__name__,
            result=str(result),
        )
        return result
    return wrapper


def build_agent(session_id: int, filepath: str):
    """Build and cache agent for a session."""
    
    agent_tools = []
    for func in tool_functions:
        logged_func = make_logged_tool(func, session_id)
        tool = StructuredTool.from_function(
            func=logged_func,
            name=func.__name__,
            description=func.__doc__ or f"Run {func.__name__}",
        )
        agent_tools.append(tool)

    file_context = f"""- Single file: {filepath}
        - session_id: {session_id}
        - Call load_dataset(filepath="{filepath}", session_id={session_id}) first."""

    agent = create_agent(
        model=model,
        system_prompt=f"""
                            You are an ML pipeline agent. You build machine learning models from CSV files, step by step.

                            SESSION RULES:
                            {file_context}
                            - ALWAYS pass session_id = {session_id} to EVERY tool call.

                            PHASE 1: LOAD & UNDERSTAND
                            1. load_dataset
                            2. get_basic_info
                            3. identify_target_column

                            PHASE 2: CLEAN
                            4. drop_missing_target_rows
                            5. drop_high_cardinality_columns
                            6. handle_missing_features
                            7. encode_categorical

                            PHASE 3: MODEL
                            8. separate_features_and_target
                            9. split_data
                            10. train_single_model
                            11. generate_predictions

                            HARD RULES:
                            - READ every tool output before calling the next tool.
                            - If a tool returns an error, FIX IT.
                            - NEVER call split_data before separate_features_and_target.
                            - NEVER call train_single_model if non-numeric columns exist.
                        """,
        checkpointer=InMemorySaver(),
        tools=agent_tools,
    )

    config = {"configurable": {"thread_id": str(session_id)}}

    return {"agent": agent, "config": config}


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    return {"status": "alive"}


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    save_path = DATA_DIR / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = pd.read_csv(save_path)
        row_count = len(df)
        column_count = len(df.columns)
        columns_json = json.dumps(df.columns.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {str(e)}")

    file_hash = generate_file_hash(str(save_path))
    dataset = create_or_get_dataset(file.filename)
    version = create_version(
        dataset_id=dataset.id,
        file_hash=file_hash,
        row_count=row_count,
        column_count=column_count,
        columns_json=columns_json,
    )
    db_session = start_session(version_id=version.id)

    # Load df into ACTIVE_DATAFRAMES
    if db_session.id not in tools.ACTIVE_DATAFRAMES:
        tools.ACTIVE_DATAFRAMES[db_session.id] = {}
    tools.ACTIVE_DATAFRAMES[db_session.id]["main"] = df.copy()

    # Build and cache agent
    _SESSIONS[db_session.id] = {
        **build_agent(db_session.id, str(save_path)),
        "filepath": str(save_path),
        "filename": file.filename,
    }

    logger.info(f"Session {db_session.id} created for {file.filename}")

    return {
        "session_id": db_session.id,
        "dataset_id": dataset.id,
        "version_id": version.id,
        "filename": file.filename,
        "rows": row_count,
        "columns": column_count,
    }


class ChatRequest(BaseModel):
    session_id: int
    message: str


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    if request.session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")

    session = _SESSIONS[request.session_id]

    logger.info(f"User input: {request.message}")
    log_event(
        session_id=request.session_id,
        event_type="message",
        content=request.message,
    )

    response = session["agent"].invoke(
        {"messages": [{"role": "user", "content": request.message}]},
        config=session["config"],
    )

    reply = response["messages"][-1].content
    logger.info(f"Agent response: {reply[:100]}")

    return {
        "session_id": request.session_id,
        "response": reply,
    }


@app.post("/session/end")
async def end_session(session_id: int):
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    complete_session(session_id)
    del _SESSIONS[session_id]
    logger.info(f"Session {session_id} ended")

    return {"status": "ended", "session_id": session_id}