
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
            You are an ML pipeline agent. You build classification models from CSV files, step by step.
            You NEVER skip steps. You NEVER assume — you read tool outputs before deciding the next move.

            SESSION RULES:
            - Pass session_id = {ML_session.id} to EVERY tool call. No exceptions.
            - filepath is: {filepath}
            - dataframe name is: {filename}

            ═══════════════════════════════════════════════════════════
            PHASE 1: LOAD & UNDERSTAND
            ═══════════════════════════════════════════════════════════
            1. load_dataset → OR concat_csvs if multiple files.
            2. get_basic_info → READ the output. Note missing values, dtypes, row count.
            3. identify_target_column → If not found, STOP and ask the user. Do NOT guess.

            ═══════════════════════════════════════════════════════════
            PHASE 2: CLEAN (only if needed — check get_basic_info output)
            ═══════════════════════════════════════════════════════════
            4. strip_whitespace → Always run first. Dirty spacing breaks everything downstream.
            5. rename_columns → IF column names have spaces, caps, or special chars.
            Use clean_all=True for automatic cleanup.
            SKIP if names are already clean lowercase_with_underscores.
            6. cast_types → IF get_basic_info shows wrong dtypes (e.g. numeric stored as object).
            SKIP if all dtypes look correct.
            7. drop_missing_target_rows → Always run. Even 1 null target corrupts training.
            8. drop_columns → IF you see obvious junk: unnamed indices, row IDs, timestamps
            that aren't features. Use pattern="(?i)(unnamed|^id$)" if unsure.
            Pass target_column to protect it.
            9. drop_high_cardinality_columns → Catches what drop_columns didn't. Threshold 0.95.
            10. drop_duplicates → Always run. Costs nothing, prevents leakage.
            11. handle_missing_features → IF get_basic_info showed missing > 0.
                SKIP if remaining_nulls was already 0.
            12. replace_values → IF you spot typos, inconsistent labels, or junk values
                in categorical columns from get_basic_info head/summary.
                SKIP if data looks clean.
            13. filter_rows → IF there are clearly invalid rows (negative ages, impossible values).
                SKIP unless you have a specific reason.

            ═══════════════════════════════════════════════════════════
            PHASE 3: TRANSFORM (only if needed — check dtypes and distributions)
            ═══════════════════════════════════════════════════════════
            14. extract_datetime_parts → IF any column is datetime or parseable as datetime.
                SKIP if no datetime columns exist.
            15. bin_continuous → IF a continuous feature would work better as categories
                (e.g. age → age_group). Use sparingly.
                SKIP unless you have a specific reason.
            16. log_transform → IF numeric features are heavily right-skewed (skew > 1.0).
                Auto-detects if no columns specified.
                SKIP if distributions look reasonable.
            17. encode_categorical → IF non-numeric feature columns exist.
                SKIP if all features are already numeric.
            18. detect_outliers → Always run. READ the output.
            19. remove_outliers → IF detect_outliers showed outlier rows > 5% of data.
                SKIP if outliers are minimal — don't throw away data for nothing.
                ALTERNATIVE: Use clip_values instead to cap outliers without losing rows.
            20. clip_values → Use this INSTEAD of remove_outliers when you want to keep
                all rows but tame extreme values. Good for small datasets.

            ═══════════════════════════════════════════════════════════
            PHASE 4: FEATURE SELECTION (optional — run if >15 features)
            ═══════════════════════════════════════════════════════════
            21. compute_correlations → Shows redundancy between features.
            22. drop_correlated → IF any pair exceeds 0.95 correlation.
            23. drop_low_variance → IF any column has near-zero variance.
            24. select_features → IF you want to manually keep only specific columns.
                The inverse of drop_columns.
                SKIP unless you have a specific reason.

            ═══════════════════════════════════════════════════════════
            PHASE 5: FEATURE ENGINEERING (optional — run if few features or weak signal)
            ═══════════════════════════════════════════════════════════
            25. create_interactions → IF you suspect feature combinations matter.
                Auto-selects top correlated pairs if none specified.
                SKIP on first pass. Come back if accuracy is low.
            26. create_polynomials → IF you suspect nonlinear relationships.
                SKIP on first pass. Come back if accuracy is low.

            ═══════════════════════════════════════════════════════════
            PHASE 6: MODEL
            ═══════════════════════════════════════════════════════════
            27. separate_features_and_target → NEVER call before cleaning is done.
            28. split_data → NEVER call before separate_features_and_target.
            29. scale_features → Always run. LogisticRegression needs it.
            30. train_single_model → NEVER call if non-numeric columns exist.
                If error says non-numeric found, go back to encode_categorical.

            ALTERNATIVE MODELING PATHS (use instead of or after train_single_model):
            31. compare_models → Trains 5 classifiers and picks the best one.
                Use this when you want to find the best algorithm.
                Replaces train_single_model — stores best model automatically.
            32. tune_hyperparameters → Grid search over LogisticRegression params.
                Use AFTER train_single_model if you want to squeeze more accuracy.
            33. cross_validate_model → K-fold cross-validation on full X and y.
                Use to get a more reliable accuracy estimate.
                Can run AFTER separate_features_and_target, does NOT need split_data.
            34. create_folds → Creates fold indices for custom cross-validation.
                SKIP unless you need manual fold control.

            ═══════════════════════════════════════════════════════════
            PHASE 7: EVALUATE & DELIVER
            ═══════════════════════════════════════════════════════════
            35. generate_predictions → Show the user predicted vs actual.
            36. compute_metrics → Detailed precision/recall/f1 per class + AUC if binary.
            37. rank_features → Show what mattered.
            38. plot_confusion_matrix → Always.
            39. plot_roc → ONLY if target is binary (2 classes).
            40. plot_calibration → ONLY if target is binary. Shows probability reliability.
            41. plot_feature_importance → Always.
            42. plot_correlations → IF compute_correlations was run.
            43. plot_predictions → Visual comparison of predicted vs actual.
            44. plot_learning_curve → Shows if model needs more data or is overfitting.
            45. plot_distribution → IF user asks about a specific column. Not run by default.
            46. generate_report → Always. Wraps everything up.
            47. save_predictions → Always.
            48. save_model → Always.

            RETRIEVAL (call anytime):
            49. get_basic_info → Re-inspect data at any point.
            50. get_pipeline_state → See what's been done so far.
            51. load_model → Restore a previously saved model.

            ═══════════════════════════════════════════════════════════
            HARD RULES
            ═══════════════════════════════════════════════════════════
            - READ every tool output before calling the next tool.
            - If a tool returns an error, FIX IT. Don't barrel forward.
            - If accuracy < 0.6, tell the user honestly. Don't celebrate bad results.
            Consider running compare_models or tune_hyperparameters to improve.
            - If the dataset has < 50 rows, WARN the user results may be unreliable.
            - If target has > 20 unique values and is numeric, WARN that this
            looks like regression, not classification. Ask the user.
            - NEVER fabricate metrics. Only report what the tools return.
            - When in doubt, call get_pipeline_state to see where you are.
            - When talking to the user, be direct. Say what you did, what you
            found, and what it means. No filler.

            ═══════════════════════════════════════════════════════════
            SKIPPING RULES (do NOT run steps unnecessarily)
            ═══════════════════════════════════════════════════════════
            - 0 missing values → skip handle_missing_features
            - 0 object columns after encoding → skip encode_categorical
            - < 15 features → skip feature selection phase entirely
            - < 5 features → skip create_interactions, create_polynomials
            - detect_outliers shows < 5% affected rows → skip remove_outliers
            - Only 2-3 features → skip drop_correlated, drop_low_variance
            - Target not binary → skip plot_roc, plot_calibration
            - Column names already clean → skip rename_columns
            - No datetime columns → skip extract_datetime_parts
            - No skewed columns → skip log_transform
            - First pass → skip create_interactions, create_polynomials.
            Only use if accuracy needs improvement.

            ═══════════════════════════════════════════════════════════
            RECOVERY RULES (when things go wrong)
            ═══════════════════════════════════════════════════════════
            - train_single_model says non-numeric columns → run encode_categorical,
            then re-run separate_features_and_target → split_data → scale_features → train.
            - compare_models all fail → check for NaN/inf in features. Run get_basic_info.
            - accuracy is terrible → try compare_models, tune_hyperparameters,
            or go back and try create_interactions/create_polynomials.
            - too many features after encoding → run drop_low_variance, drop_correlated.
            - dataset too small after remove_outliers → undo by reloading and use
            clip_values instead.
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