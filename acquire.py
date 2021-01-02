# To ignore the warnings
import warnings
warnings.filterwarnings("ignore")

# Import the libraries to handle the np array and pandas dataframe
import numpy as np
import pandas as pd

def acquire_history_users_test():
    
    # Define the data types of the training records
    dtypes_train = {
        "row_id": "int64",
        "timestamp": "int64",
        "user_id": "int32",
        "content_id": "int16", 
        "content_type_id": "boolean", 
        "task_container_id": "int16",
        "user_answer": "int8", 
        "answered_correctly": "int8", 
        "prior_question_elapsed_time": "float32", 
        "prior_question_had_explanation": "boolean"    
    }

    # Acquire the history of the users in the test dataset
    df_train = pd.read_csv('history_user_test.csv', index_col=0, dtype=dtypes_train)

    # Define the data types of the dataframe of the questions
    dtypes_question = {
        "question_id": "int16", 
        "bundle_id": "int16",
        "correct_answer": "int8", 
        "part": "int8", 
        "tags": "object", 
        "tag_count": "int8"    
    }

    # Load the questions.csv with tag_counts
    df_ques = pd.read_csv('questions_with_tag_counts.csv', index_col=0, dtype=dtypes_question)

    return df_train, df_ques