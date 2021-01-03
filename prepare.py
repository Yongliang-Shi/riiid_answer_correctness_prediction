# To ignore the warnings
import warnings
warnings.filterwarnings("ignore")

# Import the libraries to handle the np array, pandas dataframe, and visualiztion
import numpy as np
import pandas as pd

import psutil
from tqdm.notebook import tqdm


def prepare_train(df_train):
    
    # Drop column content_type_id and user_answer in the df_train
    df_train.drop(columns=['content_type_id', 'user_answer'], inplace=True)

    # Drop the lecture rows
    mask = (df_train.answered_correctly != -1)
    df_train = df_train[mask]

    return df_train

def prepare_question(df_ques):

    # Drop the column `correct_answer` in df_ques
    df_ques.drop(columns=['correct_answer'], inplace=True)

    return df_ques

def merge_train_questions(df_train, df_ques):

    df_train = prepare_train(df_train)
    df_ques = prepare_question(df_ques)

    # Merge the users' history with the df_ques
    df_train = df_train.merge(df_ques, how='left', left_on='content_id', right_on='question_id')
    
    return df_train
