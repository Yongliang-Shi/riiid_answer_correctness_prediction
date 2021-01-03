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
    df_train.drop(columns=['row_id', 'timestamp', 'content_type_id', 'user_answer'], inplace=True)

    # Drop the lecture rows
    mask = (df_train.answered_correctly != -1)
    train = df_train[mask]

    return train

def prepare_question(df_ques):

    # Drop the column `correct_answer` in df_ques
    df_ques.drop(columns=['correct_answer'], inplace=True)

    return df_ques

def prepare_test(df_test):
    '''
    The function takes the example test dataframe acquired from acquire.py and
    prepares it for its merging with df_ques dataframe.
    '''
    df_test.drop(columns=['row_id', 'group_num', 'timestamp', 'content_type_id',
                        'prior_group_answers_correct', 'prior_group_responses'], inplace=True)

    return df_test


def merge_train_questions(df_train, df_ques):

    train = prepare_train(df_train)
    ques = prepare_question(df_ques)

    # Merge the users' history with the df_ques
    train = train.merge(ques, how='left', left_on='content_id', right_on='question_id')
    
    return train


def merge_test_questions(df_test, df_ques):
    '''
    The function takes the cleaned df_test and df_ques and 
    combines them on the content/question_id.
    '''
    test = df_test.merge(df_ques, how='left', left_on='content_id', right_on='question_id')

    return test
