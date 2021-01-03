# To ignore the warnings
import warnings
warnings.filterwarnings("ignore")

# Import the libraries to handle the np array and pandas dataframe
import numpy as np
import pandas as pd

import psutil
from tqdm.notebook import tqdm

def compute_user_ques_acc (df_train): 
    
    # Compute the accuracy when a user answer an question
    user_ques_accuracy = df_train.groupby(['user_id', 'content_id']).answered_correctly.mean().round(2).to_frame().reset_index()

    # Rename the 3rd column
    user_ques_accuracy.rename(columns={'answered_correctly': 'user_ques_acc'}, inplace=True)

    return user_ques_accuracy

def compute_user_bundle_acc(df_train):
    
    # Compute the accuracy when a user answer an question
    user_bundle_accuracy = df_train.groupby(['user_id', 'bundle_id']).answered_correctly.mean().round(2).to_frame().reset_index()

    # Rename the 3rd column
    user_bundle_accuracy.rename(columns={'answered_correctly': 'user_bundle_acc'}, inplace=True)

    return user_bundle_accuracy

def compute_user_tags_acc(df_train): 

    # Compute the accuracy when a user answer an question
    user_tags_accuracy = df_train.groupby(['user_id', 'tags']).answered_correctly.mean().round(2).to_frame().reset_index()

    # Rename the 3rd column
    user_tags_accuracy.rename(columns={'answered_correctly': 'user_tags_acc'}, inplace=True)

    return user_tags_accuracy

def compute_user_part_acc(df_train):
    
    # Compute the accuracy when a user answer an question
    user_part_accuracy = df_train.groupby(['user_id', 'part']).answered_correctly.mean().round(2).to_frame().reset_index()

    # Rename the 3rd column
    user_part_accuracy.rename(columns={'answered_correctly': 'user_part_acc'}, inplace=True)

    return user_part_accuracy

def compute_user_tagcount_acc(df_train): 

    # Compute the accuracy when a user answer an question
    user_tagcount_accuracy = df_train.groupby(['user_id', 'tag_count']).answered_correctly.mean().round(2).to_frame().reset_index()

    # Rename the 3rd column
    user_tagcount_accuracy.rename(columns={'answered_correctly': 'user_tagcount_acc'}, inplace=True)

    return user_tagcount_accuracy

def compute_ques_acc(df_train):

    # Compute the overall accuracy for each question
    ques_accuracy = df_train.groupby('content_id').answered_correctly.mean().round(2).to_frame()

    # Rename the column name
    ques_accuracy.rename(columns={'answered_correctly': 'ques_acc'}, inplace=True)

    return ques_accuracy


def merge_new_features(df_train):
    