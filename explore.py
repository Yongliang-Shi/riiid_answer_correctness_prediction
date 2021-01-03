# To ignore the warnings
import warnings
warnings.filterwarnings("ignore")

# Import the libraries to handle the np array and pandas dataframe
import numpy as np
import pandas as pd

import psutil
from tqdm.notebook import tqdm

def compute_user_ques_acc(df_train): 
    
    # Compute the accuracy when a user answer an question
    user_ques_accuracy = df_train.groupby(['user_id', 'content_id']).answered_correctly.mean().round(2).to_frame().reset_index()

    # Rename the 3rd column
    user_ques_accuracy.rename(columns={'answered_correctly': 'user_ques_acc'}, inplace=True)
    
    # Cast the 3rd column to float16
    user_ques_accuracy['user_ques_acc'] = user_ques_accuracy['user_ques_acc'].astype("float16")

    return user_ques_accuracy

def compute_user_bundle_acc(df_train):
    
    # Compute the accuracy when a user answer an question
    user_bundle_accuracy = df_train.groupby(['user_id', 'bundle_id']).answered_correctly.mean().round(2).to_frame().reset_index()

    # Rename the 3rd column
    user_bundle_accuracy.rename(columns={'answered_correctly': 'user_bundle_acc'}, inplace=True)
    
    # Cast the 3rd column to float16
    user_bundle_accuracy['user_bundle_acc'] = user_bundle_accuracy['user_bundle_acc'].astype("float16")

    return user_bundle_accuracy

def compute_user_tags_acc(df_train): 

    # Compute the accuracy when a user answer an question
    user_tags_accuracy = df_train.groupby(['user_id', 'tags']).answered_correctly.mean().round(2).to_frame().reset_index()

    # Rename the 3rd column
    user_tags_accuracy.rename(columns={'answered_correctly': 'user_tags_acc'}, inplace=True)
    
    # Cast the 3rd column to float16
    user_tags_accuracy['user_tags_acc'] = user_tags_accuracy['user_tags_acc'].astype("float16")

    return user_tags_accuracy

def compute_user_part_acc(df_train):
    
    # Compute the accuracy when a user answer an question
    user_part_accuracy = df_train.groupby(['user_id', 'part']).answered_correctly.mean().round(2).to_frame().reset_index()

    # Rename the 3rd column
    user_part_accuracy.rename(columns={'answered_correctly': 'user_part_acc'}, inplace=True)
    
    # Cast the 3rd column to float16
    user_part_accuracy['user_part_acc'] = user_part_accuracy['user_part_acc'].astype("float16")

    return user_part_accuracy

def compute_user_tagcount_acc(df_train): 

    # Compute the accuracy when a user answer an question
    user_tagcount_accuracy = df_train.groupby(['user_id', 'tag_count']).answered_correctly.mean().round(2).to_frame().reset_index()

    # Rename the 3rd column
    user_tagcount_accuracy.rename(columns={'answered_correctly': 'user_tagcount_acc'}, inplace=True)
    
    # Cast the 3rd column to float16
    user_tagcount_accuracy['user_tagcount_acc'] = user_tagcount_accuracy['user_tagcount_acc'].astype("float16")

    return user_tagcount_accuracy

def compute_ques_acc(df_train):

    # Compute the overall accuracy for each question
    ques_accuracy = df_train.groupby('content_id').answered_correctly.mean().round(2).to_frame()

    # Rename the column name
    ques_accuracy.rename(columns={'answered_correctly': 'ques_acc'}, inplace=True)
    
    # Cast the column to float16
    ques_accuracy['ques_acc'] = ques_accuracy['ques_acc'].astype("float16")

    return ques_accuracy


def merge_new_features(df_train):

    user_ques_accuracy = compute_user_ques_acc(df_train)
    user_bundle_accuracy = compute_user_bundle_acc(df_train)
    user_tags_accuracy = compute_user_tags_acc(df_train)
    user_part_accuracy = compute_user_part_acc(df_train)
    user_tagcount_accuracy = compute_user_tagcount_acc(df_train)
    ques_accuracy = compute_ques_acc(df_train)

    # Concat the user_ques_acc to the df
    df = df_train.merge(user_ques_accuracy, how='left', left_on = ['user_id', 'content_id'], right_on = ['user_id', 'content_id'])
    # Concat the user_bundle_acc to the df
    df = df.merge(user_bundle_accuracy, how='left', left_on = ['user_id', 'bundle_id'], right_on = ['user_id', 'bundle_id'])
    # Concat the user_tags_acc to the df
    df = df.merge(user_tags_accuracy, how='left', left_on = ['user_id', 'tags'], right_on = ['user_id', 'tags'])
    # Concat the user_part_acc to the df
    df = df.merge(user_part_accuracy, how='left', left_on = ['user_id', 'part'], right_on = ['user_id', 'part'])
    # Concat the user_tags_acc to the df
    df = df.merge(user_tagcount_accuracy, how='left', left_on = ['user_id', 'tag_count'], right_on = ['user_id', 'tag_count'])
    # Concat the user_tags_acc to the df
    df = df.merge(ques_accuracy, how='left', on='content_id')

    return df