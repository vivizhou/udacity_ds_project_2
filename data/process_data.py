import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load data from csv files and merge to a single pandas dataframe
    
    Input:
    messages_filepath filepath to messages csv file
    categories_filepath filepath to categories csv file
    
    Returns:
    df dataframe merging categories and messages
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return(df)

def clean_data(df):
    '''
    clean_data
    Clean the dataframe of merged categories and messages
    
    Input:
    df Combined data with messages and categories
    
    Return:
    df Cleaned data containing both messages and cleaned categories
    '''
    categories = df['categories'].str.split(';', expand = True)
    
    row = categories.iloc[1,:]
    
    category_colnames = row.apply(lambda string: string[0:-2])
    
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda string: string[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        
    df = df.drop(['categories'], axis = 1)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(inplace=True)
    
    #remove rows with value 2 in the related column
    df.loc[df.related == 2,:]
    df.drop(df[df.related == 2].index, inplace = True)

    return(df)

def save_data(df, database_filename):
    '''
    save_data
    Save data to SQLite database
    
    Input:
    df Combined data containing messages and cleaned categories
    database_filename filepath to SQLite destination database
    '''
    path = 'sqlite:///' + database_filename
    engine = create_engine(path)
    df.to_sql('disaster_messages', engine, index=False, if_exists = 'replace')


def main():
    '''
    main
    Main function that processes the data, three steps included:
        Load data from csv files and merge to a single pandas dataframe with "load_data" function
        Clean the dataframe of merged categories and messages with "clean_data" function
        Save data to SQLite database with "save_data" function
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()