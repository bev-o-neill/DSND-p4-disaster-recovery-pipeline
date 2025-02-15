import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load disaster messages and categorisations into a dataframe
   
    Args:
    messages_filepath (str): CSV input containing disaster messages
    categories_filepath (str): CSV input containing the categories of the messages
    
    Returns:
    pandas dataframe of the merged input tables
    
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merge both datasets
    df = messages.merge(categories, on=["id"])
    return df


def clean_data(df):
    """
    Cleans the merged dataset up ready for ML modelling
    
    Args:
    df (pandas dataframe): merged dataset output from load_data() function
    
    Returns:
    df (pandas dataframe): cleaned up dataframe prepared for modelling
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    
    # select the first row of the categories dataframe
    row = categories[0:1]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.applymap(lambda x : x[:-2]).iloc[0,:].tolist()
    
    #Rename columns
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df=df.drop_duplicates()
    
    return df
    


def save_data(df, database_filename):
    """
    Saves data into an SQL database
    
    Args:
    df (pandas dataframe): cleaned data output from clean_data() function
    database_filename (str): fliepath name of location the cleaned dataset is to be stored
    
    Returns:
    None
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')

    
      

def main():
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