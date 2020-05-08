import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load Data (messages and categories) from specified Database.

    Parameter
    ---------
    messages_filepath: String
        path from where Data  (messages) is loaded
    categories_filepath: String
        path from where Data (categories) is loaded

    Returns
    -------
    df: Pandas.DataFrame
        merged messages and categories DataFrame

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    """
    Clean provided Dataframe by splitting categories in single columns .

    Parameter
    ---------
    df: Pandas.DataFrame
        Dataframe to be cleaned

    Returns
    -------
    df: Pandas.DataFrame
        Cleaned DataFrame

    """
    categories = df['categories'].str.split(pat=";", expand=True)

    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop(columns="categories")
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save dataframe to specific filepath.

    Parameter
    ---------
    df: Pandas.DataFrame
        Dataframe which is stored
    database_filename: String
        path/name to Database where the df is stored in

    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MessageAndCategories', engine, index=False)


def main():
    """
    Main function of process_data.

    Calls each function to first load the raw data cleans it and stores it back to an Database.

    """
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
