import sys
import pickle
import nltk
import re
import pandas as pd

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Load Data from specified Database.

    Parameter
    ---------
    database_filepath: String
        path from where Data is loaded

    Returns
    -------
    X: Pandas.DataFrame
        DataFrame with features for training
    Y: Pandas.DataFrame
        DataFrame with labels for training
    category_names: List
        String with names of all labels for visualization

    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('MessageAndCategories', engine)

    X = df.message.values

    Y_df = df.drop(['message', 'original', 'genre', 'id'], axis=1)
    Y = Y_df.values

    category_names = Y_df.columns.values

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize function using nltk to case normalize, lemmatize, and tokenize text.

    Parameter
    ---------
    text: String
        Text which will be tokenized

    Returns
    -------
    clean_tokens: List
        List with all tokens of the text

    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    ML Pipeline function that process text messages and uses GridSearch to find best paremeters for classifier.

    Returns
    -------
    cv:
        GridSearch element with best parameters for model

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english', tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search - only limited paramter, as the training takes to much time,
    # more testing was done in the jupyter notebooks
    parameters = {
        'clf__estimator__min_samples_split': (2, 20),
        'clf__estimator__criterion': ('gini', 'entropy')
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model for each label.

    Parameter
    ---------
    model: GridSearch Element
        Fitted model with best parameters from GridSearch
    X_test: List
        List with features to be predicted
    Y_test: List
        List with labels, corresponding to X_test
    category_names: List
        String with names of all labels for visualization

    """
    y_pred = model.predict(X_test)

    df_y_pred = pd.DataFrame(data=y_pred, columns=category_names)
    df_y_test = pd.DataFrame(data=Y_test, columns=category_names)

    for column in df_y_test:
        print("------------ {} -------------".format(column))
        print(classification_report(df_y_test[column], df_y_pred[column]))


def save_model(model, model_filepath):
    """
    Save trained and fitted model.

    Parameter
    ---------
    model: GridSearch Element
        Fitted model with best parameters from GridSearch
    model_filepath: String
        filepath where model is stored

    """
    pkl_filename = model_filepath
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Main function of train_classifier.

    Calls each function to first load the Trainingsdata and tokenize it to train and fit a model with it.
    Which lastly will be stored in a .pkl file.

    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
