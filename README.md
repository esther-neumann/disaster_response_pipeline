# Disaster Response Pipeline Project

## Table of Contents
1. [Description](https://github.com/esther-neumann/disaster_response_pipeline#description)
2. [Getting Started](https://github.com/esther-neumann/disaster_response_pipeline#getting-started)
   2.1 [Dependencies](https://github.com/esther-neumann/disaster_response_pipeline#dependencies)
   2.2 [Installing](https://github.com/esther-neumann/disaster_response_pipeline#installing)
   2.4 [Executing Program](https://github.com/esther-neumann/disaster_response_pipeline#executing-program)
   2.4 [Additional Material](https://github.com/esther-neumann/disaster_response_pipeline#additional-material)
3. [Acknowledgement](https://github.com/esther-neumann/disaster_response_pipeline#acknowledgements)
 
    
## Description
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The initial dataset contains pre-labelled tweet and messages from real-life disaster. The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
1. Machine Learning Pipeline to train a model able to classify text message in categories
2. Web App to show model results in real time.

## Getting Started

### Dependencies
 - Python 3.5+ (I used Python 3.7)
 - Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn, Pickle
 - Natural Language Process Libraries: NLTK
 - SQLlite Database Libraqries: SQLalchemy
 - Web App and Data Visualization: Flask, Plotly

### Installing:
Clone this GIT repository:

    git clone https://github.com/esther-neumann/disaster-response-pipeline.git

### Executing Program:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

Tipp:
In this case, it is warmly recommended to use a Linux machine to train the model. Using a standard desktop/laptop (4 CPUs, RAM 8Gb or above) it may take several hours to complete.

## Aditional Material 
In the data and models folder you can find two jupyter notebook that will help you understand how the model works step by step:

1. ETL Preparation Notebook: learn everything about the implemented ETL pipeline
2. ML Pipeline Preparation Notebook: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn

## Acknowledgements
- Udacity for providing such a complete Data Science Nanodegree Program
- Figure Eight for providing messages dataset to train my model

