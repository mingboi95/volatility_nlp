#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["KERAS_BACKEND"] = "theano"
import keras.backend
keras.backend.set_image_dim_ordering('th') # keras < 2.2.5
#keras.backend.set_image_data_format('channels_last') # keras >= 2.2.5 & keras < 2.3

import keras
print(f"Keras version: {keras.__version__}")
print(f"Keras Backend: {keras.backend.backend()}")

# Libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import cm

import time
from datetime import datetime
from tqdm.auto import tqdm
tqdm.pandas(desc = "Progress")

# NLP
import nltk
from emotion_predictor import EmotionPredictor
from nltk.sentiment import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')

# Stats Models
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests

# Scikit-learn
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Exponentiation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, fbeta_score, make_scorer, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit, GridSearchCV, RandomizedSearchCV


# GBRT
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# This is the pipeline module we need from imblearn for Undersampling
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Feature Importance
import shap 

pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 99)
#pd.set_option('display.float_format', lambda x : f'{x:.3f}')
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2
colors = cm.get_cmap('tab20', 10)

SEED = 4222
np.random.seed(SEED)


def preprocess_tweets(data_dict):
    """ 
    Preprocess tweets.csv
    
    Remove Twitter Handles, URLs, Linebreaks
    Strip # from Hashtags, Trailing White spaces, Unnecessary Columns
    Converts Datetime to Date
    
    Input:
    data_dict:: Dictionary with S&P OHLC in data_dict['stock_data']
    
    Returns
    stock_data_df::pd.DataFrame: S&P Daily Returns, DateIndex and Close col
    
    """
    tweets_df = data_dict['tweets']
    ## Remove Twitter Handles
    tweets_df['text'] = tweets_df['text'].str.replace(r'@[^\s]+ ','')
    # Remove URLs
    tweets_df['text'] = tweets_df['text'].str.replace(r'http\S+','')
    tweets_df['text'] = tweets_df['text'].str.replace(r't.co\S+','')
    # Strip # from hashtags
    tweets_df['text'] = tweets_df['text'].str.replace('#', '')
    # Line breaks
    tweets_df['text'] = tweets_df['text'].str.replace('\n', ' ')
    # Trailing white spaces
    tweets_df['text'] = tweets_df['text'].str.rstrip() 
    # Filter out unnecessary columns
    tweets_df = tweets_df[['created_at', 'text', 'retweet_count', 'favorite_count']]
    # Drop NaNs
    tweets_df = tweets_df.dropna()
    # Convert all counts to numeric
    tweets_df['retweet_count'] = tweets_df['retweet_count'].map(lambda x: x if str(x).isnumeric() else 0)
    tweets_df['favorite_count'] = tweets_df['favorite_count'].map(lambda x: x if str(x).isnumeric() else 0)
    # Set Type of Retweet Count and Favorite Count
    tweets_df[['retweet_count', 'favorite_count']] = tweets_df[['retweet_count', 'favorite_count']].astype('int32')
    # Convert Datetime to Date
    tweets_df['created_at'] = tweets_df['created_at'].map(lambda date: datetime.strptime(date, '%a %b %d %H:%M:%S %z %Y').date())
    return tweets_df

def preprocess_vix(data_dict):
    """ Preprocess VIX OHLC """
    vix_df = data_dict['vix']
    vix_df = vix_df.set_index('Date')['Close']
    return vix_df

def preprocess_stocks_data(data_dict):
    """ 
    Preprocess S&P Data:
        1. Shifts Data by one
        2. Compute Daily Returns
    
    Input:
    data_dict:: Dictionary with S&P OHLC in data_dict['stock_data']
    
    Returns
    stock_data_df::pd.DataFrame: S&P Daily Returns, DateIndex and Close col
    
    """
    stock_data_df = data_dict['stock_data']
    stock_data_df = pd.merge(stock_data_df[['Date']], stock_data_df.shift(1).drop(['Date'], axis=1), left_index=True, right_index=True, suffixes=('', '_lag_1'))
    stock_data_df['Close'] = stock_data_df['Close'].diff()
    stock_data_df = stock_data_df.set_index('Date')[['Close']].dropna()
    return stock_data_df

def apply_sentiments(stock_data_df, tweets_df, verbose=True):
    """ 
    Applies Colneric and Demsar's Bidrectional RNN and VADER for sentiment analysis
    Aggregate Sentiment Features by day, create weighted features
    
    Input:
    stock_data_df::pd.DataFrame: preprocessed S&P
    tweet_df::pd.DataFrame: preprocessed Tweets
    verbose::bool: Print Runtime information
    
    Returns
    feature_list::list: List of features (emotions + sentiments + weighted_emotions + weighted_sentiments)
    merged_data::pd.DataFrame: DataFrame with index of days, columns are sentiment and weighted sentiment features
    
    """
    tweets = tweets_df.copy()
    stock_data = stock_data_df.copy()
    
    # Bidirectional RNN
    start_time = time.time()
    model = EmotionPredictor(classification='ekman', setting='mc') 
    tweet_emotions = model.predict_probabilities(tweets['text'])
    runtime = time.time() - start_time 
    if verbose:
        print(f"Bidrectional RNN sentiments runtime: {runtime:.1f} second(s)")
    
    # VADER
    start_time = time.time()
    emotions = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']
    tweet_emotions.columns = ["Tweet"] + emotions

    tweets = tweets.join(tweet_emotions.drop('Tweet', axis = 1))
    
    sia = SentimentIntensityAnalyzer()
    tweet_sentiments = tweets['text'].map(sia.polarity_scores)

    sentiments = ['neg', 'neu', 'pos', 'compound']
    tweets[sentiments] = tweet_sentiments.apply(pd.Series)
    runtime = time.time() - start_time 
    if verbose:
        print(f"VADER sentiments runtime: {runtime:.1f} second(s)")
        
    # Aggregating Data
    weighted_emotions = ["weighted_" + emotion for emotion in emotions]
    for emotion in emotions:
        tweets["weighted_" + emotion] = tweets[emotion] * (tweets['favorite_count'] + 1)

    weighted_sentiments = ["weighted_" + sentiment for sentiment in sentiments]
    for sentiment in sentiments:
        tweets["weighted_" + sentiment] = tweets[sentiment] * (tweets['favorite_count'] + 1) 

    features = emotions + sentiments + weighted_emotions + weighted_sentiments
    aggregate_dict = {feature:'mean' for feature in features}

    aggregated_tweets = tweets.groupby('created_at').aggregate(aggregate_dict)

    # Merge Data
    merged_data = pd.merge(aggregated_tweets, stock_data, left_index = True, right_index = True)
    
    return features, merged_data

def feature_selection(features, stocks_data_df, merged_data_df, pval_threshold=0.01, maxlags=1, verbose=True):
    """ 
    Applies Granger Causality for Feature Selection.
    
    Feature Selection:
        For each feature in features list, 
        if its is granger causal on Close:
        append feature name + lags
        to the selected features list

    Input:
    features::List of str: List of Features
    stock_data_df::pd.DataFrame: preprocessed S&P
    merged_data_df::pd.DataFrame: merged_data Sentiments DataFrame
    pval_threshold::float: Threshold for p-value to determine significance
    maxlags::int: No. of Max Lags to Run Granger Casuality
    verbose::bool: Print Runtime information
    
    Returns
    selected_features::List: List of selected Features
    shifted_data::pd.DataFrame: DataFrame with index of days, 
    columns are significant sentiment and weighted sentiment features
    """
    stock_data = stocks_data_df.copy()
    shifted_data = merged_data_df.copy()
    
    stock_data = stock_data.dropna()
    # Test Box Plots
    quantile1 = stock_data['Close'].quantile(0.25)
    quantile2 = stock_data['Close'].quantile(0.5)
    quantile3 = stock_data['Close'].quantile(0.75)
    #print(quantile3)
    iqr = quantile3 - quantile1

    # Test method to automatically include all features with high confidence of Granger Causality
    selected_features = []
    shifted_data['Close'] = shifted_data['Close'].map(lambda x: 0 if x > quantile1 - iqr * 1.5 and x < quantile3 + iqr * 1.5 else 1)

    # For each feature in features list, if its is granger causal on Close, append feature name + lags to the selected list
    for feature in features:
        #print(shifted_data[['Close', feature]])
        granger_causality = grangercausalitytests(shifted_data[['Close', feature]], maxlag=maxlags, verbose=False)
        for (i, result) in enumerate(granger_causality.items()):
            p_value = result[1][0]['ssr_ftest'][1]
            if p_value < pval_threshold:
                if verbose:
                    print(f"{feature} affects Close at lag {result[0]}, pvalue = {p_value}")
                selected_features.append({'feature': feature, 'lag': i + 1})

    # Add columns for each selected feature and shift it by number of lags
    for selected_feature in selected_features:
        column_name = selected_feature['feature'] + str(selected_feature['lag'])
        shifted_data[column_name] = shifted_data[selected_feature['feature']].shift(selected_feature['lag'])
    shifted_data = shifted_data.dropna()
    # the selected features for testing, can simply override this.
    selected_features = list(map(lambda f: f['feature'] + str(f['lag']), selected_features))
    all_cols = list(shifted_data.columns)
    selected_features = [x for x in all_cols if any(x in y for y in selected_features)]
    if verbose:
        print(selected_features)
    return selected_features, shifted_data

def model_selection(selected_features, val_time, test_time, shifted_data, vix_data, verbose=True):
    """ Hyperparameter tuning for best model 
    
    Input:
    selected_features::List: List of strings of selected feature from Feature Selection
    val_time::pd.Timestamp or similar to split for Validation
    test_time::pd.Timestamp or similar to split for Test
    shift_data::pd.DataFrame: Feature Selection DataFrame
    vix_data::pd.DataFrame:: preprocessed VIX data
    
    Returns
    estimators::estimator: Sklearn Estimator of the best models by CV
    cross_validation_scores::pd.DataFrame: Cross-Validation Scores, 
    results::DataFrame: Test Results
    
    """
    # Supply a time for prediction   
    val_timestamp = val_time
    test_timestamp = test_time
    model_data = shifted_data.copy()
    vix = vix_data.copy()
    train_data = model_data[model_data.index < test_timestamp]
    test_data = model_data[model_data.index >= test_timestamp]

    train = train_data[selected_features]
    train_label = train_data['Close']
    X_train = train[train.index < val_timestamp].copy()
    y_train = train_label[train_label.index < val_timestamp].copy()
    #print(X_train.shape, y_train.shape)

    val = train_data[selected_features]
    val_label = train_data['Close']
    X_val = val[val.index >= val_timestamp]
    y_val = val_label[val_label.index >= val_timestamp]
    #print(X_val.shape, y_val.shape)

    X_test = test_data[selected_features]
    y_test = test_data['Close']
    #print(X_test.shape, y_test.shape)
    
    # F2-Scorer
    ftwo_scorer = make_scorer(fbeta_score, beta=2, average='weighted')
    # https://stackoverflow.com/questions/48390601/explicitly-specifying-test-train-sets-in-gridsearchcv
    train_indices = np.arange(0,X_train.shape[0]).tolist()
    val_indices = np.arange(X_train.shape[0],train.shape[0]).tolist()
    folds = np.repeat([-1, 0], [len(train_indices), len(val_indices)]) # values 0 or positive are kept for validation
    cv = PredefinedSplit(folds)
    
    # Parameter Grid
    param_grids = {}
    param_grids['Nearest Neighbors'] = {
        'Model__n_neighbors': [3, 5, 10],
        'Model__weights' : ['uniform', 'distance']
    }
    param_grids['Gaussian Process'] = {
        #'Model__kernel': [1.0 * RBF(1.0), Exponentiation(RationalQuadratic(), exponent=2)]
        'Model__kernel': [1.0 * RBF(1.0)]
    }
    param_grids['Decision Tree'] = {
        'Model__max_depth': [1, 2, 8, None],
        'Model__min_samples_split': [1, 2, 10],
        'Model__min_samples_leaf': [1, 5, 10]
    }
    param_grids['Random Forest'] = {
        'Model__n_estimators': [50, 100, 250],
        'Model__min_samples_split': [1, 2, 10],
        'Model__min_samples_leaf': [1, 5, 10],
        'Model__max_depth': [1, 2, 8, None]
    }
    param_grids['Neural Net'] = {
        'Model__hidden_layer_sizes': [(64, 64), (64,)],
        'Model__alpha': [0.0001, 0.001, 0.01, 1]
    }
    param_grids['AdaBoost'] = {
        'Model__n_estimators': [15, 50, 150],
        'Model__learning_rate': [0.01, 0.1, 1]
    }
    param_grids['Naive Bayes'] = {
        'Model__var_smoothing': [1e-9]
    }
    param_grids['QDA'] = {
        'Model__reg_param': [0.0, 0.01, 0.1, 1, 15]
    }
    param_grids['Logistic Regression'] = {
        'Model__C': [0.1, 1.0, 5.0]
    }
    param_grids['LightGBM'] = {
        'Model__learning_rate': [0.01, 0.1, 0.5],
        'Model__n_estimators': [50, 100, 250],
        'Model__lambda_l1': [0.0, 0.01, 0.1],
        'Model__lambda_l2': [0.0, 0.01, 0.1],
        'Model__max_depth': [1, 2, 8, -1]
    }
    param_grids['XGBoost'] = {
        'Model__max_depth': [1, 2, 6, 8],
        'Model__learning_rate': [0.1, 0.3, 0.6]
    }
    
    # Cross-Validation 
    names = ["Nearest Neighbors", "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA", "Logistic Regression", "LightGBM", "XGBoost"]

    classifiers = [
        KNeighborsClassifier(),
        GaussianProcessClassifier(random_state=SEED),
        DecisionTreeClassifier(random_state=SEED),
        RandomForestClassifier(random_state=SEED),
        MLPClassifier(activation='relu', max_iter=150, shuffle=False, random_state=SEED, early_stopping=True, n_iter_no_change=5),
        AdaBoostClassifier(random_state=SEED),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression(penalty='l2', max_iter=1000),
        LGBMClassifier(),
        XGBClassifier(use_label_encoder=False)
    ]

    cv_scores = {}
    best_params = {}
    estimators = {}

    # iterate over classifiers
    for name, classifier in zip(names, classifiers):
        # Define which resampling method and which ML model to use in the pipeline
        resampling = RandomOverSampler(random_state=SEED)
        param_grid = param_grids[name]
        # Define the pipeline and combine sampling method with the model
        pipe = Pipeline([
            ('Scaler', StandardScaler()),
            ('RandomOverSampling', resampling), 
            ('Model', classifier)
        ])
        start_time = time.time()
        clf = GridSearchCV(pipe, param_grid=param_grid, scoring=ftwo_scorer, n_jobs=-1, cv=cv)
        clf.fit(train, train_label)
        estimators[name] = clf.best_estimator_
        best_params[name] = clf.best_params_
        f2_score = clf.best_score_
        cv_scores[name] = [f2_score]

        runtime = time.time() - start_time 
        if verbose:
            print(f"{name}'s Validation F2 Score: {f2_score*100:.2f}%, Runtime: {runtime:.1f} second(s)")
        cross_validation = pd.DataFrame(cv_scores).T
        cross_validation.rename(columns={0: 'F2'}, inplace=True)
        best_estimator = cross_validation.nlargest(1, columns='F2').index.tolist()[0]

    
    # Best Performance
    if verbose:
        print(f"Best Estimator: {best_estimator}, Specifications: {estimators[best_estimator]}")
    final_estimator = estimators[best_estimator]
    
    # Test Performance
    d = {}
    # sentiment only
    name = 'sentiment only'
    final_estimator.fit(train, train_label)
    y_preds = final_estimator.predict(X_test)
    f1 = fbeta_score(y_test, y_preds, beta=1, average='weighted', zero_division=0)
    f2 = fbeta_score(y_test, y_preds, beta=2, average='weighted', zero_division=0)
    acc = accuracy_score(y_test, y_preds)
    d[name] = {
        'f1': f1,
        'f2': f2,
        'acc': acc
    }
    if verbose:
        print(f"Sentiment Features Only: Accuracy {acc*100:.2f}%, F1-Score: {f1*100:.2f}%, F2-Score: {f2*100:.2f}%")
    
    # vix only
    vix_train = vix[vix.index < test_timestamp]
    vix_test = vix[vix.index >= test_timestamp]
    #print(vix_train.shape, vix_test.shape)

    combined_train = pd.merge(vix_train, train, left_index=True, right_index=True)
    combined_test = pd.merge(vix_test, X_test, left_index=True, right_index=True)
    #print(combined_train.shape, combined_test.shape)
    name = 'vix only'
    final_estimator.fit(combined_train['Close'].values.reshape(-1, 1), train_label)
    y_preds = final_estimator.predict(combined_test['Close'].values.reshape(-1, 1))
    f1 = fbeta_score(y_test, y_preds, beta=1, average='weighted', zero_division=0)
    f2 = fbeta_score(y_test, y_preds, beta=2, average='weighted', zero_division=0)
    acc = accuracy_score(y_test, y_preds)
    d[name] = {
        'f1': f1,
        'f2': f2,
        'acc': acc
    }
    if verbose:
        print(f"VIX Only: Accuracy {acc*100:.2f}%, F1-Score: {f1*100:.2f}%, F2-Score: {f2*100:.2f}%")
        
    name = 'combined'
    final_estimator.fit(combined_train, train_label)
    y_preds = final_estimator.predict(combined_test)
    f1 = fbeta_score(y_test, y_preds, beta=1, average='weighted', zero_division=0)
    f2 = fbeta_score(y_test, y_preds, beta=2, average='weighted', zero_division=0)
    acc = accuracy_score(y_test, y_preds)
    d[name] = {
        'f1': f1,
        'f2': f2,
        'acc': acc
    }
    if verbose:
        print(f"Combined VIX + Sentiment: Accuracy {acc*100:.2f}%, F1-Score: {f1*100:.2f}%, F2-Score: {f2*100:.2f}%")
    
    results = pd.DataFrame(d)
    return estimators, cross_validation, results


