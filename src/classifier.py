import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, log_loss
from utils import load_and_clean_data

def run_classification_pipeline():
    print('EXECUTING CLASSIFICATION PIPELINE')
    
    # data preprocessing and engineering
    print("Preprocessing Data...")
    df_clean = load_and_clean_data()
    
    # feature engineering
    df_clean['total_financial_flow'] = df_clean['capital gains'] - df_clean['capital losses'] + df_clean['dividends from stocks'] 
    df_clean['life_productivity'] = df_clean['age'] * df_clean['weeks worked in year']
    df_clean['employment_stability'] = df_clean['class of worker'].astype(str) + "_" + df_clean['education'].astype(str) # career category
    # prime earning window where >$50K is statistically concentrated
    df_clean['is_peak_earning_age'] = df_clean['age'].between(35, 55).astype(int)
    
    # separate features and convert target to binary
    y = df_clean['label'].apply(lambda x:1 if '50000+' in str(x) else 0).values 
    weights = df_clean['weight'].values # extract weights
    X = df_clean.drop(['label', 'weight', 'year'], axis=1) # prepare features
    
    # encode categorical features as integers for xgboost
    categorical = X.select_dtypes(include=['object']).columns
    for cat in categorical:
        le = LabelEncoder()
        X[cat] = le.fit_transform(X[cat].astype(str))

    os.makedirs('outputs', exist_ok=True)
    
    
    # generate data visualizations
    print('Generating Data Visualizations...')
    # temp dataframe that matches encoded feature set + target
    X_vis = X.copy()
    X_vis['target'] = y

    # calculate correlation matrix
    corr_matrix = X_vis.corr()
    
    # feature correlation with target
    target_corr = corr_matrix['target'].drop('target').sort_values(ascending=True)

    plt.figure(figsize=(10, 12))
    target_corr.plot(kind='barh', color='skyblue')
    plt.title('Feature Correlation with Target (Income > $50K)', fontsize=14, pad=15)
    plt.xlabel('Pearson Correlation Coefficient($r$)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('outputs/feature_target_correlation.png', dpi=300)
    plt.close()
    
    # correlation matrix
    plt.figure(figsize=(22, 18))

    sns.heatmap(
        corr_matrix, 
        annot=False,
        cmap='coolwarm',
        square=True,
        vmin=-1, vmax=1,
        cbar_kws={'shrink': 0.7, 'label': 'Pearson Correlation Coefficient ($r$)'}
    )

    plt.title('Global Correlation Map Matrix', fontsize=18, pad=25)
    # label rotation and font sixes for readability
    plt.xticks(rotation=90, fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig('outputs/global_correlation_matrix.png', dpi=300)
    plt.close()
    
    
    # split data
    print('Splitting Data...') 
    # 80% train & validation / 20% test
    X_temp, X_test, y_temp, y_test, w_temp, w_test = train_test_split(
        X, y, weights, test_size=0.20, random_state=42, stratify=y
    )

    # take 20% from 80% split earlier for validation (total: 60% train/20% test/20% val)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_temp, y_temp, w_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    
    # grid search for best hyper parameters
    print('Executing Hyperparameter Grid Search Optimization...')
    # scale_pos_weight helps with class imbalance
    ratio = np.sum(y == 0) / np.sum(y == 1)

    # grid search setup
    weighted_logloss = make_scorer(
        log_loss,
        response_method='predict_proba',
        greater_is_better=False,
    )

    param_grid = {
        'max_depth': [6, 8, 10, 12],
        'learning_rate': [0.03, 0.05],
        'n_estimators': [100, 200, 300],
        'colsample_bytree': [0.7, 0.8]
    }

    # base classifier setup
    base_classifier = XGBClassifier(
        scale_pos_weight=ratio,
        subsample=0.8, # prevents rows from forcing overfit
        random_state=42,
        eval_metric=['logloss']
    )

    grid_search = GridSearchCV(
        estimator=base_classifier,
        param_grid=param_grid,
        scoring=weighted_logloss,
        cv=3,
        verbose=1,
        n_jobs=1,
    )

    grid_search.fit(
        X_temp, y_temp, 
        sample_weight=w_temp
    )
    print('Optimal Hyperparameters:', grid_search.best_params_)
    
    
    # grab best settings and initialize deep execution run
    print('Compiling Final Classifier...')
    best_settings = grid_search.best_params_
    best_settings['scale_pos_weight'] = ratio
    best_settings['subsample'] = 0.8
    best_settings['random_state'] = 42
    best_settings['eval_metric'] = ['logloss', 'error']
    best_settings['early_stopping_rounds'] = 50
    best_settings['n_estimators'] = 1000

    classifier = XGBClassifier(**best_settings)

    classifier.fit(
        X_train, y_train,
        sample_weight=w_train, 
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight_eval_set=[w_train, w_val],
        verbose=10
    )
    
    
    # validation & evaluation
    print('\nExecuting Model Evaluation...')
    y_pred = classifier.predict(X_test)
    print('\nPopulation-Weighted Classification Report')
    print(classification_report(y_test, y_pred, sample_weight=w_test)) # use sample_weight in report to reflect population distribution

    # confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred, sample_weight=w_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Predicted <50k', 'Predicted >50k'],
                yticklabels=['Actual <50k', 'Actual >50k'])
    plt.title('Weighted Confusion Matrix (Population Scale)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('outputs/final_weighted_confusion_matrix.png', dpi=300)
    plt.close()

    important_features = pd.Series(classifier.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top 10 Predictive Variables for Income")
    print(important_features.head(10))

    
if __name__ == '__main__':
    run_classification_pipeline()