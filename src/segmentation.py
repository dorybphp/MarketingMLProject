import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from utils import load_and_clean_data

def run_segmentation_pipeline():
    print('EXECUTING SEGMENTATION PIPELINE')
    
    # data preprocessing and engineering
    print("Preprocessing Data...")
    df_clean = load_and_clean_data()

    # separate metadata from features
    X_raw = df_clean.drop(['label', 'weight', 'year'], axis=1).copy()
    w_km = df_clean['weight'].values

    # isolate continuous numerical features for scaling
    numeric_features = [
        'age', 'wage per hour', 'capital gains', 'capital losses', 'dividends from stocks',
        'num persons worked for employer', 'weeks worked in year'
    ]
    # remaining treated as categorical column for one-hot encoding
    categorical_features = [col for col in X_raw.columns if col not in numeric_features]

    # construct parallel processing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ]
    )

    X_scaled_km = preprocessor.fit_transform(X_raw)
    
    os.makedirs('outputs', exist_ok=True)
    
    
    # find optimal number of clusters
    print('Finding Optimal Number of Clusters...')
    # 10% representative sample to compute elbow curve efficiently
    sample_size = int(len(X_scaled_km) * 0.10)
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_scaled_km), size=sample_size, replace=False)

    X_sample = X_scaled_km[sample_indices]
    w_sample = w_km[sample_indices]

    # run elbow loop
    inertia_scores = []
    k_range = range(1, 15)

    for k in k_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        kmeans_test.fit(X_sample, sample_weight=w_sample)
        inertia_scores.append(kmeans_test.inertia_)
    
    # elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia_scores, marker='o', linestyle='-', color='blue')
    plt.title('K-Means Elbow Curve for Segmentation', fontsize=14)
    plt.xlabel('Number of Target Marketing Clusters (K)', fontsize=11)
    plt.ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=11)
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('outputs/kmeans_elbow_plot.png', dpi=300)
    plt.close()
    
    # delta rate-of-change bar chart
    inertia_drops = np.diff(inertia_scores)
    plt.figure(figsize=(8, 4))
    plt.bar(range(2, 15), -inertia_drops, color='blue')
    plt.title('Inertia Drop Per Additional Cluster')
    plt.xlabel('Clustering Step (K)')
    plt.ylabel('Reduction in WCSS')
    plt.xticks(range(2, 15))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('outputs/inertia_drop_plot.png', dpi=300)
    plt.close()
    
    
    # execution with optimal number of clusters (k=5)
    selected_k = 5
    print(f'Execution with Selected Cluster Count K={selected_k}')
    k_means_final = KMeans(n_clusters=selected_k, random_state=42, n_init=10, max_iter=300)

    # fit final model with selected k on entire population
    k_means_final.fit(X_scaled_km, sample_weight=w_km)

    # assign cluster labels to unscaled dataframe
    df_clean['cluster_id'] = k_means_final.labels_

    # calculate population-weighted averages for each cluster
    cluster_profiles = df_clean.groupby('cluster_id')[numeric_features].mean().T
    print('\nWeighted Averages Per Cluster:\n', cluster_profiles.round(2))
    
    # calculate real-world population size of each group
    cluster_sizes = df_clean.groupby('cluster_id')['weight'].sum()
    print('\nActual Population Size Per Cluster:')
    print(cluster_sizes.map('{:,.0f}'.format))
    
    marketing_categories = ['education', 'marital stat', 'class of worker', 'major occupation code', 'sex']
    print('\nDominant Categorical Traits Per Cluster')
    # calculate the most common text lable for each category within each cluster
    for cat in marketing_categories:
        print(f'\nMost Commmon {cat.title()}:')
        for cluster in range(5):
            cluster_data = df_clean[df_clean['cluster_id'] == cluster]
            # find category with highest sum of population weights
            top_category = cluster_data.groupby(cat)['weight'].sum().idxmax()
            print(f'Cluster {cluster}: {top_category}')
            
    # custom age brackets to see the generational split
    age_bins = [0, 17, 29, 54, 100]
    age_labels = ['Youth (0-17)', 'Young Adults (18-29)', 'Peak Career (30-54)', 'Seniors (55+)']

    df_clean['age_group'] = pd.cut(df_clean['age'], bins=age_bins, labels=age_labels)

    # weighted cross-tabulation table
    generational_mix = pd.crosstab(
        index=df_clean['age_group'],
        columns=df_clean['cluster_id'],
        values=df_clean['weight'],
        aggfunc='sum'
    ).fillna(0)
    print("Generational Representation in Each Group")
    print(generational_mix.map('{:,.0f}'.format))


if __name__ == '__main__':
    run_segmentation_pipeline()