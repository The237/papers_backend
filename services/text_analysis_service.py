import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class TextAnalysisService:
    def __init__(self):
        pass

    def calculate_tf_idf_matrix(self, df, text_column):
        # TF-IDF take only one word
        tf_idf = TfidfVectorizer(ngram_range=(1, 1), min_df=1, stop_words='english')

        # Features extraction based on TF-IDF
        tf_idf_matrix = tf_idf.fit_transform(df[text_column])

        # Reset indexes to align them with tf-idf matrix
        df.reset_index(drop=True, inplace=True)

        return tf_idf_matrix, df

    def calculate_cosine_similarity(self, tf_idf_matrix, seeds=1, aggregation='mean',weights=None):
        sg = cosine_similarity(tf_idf_matrix[:seeds, :], tf_idf_matrix)

        if seeds > 1:
            if aggregation == 'mean':
                if weights is not None:
                    sg = np.average(sg, axis=0, weights=weights)
                else:
                    sg = np.mean(sg, axis=0)
            elif aggregation == 'min':
                sg = np.min(sg, axis=0)
            elif aggregation == 'max':
                sg = np.max(sg, axis=0)
            elif aggregation == 'median':
                sg = np.median(sg, axis=0)
            else:
                raise ValueError("Invalid aggregation method. Supported methods are 'mean', 'min', 'max'.")

        return sg.reshape(-1, 1)

    def get_threshold(self, df, relevant_docs, label_col, distance_col, pct=0.95):
        df['cumulative_sum'] = df[label_col].cumsum()
        filtered_df = df[df['cumulative_sum'] >= pct * relevant_docs]
        if not filtered_df.empty:
            return filtered_df.iloc[0][distance_col]
        return None

    def analyze_similarity(self, dataset, seeds, tf_idf_matrix, df_cleaned_sorted, relevant_docs, total_docs,weights,
            label_col='label_included'):
        def number_of_ones(df, n, label_col):
            return df.head(n)[label_col].sum()

        cos_similarities_ = self.calculate_cosine_similarity(tf_idf_matrix, seeds, aggregation="mean",weights=weights)
        df_cleaned_sorted["similarity"] = cos_similarities_
        df_sorted_by_sim = df_cleaned_sorted.sort_values(by="similarity", ascending=False)
        threshold = self.get_threshold(df_sorted_by_sim, relevant_docs, label_col=label_col, distance_col='similarity')

        L = []
        LR_ = []
        R_ = []
        X_R_ = []
        WSS_ = []

        for i in range(50, total_docs, 50):
            LR_.append(number_of_ones(df_sorted_by_sim, i, label_col))
            L.append(i)

        R_ = [(LR_i / relevant_docs) * 100 for LR_i in LR_]
        X_R_ = [(i / total_docs) * 100 for i in L]
        WSS_ = [(recall - i) for (recall, i) in zip(R_, X_R_)]
        
        # TO Do return the df with similarities adding to metrics
        # return L, LR_, R_, X_R_, WSS_, threshold
        return df_sorted_by_sim

def transform_to_probabilities(df, number_column,probability_col_name):
    # Extraction des nombres à partir du dataframe
    numbers = df[number_column].values

    # normalize  values
    scaler = MinMaxScaler()
    normalized_numbers = scaler.fit_transform(numbers.reshape(-1, 1))

    # apply softmax function
    exp_numbers = np.exp(normalized_numbers)
    probabilities = exp_numbers / np.sum(exp_numbers)

    # add probability col to the dataframe
    df[probability_col_name] = probabilities

    return df