import string
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import pandas as pd

class TextPreprocessor:
    def __init__(self, title_col, abstract_col, title_abstract_col):
        self.title_col = title_col
        self.abstract_col = abstract_col
        self.title_abstract_col = title_abstract_col

    def removeNonAscii(self, s):
        if not isinstance(s, str):
            s = str(s)  # convert s to string if it's not yet the case
        return "".join(i for i in s if ord(i) < 128)

    def make_lower_case(self, text):
        return text.lower()

    def remove_stop_words(self, text):
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
        return text

    def remove_punctuation(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        text_without_punctuation = " ".join(tokens)
        return text_without_punctuation

    def remove_html(self, text):
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)

    def preprocess_column(self, df):
        df[self.title_abstract_col] = df[self.title_col] + " " + df[self.abstract_col]
        df[self.title_abstract_col] = df[self.title_abstract_col].apply(self.removeNonAscii)
        df[self.title_abstract_col] = df[self.title_abstract_col].apply(self.make_lower_case)
        df[self.title_abstract_col] = df[self.title_abstract_col].apply(self.remove_stop_words)
        df[self.title_abstract_col] = df[self.title_abstract_col].apply(self.remove_punctuation)
        df[self.title_abstract_col] = df[self.title_abstract_col].apply(self.remove_html)
        return df


    def analyze_data(self, df, label_col="label_included"):
        # Prétraiter les colonnes
        df_clean = self.preprocess_column(df)

        # Trier le DataFrame par 'label_included' en ordre décroissant
        df_clean_sorted = df_clean.sort_values(by=label_col, ascending=False)

        # Afficher les premières lignes du DataFrame trié
        # print(df_clean_sorted.head())

        # Afficher le décompte des valeurs dans 'label_included'
        # print("Values_count label : \n")
        # print(df_clean_sorted[label_col].value_counts())

        # Afficher la forme du DataFrame
        # print("Shape : \n")
        # print(df_clean_sorted.shape)

        # Afficher le pourcentage des valeurs dans 'label_included'
        # print("Values_count label (%): \n")
        #  print((df_clean_sorted[label_col].value_counts(normalize=True)) * 100)

        return df_clean_sorted