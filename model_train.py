# Import libraries
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import faiss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  


df = pd.read_csv(r"C:\Users\spriy\OneDrive\Desktop\Anish Python Projects 1\LDA\ratings_Electronics (1).csv")  
df.columns = ["user_id", "item_id", "rating", "timestamp"]  

df.drop(columns=['timestamp'], inplace=True)

df['user_id'] = df['user_id'].astype('category').cat.codes.values
df['item_id'] = df['item_id'].astype('category').cat.codes.values


df['user_avg_rating'] = df.groupby('user_id')['rating'].transform('mean')
df['item_avg_rating'] = df.groupby('item_id')['rating'].transform('mean')


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


model = SVD(n_factors=100, lr_all=0.005, reg_all=0.02)
model.fit(trainset)


joblib.dump(model, 'svd_model.pkl')  