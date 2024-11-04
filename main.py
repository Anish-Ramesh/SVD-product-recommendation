
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


print("Dataset preview:")
print(df.head(5))
print("Dataset size:", df.size)
print("Number of rows (entries):", len(df))
print("Number of columns:", df.shape[1])
print("Number of unique users:", df['user_id'].nunique())
print("Number of unique items:", df['item_id'].nunique())
print("Average rating across all entries:", df['rating'].mean())
print("Rating distribution:\n", df['rating'].value_counts())

# Visualizations
# 1. Distribution of Ratings
plt.figure(figsize=(8, 4))
sns.countplot(x='rating', data=df, palette="viridis")
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# 2. Distribution of Ratings per User
plt.figure(figsize=(8, 4))
df['user_id'].value_counts().plot(kind='hist', bins=30, color='skyblue')
plt.title('Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.show()

# 3. Distribution of Ratings per Item
plt.figure(figsize=(8, 4))
df['item_id'].value_counts().plot(kind='hist', bins=30, color='salmon')
plt.title('Ratings per Item')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Items')
plt.show()

# Encode user and item IDs for FAISS
df['user_id'] = df['user_id'].astype('category').cat.codes.values
df['item_id'] = df['item_id'].astype('category').cat.codes.values


df['user_avg_rating'] = df.groupby('user_id')['rating'].transform('mean')
df['item_avg_rating'] = df.groupby('item_id')['rating'].transform('mean')


reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


model = joblib.load('svd_model.pkl')  


item_factors = model.qi
d = item_factors.shape[1]
index = faiss.IndexFlatL2(d)
index.add(item_factors)


def similar_product(item_id):
    _, similar_items = index.search(item_factors[item_id:item_id+1], 5)
    return similar_items[0]


def recommend_n_items(user_id):
    user_df = df[df['user_id'] == user_id].sort_values(by='rating', ascending=False).head(5)
    print("Top 5 products for the user, sorted by rating in descending order:")
    print()
    print(user_df)
    print()
    top_5_products = user_df['item_id'].tolist()
    print(f"Highly rated products by user {user_id}:", top_5_products)
    print()
    for i in top_5_products:
        s = similar_product(i)
        print(f"Recommended items for user: {user_id} by product_id: {i}:", s)

li = df['user_id'].to_list()
for i in range(1):
    recommend_n_items(li[i])
    print("="*80)

    
'''
while True:
    i = input("Do you want recommendation of product by user or item? ")

    if i.lower() == "user":
        u = input("Enter user_id to continue:")
        recommend_n_items(u)

    elif i.lower() == "item":
        i = input("Enter item_id to continue:")
        similar_product(i)

    else:
        break

    f = input("Do you want to continue searching? (yes/no) ")

    if f.lower() != "yes":
        break
'''