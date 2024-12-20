import numpy as np
from scipy import spatial
import openai
import os
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI



client = AzureOpenAI(
    api_key="",
    api_version="",
    azure_endpoint=""
)

# Function to get embeddings
def generate_embeddings(text):
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002-model")
    return response.data[0].embedding
    
# Read CSV data
df = pd.read_csv("products2.csv")

df.columns = df.columns.str.strip()
df['ProductID'] = df['ProductID'].fillna("Unknown").astype(str)



# Concatenate relevant columns into a single text representation
df['Concat'] = (
    df["ProductID"].astype(str) + " " +
    df["name"].astype(str) + " " +
    df['slug'].astype(str) + " " +
    df['description'].astype(str) + " " +
    df['assets'].astype(str) + " " +
    df['facets'].astype(str) + " " +
    df['optionGroups'].astype(str) +
    df["optionValues"].astype(str) + " " +
    df['sku'].astype(str) + " " +
    df['price'].astype(str) + " " +
    df['taxCategory'].astype(str) + " " +
    df['stockOnHand'].astype(str) + " " +
    df['trackInventory'].astype(str) + " " +
    df['variantAssets'].astype(str) + " " +
    df['variantFacets'].astype(str)
)

# Generate embeddings for the 'Concat' column
df['ada_v2'] = df['Concat'].apply(lambda x: generate_embeddings(x))

# Generate embeddings for the 'ProductID' column
df['ada_v1'] = df['ProductID'].apply(lambda x: generate_embeddings(x))
# df.to_csv('product_data_embedded.csv', index=False)
