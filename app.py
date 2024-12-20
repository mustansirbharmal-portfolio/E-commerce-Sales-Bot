import openai  # Ensure you have installed the `openai` package
from flask import Flask, render_template, jsonify, request, session
import pandas as pd
from openai import AzureOpenAI
from datetime import datetime
import tiktoken
import re
from scipy.spatial import distance
from flask_session import Session
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

client = AzureOpenAI(
    api_key="",
    api_version="",
    azure_endpoint=""
)

# Load CSV data
csv_file = "products2.csv"  # Path to your CSV file
df = pd.read_csv(csv_file)
csv_file2 = "product_data_embedded.csv"
df2 = pd.read_csv(csv_file2)

@app.route('/')
def home():
   session.clear()
   return render_template('index.html')

 
@app.route('/store_product_id', methods=['POST'])
def store_campaign_id():
    data = request.get_json()
    product_id = data.get('product_id')
    print(f"Received campaign ID: {product_id}")
    session['desired_Id'] = product_id
    
    print(f"Stored Product ID in session: {session['desired_Id']}")
    result= "Product ID stored successfully"
    return '', 204  # No content returned, just an acknowledgment of success


@app.route('/ask', methods=['POST'])
def ask_route():
    data = request.get_json()       # Gets the JSON data sent in the request body.
    user_query = data.get('query')  # Extracts the query key from the JSON data.

    product_id = session.get('desired_Id')
    print(f"Stored product ID in session: {product_id}")

    print(product_id)
    print(type(product_id))

    response_message = ask(user_query, product_id,df2, token_budget=4096 - 100, print_message=False)
    return jsonify({"response": response_message})


# Cleans text by removing HTML tags and extra whitespace.
def clean_text(text):
    cleaned_text = re.sub(r'<.*?>', '', text)
    cleaned_text = re.sub(r'[\t\r\n]+', '', cleaned_text)
    return cleaned_text

def generate_embeddings(text, model="text-embedding-3-large-model"):
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(a, b):
    # np.dot(a, b): Computes the dot product between the two vectors.
    # np.linalg.norm(a): Computes the Euclidean norm (magnitude) of vector a.
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


import ast  # for converting embeddings saved as strings back to arrays
import numpy as np

def strings_ranked_by_relatedness(query: str, df2: pd.DataFrame, product_id: str, top_n: int = 100):
    df_filtered = df2[df2['ProductID'] == product_id]

    # Get the embedding for the query
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002-model",
        input=query
    ).data[0].embedding # The .data[0].embedding extracts the embedding from the API response 
                        # (which is typically a list of embeddings for a batch of inputs, so we access the first element in this list).
    
    # Calculate relatedness for each row in the DataFrame
    results = []
    for _, row in df_filtered.iterrows():
        text = row["Concat"]
        embedding_str = row["ada_v2"]

        # Convert the string representation of the embedding back to a NumPy array
        # This is necessary because embeddings are typically stored as strings in CSV or databases and 
        # need to be converted back to arrays for calculations.
        embedding = np.array(ast.literal_eval(embedding_str))
        
        # Calculate cosine similarity
        relatedness = 1 - distance.cosine(query_embedding, embedding)
        results.append((text, relatedness))
    
    # Sort the results by relatedness in descending order
    # The reverse=True ensures that the most related texts come first.
    # e.g: results = [("Text1", 0.9), ("Text2", 0.7), ("Text3", 0.95)]
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Extract the strings and their relatedness
    strings = [item[0] for item in results]
    relatednesses = [item[1] for item in results]
    
    return strings, relatednesses


#  Returns the number of tokens in a string based on the model being used (e.g., GPT-4).
def num_tokens(text: str, model: str = "gpt-4") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df2: pd.DataFrame,
    product_id: str,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, _ = strings_ranked_by_relatedness(query, df2, product_id)
    introduction = 'You are a customer assistant that answers questions or give information about text entered by the user from the given data. The Characters before the fisrt space are the Campaign Ids.'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nConcat:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    product_id: str,
    df2: pd.DataFrame = df,
    model: str = "gpt-4",
    token_budget: int = 4096 - 100,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df2, product_id, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You are a customer assistant that answers questions based on the given product data."},
        {"role": "user", "content": message}, 
    ]
    response = client.chat.completions.create(
        model="gpt-4",  # Directly pass model here instead of in query_message
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content.strip()

     # Split the response into meaningful paragraphs
    formatted_response = response_message.split('\n\n')

    return response_message
if __name__ == '__main__':
    app.run(debug=True, port=5002)
