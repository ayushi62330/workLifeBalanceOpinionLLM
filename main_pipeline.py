import os
import json
import logging
import tweepy
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from chromadb import Client  # Ensure chromadb is installed via pip
from dotenv import load_dotenv
import boto3

# Load environment variables (e.g., from .env)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# === Twitter API Setup ===
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
twitter_client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN,
                               consumer_key=TWITTER_API_KEY,
                               consumer_secret=TWITTER_API_SECRET,
                               access_token=TWITTER_ACCESS_TOKEN,
                               access_token_secret=TWITTER_ACCESS_TOKEN_SECRET)

# === Embedding & Vector DB Setup ===
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = Client()  # Assumes a local/default configuration

# === Data Ingestion Functions ===
def ingest_tweets(query: str, max_results: int = 50):
    """
    Ingest tweets matching a query using Twitter API v2 Recent Search endpoint.
    Requires TWITTER_BEARER_TOKEN to be set.
    """
    response = twitter_client.search_recent_tweets(query=query, max_results=max_results)
    tweet_texts = []
    if response.data:
        tweet_texts = [tweet.text for tweet in response.data]
    logging.info(f"Ingested {len(tweet_texts)} tweets for query '{query}'.")
    return tweet_texts

def ingest_article(url: str) -> str:
    """Scrape an article from a URL using Requests and BeautifulSoup."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    article_text = "\n".join(p.get_text() for p in paragraphs)
    logging.info(f"Ingested article from {url}.")
    return article_text

# === Embedding & Storage ===
def embed_and_store(texts, collection_name: str):
    """Embed texts with a Sentence Transformer and store them in ChromaDB."""
    embeddings = model.encode(texts)
    collection = chroma_client.get_or_create_collection(name=collection_name)
    # Generate unique IDs for each document, e.g., by using their index
    ids = [str(i) for i in range(len(texts))]
    collection.add(ids=ids, documents=texts, embeddings=embeddings)
    logging.info(f"Stored {len(texts)} documents in collection '{collection_name}'.")


# === Opinion Quantification Functions ===

# (1) Simulated function for local testing
def quantify_opinion_simulated(text: str) -> dict:
    """Simulated quantification function returning fixed scores."""
    return {
        "work_flexibility": 4,
        "burnout_risk": 2,
        "remote_work_appeal": 5,
        "productivity_impact": 4,
        "overall_sentiment": 4
    }

# (2) Amazon Bedrock Integration with Guardrails using create_guardrail



def quantify_opinion_bedrock_with_guardrails(text: str) -> dict:
    """
    Uses Amazon Bedrock to generate opinion scores and then validates the output
    using the built-in Guardrail API (create_guardrail).

    The prompt instructs the model to return a JSON object with exactly these keys:
      - work_flexibility
      - burnout_risk
      - remote_work_appeal
      - productivity_impact
      - overall_sentiment

    Each value must be an integer between 1 (low) and 5 (high).
    """
    # Initialize the Bedrock client with a specified region
    bedrock_client = boto3.client("bedrock-runtime", region_name="ap-south-1")
    
    # Construct the prompt as a single-line string and wrap it in a JSON object
    prompt_text = (
        "Analyze the following text regarding work-life balance. "
        "Return a JSON object with exactly these keys: "
        "\"work_flexibility\", \"burnout_risk\", \"remote_work_appeal\", "
        "\"productivity_impact\", \"overall_sentiment\". "
        "Each value must be an integer between 1 and 5. "
        f"Text: \"{text}\""
    )
    # Wrap the prompt in a JSON object
    body = json.dumps({"prompt": prompt_text})
    
    # Invoke the Bedrock model using correct, lower-case parameter names
    bedrock_response = bedrock_client.invoke_model(
    modelId="meta.llama3-70b-instruct-v1:0",
    body=body.encode("utf-8"),
    contentType="application/json"
    )
    print(bedrock_response)
    raw_response_str = bedrock_response["body"].read().decode("utf-8")
    print(raw_response_str)
    try:
        raw_response = json.loads(raw_response_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from Bedrock: {raw_response_str}") from e
    print(raw_response)
    try:
    # Define guardrail parameters for validation
      guardrail_params = {
        "desired_keys": [
            "work_flexibility",
            "burnout_risk",
            "remote_work_appeal",
            "productivity_impact",
            "overall_sentiment"
        ],
        "value_range": {"min": 1, "max": 5}
    }
    
    # Validate using the built-in Guardrail API
    print(dir(bedrock_client))
    guard_response = bedrock_client.apply_guardrail(
        modelOutput=json.dumps(raw_response),
        guardrailParameters=json.dumps(guardrail_params),
        contentType="application/json"
    )
    guard_response_str = guard_response["Body"].read().decode("utf-8")
    try:
        validated_response = json.loads(guard_response_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from Guardrails API: {guard_response_str}") from e

    return validated_response


# For production, assign the Bedrock-based function:
quantify_opinion = quantify_opinion_bedrock_with_guardrails
# For this demo, we use the simulated function:
#quantify_opinion = quantify_opinion_simulated

# === Main Pipeline Function ===
def pipeline():
    # Step 1: Ingest tweets about "work-life balance"
    query = "work-life balance"
    #tweets = ingest_tweets(query)
    #embed_and_store(tweets, collection_name="tweets_worklife")
    
    # Step 2: Ingest an article (replace URL with a valid one for production)
    article_url = "https://hbr.org/2021/01/work-life-balance-is-a-cycle-not-an-achievement"
    article_text = ingest_article(article_url)
    embed_and_store([article_text], collection_name="articles_worklife")
    
    # Step 3: Quantify opinions for each tweet and for the article
    #tweet_opinions = [quantify_opinion(tweet) for tweet in tweets]
    article_opinion = quantify_opinion(article_text)
    
    # Step 4: Save the quantified opinions to a JSON file
    #with open("opinions.json", "w") as f:
        #json.dump({"tweets": tweet_opinions, "article": article_opinion}, f, indent=2)
    
    logging.info("Pipeline complete.")
    print(article_opinion)
    #return tweet_opinions, article_opinion
    return

if __name__ == "__main__":
    pipeline()

