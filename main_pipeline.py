import os
import re
import json
import logging
import tweepy
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from chromadb import Client  # Ensure chromadb is installed via pip
from dotenv import load_dotenv
import boto3

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = Client()  

def ingest_article(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    article_text = "\n".join(p.get_text() for p in paragraphs)
    logging.info(f"Ingested article from {url}.")
    return article_text

def embed_and_store(texts, collection_name: str):
    embeddings = model.encode(texts)
    collection = chroma_client.get_or_create_collection(name=collection_name)
    # Generate unique IDs for each document, e.g., by using their index
    ids = [str(i) for i in range(len(texts))]
    collection.add(ids=ids, documents=texts, embeddings=embeddings)
    logging.info(f"Stored {len(texts)} documents in collection '{collection_name}'.")

def extract_json_from_generation(raw_output: dict) -> dict:
    if "generation" in raw_output:
        gen_str = raw_output["generation"]
        start = gen_str.find('{')
        end = gen_str.rfind('}')
        if start == -1 or end == -1:
            raise ValueError("Could not find JSON object in generation string.")
        json_str = gen_str[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError("Error parsing JSON from generation string: " + json_str) from e
    return raw_output

def quantify_opinion(text: str) -> dict:
    bedrock_client = boto3.client("bedrock-runtime", region_name="ap-south-1")
    
    prompt_text = (
        "Analyze the following text regarding work-life balance. "
        "Return a JSON object with exactly these keys: "
        "\"work_flexibility\", \"burnout_risk\", \"remote_work_appeal\", "
        "\"productivity_impact\", \"overall_sentiment\". "
        "Each value must be an integer between 1 and 5. "
        f"Text: \"{text}\""
    )
    body = json.dumps({"prompt": prompt_text})
    
    try:
        bedrock_response = bedrock_client.invoke_model(
            modelId="meta.llama3-70b-instruct-v1:0",  # Replace with a valid model ID
            body=body.encode("utf-8"),
            contentType="application/json"
        )
    except Exception as e:
        logger.error("Error invoking Bedrock model: %s", e)
        raise

    if "body" not in bedrock_response:
        raise ValueError(f"InvokeModel response missing 'body': {bedrock_response}")
    
    raw_response_str = bedrock_response["body"].read().decode("utf-8")
    logger.info("Raw response from Bedrock: %s", raw_response_str)
    try:
        raw_output = json.loads(raw_response_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from Bedrock: {raw_response_str}") from e

    try:
        cleaned_output = extract_json_from_generation(raw_output)
    except Exception as e:
        logger.warning("Error extracting JSON: %s. Falling back to simulated analysis.", e)
        return quantify_opinion_simulated(text)
    logger.info("Cleaned output: %s", cleaned_output)
    return cleaned_output
    

def pipeline():
    article_urls = [
        "https://hbr.org/2021/01/work-life-balance-is-a-cycle-not-an-achievement",
        "https://thehappinessindex.com/blog/importance-work-life-balance/",
        "https://auroratrainingadvantage.com/articles/importance-of-work-life-balance/",
        "https://www.businessnewsdaily.com/5244-improve-work-life-balance-today.html",
        "https://www.bbc.com/worklife/article/20230227-what-does-work-life-balance-mean-in-a-changed-work-world"
    ]
    
    articles_texts = []
    for url in article_urls:
        try:
            text = ingest_article(url)
            articles_texts.append(text)
        except Exception as e:
            logger.error("Error processing article %s: %s", url, e)
    
    if articles_texts:
        embed_and_store(articles_texts, collection_name="articles_worklife")
        articles_opinions = [quantify_opinion(text) for text in articles_texts]
    else:
        articles_opinions = []

    output = {
        "articles": articles_opinions
    }
    try:
        with open("opinions.json", "w") as f:
            json.dump(output, f, indent=2)
    except Exception as e:
        logger.error("Error writing opinions.json: %s", e)
        raise

    logger.info("Pipeline complete.")
    print("Final output:", output)
    return output

if __name__ == "__main__":
    pipeline()
