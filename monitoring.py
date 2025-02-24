import os
import json
import pandas as pd
import datapane as dp

def load_opinions(file_path="opinions.json"):
    """Load opinions data from the JSON file if it exists."""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def compute_metrics(data):
    """
    Compute dynamic metrics from the opinions data.
    
    Expected JSON structure:
    {
      "tweets": [
         { "work_flexibility": int, "burnout_risk": int, "remote_work_appeal": int,
           "productivity_impact": int, "overall_sentiment": int },
         ...
      ],
      "article": { "work_flexibility": int, "burnout_risk": int, "remote_work_appeal": int,
                   "productivity_impact": int, "overall_sentiment": int }
    }
    """
    metrics = []
    # Process tweets
    tweets = data.get("tweets", [])
    num_tweets = len(tweets)
    if num_tweets > 0:
        keys = ["work_flexibility", "burnout_risk", "remote_work_appeal", "productivity_impact", "overall_sentiment"]
        tweet_avgs = { key: sum(tweet.get(key, 0) for tweet in tweets) / num_tweets for key in keys }
        metrics.append({
            "Stage": "Tweets",
            "Count": num_tweets,
            **tweet_avgs
        })
    else:
        metrics.append({
            "Stage": "Tweets",
            "Count": 0,
            "work_flexibility": 0,
            "burnout_risk": 0,
            "remote_work_appeal": 0,
            "productivity_impact": 0,
            "overall_sentiment": 0
        })
    
    # Process article
    article = data.get("article", {})
    if article:
        metrics.append({
            "Stage": "Article",
            "Count": 1,
            **article
        })
    else:
        metrics.append({
            "Stage": "Article",
            "Count": 0,
            "work_flexibility": 0,
            "burnout_risk": 0,
            "remote_work_appeal": 0,
            "productivity_impact": 0,
            "overall_sentiment": 0
        })
    return metrics

def create_dashboard_dynamic(file_path="opinions.json"):
    data = load_opinions(file_path)
    
    if data is None:
        metrics_data = [
            {
                "Stage": "Tweets",
                "Count": 0,
                "work_flexibility": 0,
                "burnout_risk": 0,
                "remote_work_appeal": 0,
                "productivity_impact": 0,
                "overall_sentiment": 0
            },
            {
                "Stage": "Article",
                "Count": 0,
                "work_flexibility": 0,
                "burnout_risk": 0,
                "remote_work_appeal": 0,
                "productivity_impact": 0,
                "overall_sentiment": 0
            }
        ]
    else:
        metrics_data = compute_metrics(data)
    
    df = pd.DataFrame(metrics_data)
    report = dp.Report(dp.Table(df, caption="Dynamic Pipeline Metrics"))
    report.save(path="dashboard.html", open=True)
    print("Dashboard generated and saved as dashboard.html")

if __name__ == "__main__":
    create_dashboard_dynamic()
