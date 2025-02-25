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

def compute_article_metrics(articles: list) -> list:
    """
    Compute metrics for each article in the provided list.
    Each article is expected to be a dictionary with keys:
      - work_flexibility, burnout_risk, remote_work_appeal, productivity_impact, overall_sentiment
    Optionally, it may contain extra keys (e.g., generation, token counts, etc.).
    Returns a list of dictionaries, one per article, with a 'Stage' label added.
    """
    metrics_list = []
    numeric_keys = ["work_flexibility", "burnout_risk", "remote_work_appeal", "productivity_impact", "overall_sentiment"]
    extra_keys = ["generation", "prompt_token_count", "generation_token_count", "stop_reason"]
    
    for idx, article in enumerate(articles):
        article_metrics = {"Stage": f"Article {idx+1}", "Count": 1}
        for key in numeric_keys:
            article_metrics[key] = article.get(key, None)
        for key in extra_keys:
            article_metrics[key] = article.get(key, None)
        metrics_list.append(article_metrics)
    return metrics_list

def create_dashboard_dynamic(file_path="opinions.json"):
    """
    Generate an HTML dashboard summarizing metrics for all articles.
    """
    data = load_opinions(file_path)
    if data is None or "articles" not in data:
        print("No article data found in opinions.json under 'articles' key.")
        return

    articles = data["articles"]
    metrics_data = compute_article_metrics(articles)
    df = pd.DataFrame(metrics_data)
    report = dp.Report(dp.Table(df, caption="Article Metrics"))
    report.save(path="dashboard.html", open=True)
    print("Dashboard generated and saved as dashboard.html")

if __name__ == "__main__":
    create_dashboard_dynamic()
