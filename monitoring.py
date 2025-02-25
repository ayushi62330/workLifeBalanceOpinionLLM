import os
import json
import pandas as pd
import datapane as dp
import plotly.express as px

def load_opinions(file_path="opinions.json"):
    """Load opinions data from the JSON file if it exists."""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def compute_article_metrics(articles: list) -> pd.DataFrame:
    """
    Compute metrics for each article in the provided list.
    Each article is expected to be a dictionary with keys:
      - work_flexibility, burnout_risk, remote_work_appeal, productivity_impact, overall_sentiment
    Optionally, it may contain extra keys.
    Returns a DataFrame with one row per article.
    """
    records = []
    for idx, article in enumerate(articles):
        record = {
            "Article": f"Article {idx+1}",
            "work_flexibility": article.get("work_flexibility"),
            "burnout_risk": article.get("burnout_risk"),
            "remote_work_appeal": article.get("remote_work_appeal"),
            "productivity_impact": article.get("productivity_impact"),
            "overall_sentiment": article.get("overall_sentiment"),
            "generation": article.get("generation", ""),
            "prompt_token_count": article.get("prompt_token_count"),
            "generation_token_count": article.get("generation_token_count"),
            "stop_reason": article.get("stop_reason")
        }
        records.append(record)
    df = pd.DataFrame(records)
    return df

def create_dashboard_dynamic(file_path="opinions.json"):
    """
    Generate an enhanced HTML dashboard summarizing article metrics.
    The dashboard includes:
      - A table with key metrics for each article.
      - A bar chart showing overall sentiment for each article.
    """
    data = load_opinions(file_path)
    if data is None or "articles" not in data:
        print("No article data found in opinions.json under 'articles' key.")
        return

    articles = data["articles"]
    df = compute_article_metrics(articles)
    
    # Create an interactive bar chart for overall sentiment
    fig = px.bar(df, x="Article", y="overall_sentiment",
                 title="Overall Sentiment per Article",
                 labels={"overall_sentiment": "Overall Sentiment (1-5)"})
    
    # Build the Datapane report with a table and the bar chart
    report = dp.Report(
        dp.Markdown("# Article Metrics Dashboard"),
        dp.Table(df, caption="Detailed Article Metrics"),
        dp.Plot(fig, caption="Overall Sentiment per Article")
    )
    report.save(path="dashboard.html", open=True)
    print("Dashboard generated and saved as dashboard.html")

if __name__ == "__main__":
    create_dashboard_dynamic()
