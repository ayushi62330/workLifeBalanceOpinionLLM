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

def compute_dataframe(opinions, stage: str) -> pd.DataFrame:
    """
    Convert a list of opinion objects into a Pandas DataFrame and add a 'Stage' column.
    If opinions is a single dictionary, wrap it into a list.
    """
    if not isinstance(opinions, list):
        opinions = [opinions]
    df = pd.DataFrame(opinions)
    df["Stage"] = stage
    return df

def create_dashboard_dynamic(file_path="opinions.json"):
    """
    Generate an integrated HTML dashboard for the entire application.
    The dashboard includes:
      - A combined table of opinion metrics for tweets and articles.
      - An interactive bar chart showing overall sentiment by stage.
    """
    data = load_opinions(file_path)
    if data is None:
        print("No opinions data found.")
        return

    dfs = []
    if "tweets" in data and data["tweets"]:
        df_tweets = compute_dataframe(data["tweets"], "Tweets")
        dfs.append(df_tweets)
    if "articles" in data and data["articles"]:
        df_articles = compute_dataframe(data["articles"], "Articles")
        dfs.append(df_articles)
    
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        print("No opinion data available.")
        return

    # Create a bar chart of average overall sentiment by stage.
    sentiment_df = df_all.groupby("Stage", as_index=False)["overall_sentiment"].mean()
    fig = px.bar(sentiment_df, x="Stage", y="overall_sentiment",
                 title="Average Overall Sentiment by Stage",
                 labels={"overall_sentiment": "Overall Sentiment (1-5)"})
    
    # Build the Datapane report with a text title, table, and bar chart.
    report = dp.Report(
        dp.Text("# Integrated Monitoring Dashboard"),
        dp.Table(df_all, caption="Opinion Metrics for Tweets and Articles"),
        dp.Plot(fig, caption="Average Overall Sentiment by Stage")
    )
    report.save(path="dashboard.html", open=True)
    print("Dashboard generated and saved as dashboard.html")

if __name__ == "__main__":
    create_dashboard_dynamic()
