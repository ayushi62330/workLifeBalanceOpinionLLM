import os
import json
import pandas as pd
import datapane as dp
import plotly.express as px

def load_opinions(file_path="opinions.json"):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def compute_dataframe(opinions) -> pd.DataFrame:
    if not isinstance(opinions, list):
        opinions = [opinions]
    return pd.DataFrame(opinions)

def create_dashboard_dynamic(file_path="opinions.json"):
    data = load_opinions(file_path)
    if data is None:
        print("No opinions data found.")
        return

    if "articles" in data and data["articles"]:
        df_articles = compute_dataframe(data["articles"])
    else:
        print("No article opinion data available.")
        return

    numeric_df = df_articles.select_dtypes(include=["number"])
    
    averages = numeric_df.mean().reset_index()
    averages.columns = ["Metric", "Average Value"]

    fig = px.bar(
        averages,
        x="Metric",
        y="Average Value",
        title="Average Opinion Metrics for Articles",
        labels={"Average Value": "Average Value"}
    )
    
    report = dp.Report(
        dp.Text("# Articles Opinion Dashboard"),
        dp.Table(averages, caption="Average Opinion Metrics for Articles"),
        dp.Plot(fig, caption="Average Opinion Metrics for Articles")
    )
    report.save(path="dashboard.html", open=True)
    print("Dashboard generated and saved as dashboard.html")

if __name__ == "__main__":
    create_dashboard_dynamic()
