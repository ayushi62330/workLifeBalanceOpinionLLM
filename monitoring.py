import datapane as dp
import pandas as pd

def create_dashboard(metrics: list):
    df = pd.DataFrame(metrics)
    report = dp.Report(dp.Table(df))
    report.save(path="dashboard.html", open=True)

if __name__ == "__main__":
    # Example metrics; in practice, aggregate these from your pipeline logs or outputs
    metrics_data = [
        {"stage": "ingestion", "documents": 51},
        {"stage": "embedding", "documents": 51},
        {"stage": "quantification", "average_sentiment": 4.0}
    ]
    create_dashboard(metrics_data)
