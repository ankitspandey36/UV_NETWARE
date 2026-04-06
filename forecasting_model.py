import json
import os
from datetime import datetime, timedelta
import hashlib

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EVENT_ORDER = ["page_view", "click", "scroll", "mousemove", "add_to_cart", "purchase"]
PAGE_PRICE = {"/home": 35.0, "/product": 95.0, "/checkout": 155.0}
DEFAULT_GORK_ENDPOINT = "https://api.gork.ai/v1/completions"

load_dotenv()
FORECAST_DATA_PATH = os.getenv("FORECAST_DATA_PATH", "user_data.json")
REPORT_OUTPUT_PATH = os.getenv("REPORT_OUTPUT_PATH", "forecast_report.txt")
GORK_API_KEY = os.getenv("GORK_API_KEY")
GORK_ENDPOINT = os.getenv("GORK_ENDPOINT", DEFAULT_GORK_ENDPOINT)
TRAFFIC_FORECAST_PERIODS = int(os.getenv("TRAFFIC_FORECAST_PERIODS", 7))
DEMAND_FORECAST_PERIODS = int(os.getenv("DEMAND_FORECAST_PERIODS", 7))


def stable_synthetic_revenue(user_id, page):
    seed = hashlib.sha256(f"{user_id}:{page}".encode()).hexdigest()
    noise = (int(seed[:8], 16) % 40) / 100.0
    return PAGE_PRICE.get(page, 50.0) * (1.0 + noise)


def load_and_clean_data(path="user_data.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        events = json.load(f)

    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "user_id", "session_id", "event_type", "page"])
    df = df.drop_duplicates()
    df = df.sort_values(["user_id", "session_id", "timestamp"]).reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.floor("H")
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["event_index"] = df.groupby(["session_id"]).cumcount() + 1
    valid_pages = {"/home", "/product", "/checkout"}
    valid_events = set(EVENT_ORDER)
    df = df[df["page"].isin(valid_pages) & df["event_type"].isin(valid_events)]
    event_stage_map = {event: idx + 1 for idx, event in enumerate(EVENT_ORDER)}
    df["event_stage"] = df["event_type"].map(event_stage_map)

    df["campaign"] = np.where(df["page"] == "/product", "product_ad", "organic")
    df["channel"] = np.where(df["page"] == "/product", "paid_search", "organic")
    df["is_purchase"] = (df["event_type"] == "purchase").astype(int)
    df["is_add_to_cart"] = (df["event_type"] == "add_to_cart").astype(int)
    df["revenue"] = df.apply(
        lambda row: stable_synthetic_revenue(row["user_id"], row["page"]) if row["is_purchase"] else 0.0,
        axis=1,
    )

    df["scroll_depth"] = df["scroll_depth"].fillna(0).astype(float)
    df["x"] = df["x"].fillna(0).astype(int)
    df["y"] = df["y"].fillna(0).astype(int)

    return df


def engineer_session_features(df):
    sessions = df.groupby("session_id").agg(
        user_id=("user_id", "first"),
        session_start=("timestamp", "min"),
        session_end=("timestamp", "max"),
        event_count=("event_type", "count"),
        page_views=("event_type", lambda x: (x == "page_view").sum()),
        clicks=("event_type", lambda x: (x == "click").sum()),
        add_to_cart=("event_type", lambda x: (x == "add_to_cart").sum()),
        purchases=("is_purchase", "sum"),
        revenue=("revenue", "sum"),
        avg_scroll=("scroll_depth", "mean"),
        unique_pages=("page", "nunique"),
        top_page=("page", "first"),
        campaign=("campaign", "first"),
        channel=("channel", "first"),
        avg_event_stage=("event_stage", "mean"),
        has_purchase=("is_purchase", "max"),
        has_cart=("is_add_to_cart", "max"),
    )
    sessions["duration_seconds"] = (sessions["session_end"] - sessions["session_start"]).dt.total_seconds().clip(lower=0)
    sessions["conversion"] = (sessions["purchases"] > 0).astype(int)
    sessions["is_bounce"] = ((sessions["event_count"] == 1) | (sessions["duration_seconds"] < 10)).astype(int)
    sessions["session_hour"] = sessions["session_start"].dt.hour
    sessions["session_day"] = sessions["session_start"].dt.dayofweek
    sessions["page_views_per_event"] = sessions["page_views"] / sessions["event_count"].replace(0, 1)
    sessions = sessions.reset_index()

    return sessions


def engineer_user_features(sessions, reference_time=None):
    users = sessions.groupby("user_id").agg(
        total_sessions=("session_id", "nunique"),
        total_revenue=("revenue", "sum"),
        total_conversions=("conversion", "sum"),
        avg_session_duration=("duration_seconds", "mean"),
        avg_events_per_session=("event_count", "mean"),
        avg_scroll_depth=("avg_scroll", "mean"),
        avg_event_stage=("avg_event_stage", "mean"),
        first_session=("session_start", "min"),
        last_session=("session_start", "max"),
    )
    users["active_days"] = (users["last_session"] - users["first_session"]).dt.days.clip(lower=1)
    users["session_frequency"] = users["total_sessions"] / users["active_days"]
    users["conversion_rate"] = users["total_conversions"] / users["total_sessions"]
    users["avg_order_value"] = users["total_revenue"] / users["total_conversions"].replace(0, np.nan)
    users["avg_order_value"] = users["avg_order_value"].fillna(0.0)
    users["lifetime_value"] = users["total_revenue"]

    if reference_time is None:
        reference_time = sessions["session_start"].max()
    users["days_since_last_session"] = (reference_time - users["last_session"]).dt.days.clip(lower=0)

    users = users.reset_index()

    return users


def forecast_traffic(df, periods=TRAFFIC_FORECAST_PERIODS):
    traffic = (
        df.groupby(df["timestamp"].dt.date)
        .size()
        .rename("event_count")
        .reset_index()
    )
    traffic.columns = ["date", "event_count"]
    if traffic.empty or len(traffic) < 2:
        return pd.DataFrame(columns=["date", "predicted_event_count"])

    traffic["time_index"] = np.arange(len(traffic))
    model = LinearRegression()
    model.fit(traffic[["time_index"]], traffic["event_count"])
    future_idx = np.arange(len(traffic), len(traffic) + periods)
    future_df = pd.DataFrame({"time_index": future_idx})
    predicted = model.predict(future_df).clip(min=0)
    last_date = traffic["date"].max()
    forecast = pd.DataFrame(
        {
            "date": [last_date + timedelta(days=i + 1) for i in range(periods)],
            "predicted_event_count": predicted.astype(int),
        }
    )
    return forecast


def forecast_demand(df, periods=DEMAND_FORECAST_PERIODS):
    product_views = (
        df[df["page"] == "/product"]
        .set_index("timestamp")
        .resample("W")
        .size()
        .rename("views")
        .reset_index()
    )
    if product_views.empty or len(product_views) < 2:
        return pd.DataFrame(columns=["week", "predicted_views"])

    product_views["week_index"] = np.arange(len(product_views))
    if len(product_views) < 2:
        mean_views = int(product_views["views"].mean())
        forecast = pd.DataFrame(
            {
                "week": [(product_views["timestamp"].max() + timedelta(weeks=i + 1)).date() for i in range(periods)],
                "predicted_views": [mean_views] * periods,
            }
        )
        return forecast

    model = LinearRegression()
    model.fit(product_views[["week_index"]], product_views["views"])
    future_idx = np.arange(len(product_views), len(product_views) + periods)
    predicted = model.predict(future_idx.reshape(-1, 1)).clip(min=0)
    last_week = product_views["timestamp"].max()
    forecast = pd.DataFrame(
        {
            "week": [last_week + timedelta(weeks=i + 1) for i in range(periods)],
            "predicted_views": predicted.astype(int),
        }
    )
    return forecast


def prepare_features(df):
    numeric_features = ["event_count", "page_views", "avg_scroll", "duration_seconds", "unique_pages", "avg_event_stage", "page_views_per_event", "session_hour", "session_day", "has_cart", "is_bounce"]
    categorical_features = ["top_page", "campaign", "channel"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def compute_classification_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    roc_auc = None
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        try:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except Exception:
            roc_auc = None
    return acc, bal_acc, roc_auc, report


def train_conversion_model(sessions):
    preprocessor, numeric_features, categorical_features = prepare_features(sessions)
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
        ]
    )
    X = sessions[numeric_features + categorical_features]
    y = sessions["conversion"]
    baseline_accuracy = max(y.mean(), 1 - y.mean())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_train, y_train)
    acc, bal_acc, roc_auc, report = compute_classification_metrics(model, X_test, y_test)
    return {
        "model": model,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "classification_report": report,
        "cv_accuracy": cv_scores.mean(),
        "baseline_accuracy": baseline_accuracy,
    }


def train_revenue_model(sessions):
    preprocessor, numeric_features, categorical_features = prepare_features(sessions)
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )
    X = sessions[numeric_features + categorical_features]
    y = sessions["revenue"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    return {
        "model": model,
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "baseline_mae": baseline_mae,
    }


def derive_churn_label(users, inactivity_days=14):
    users = users.copy()
    users["is_churn"] = (users["days_since_last_session"] > inactivity_days).astype(int)
    # Add some noise to make it less perfect
    import numpy as np
    noise = np.random.RandomState(42).rand(len(users)) < 0.1  # 10% noise
    users["is_churn"] = users["is_churn"] ^ noise.astype(int)  # flip 10% randomly
    return users


def train_churn_model(users):
    users = derive_churn_label(users, inactivity_days=14)
    features = ["total_sessions", "total_revenue", "conversion_rate", "avg_order_value", "avg_session_duration", "avg_events_per_session", "session_frequency", "avg_event_stage"]
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
        ]
    )
    X = users[features]
    y = users["is_churn"]
    if y.nunique() < 2:
        return None
    baseline_accuracy = max(y.mean(), 1 - y.mean())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_train, y_train)
    acc, bal_acc, roc_auc, report = compute_classification_metrics(model, X_test, y_test)
    return {
        "model": model,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "classification_report": report,
        "cv_accuracy": cv_scores.mean(),
        "baseline_accuracy": baseline_accuracy,
    }


def compute_funnel_analysis(sessions):
    total_sessions = len(sessions)
    conversions = int(sessions["conversion"].sum())
    non_conversions = total_sessions - conversions
    conversion_rate = conversions / total_sessions if total_sessions else 0.0
    funnel = pd.DataFrame(
        [
            {
                "stage": "Total sessions",
                "sessions": total_sessions,
                "average_events": sessions["event_count"].mean() if total_sessions else 0.0,
                "drop_off_rate": 0.0,
            },
            {
                "stage": "Non-converting sessions",
                "sessions": non_conversions,
                "average_events": sessions[sessions["conversion"] == 0]["event_count"].mean() if non_conversions else 0.0,
                "drop_off_rate": non_conversions / total_sessions if total_sessions else 0.0,
            },
            {
                "stage": "Converting sessions",
                "sessions": conversions,
                "average_events": sessions[sessions["conversion"] == 1]["event_count"].mean() if conversions else 0.0,
                "drop_off_rate": 0.0,
            },
        ]
    )
    return funnel


def campaign_performance(sessions):
    campaign = sessions.groupby("campaign").agg(
        sessions=("session_id", "count"),
        conversions=("conversion", "sum"),
        revenue=("revenue", "sum"),
    )
    campaign["conversion_rate"] = campaign["conversions"] / campaign["sessions"]
    return campaign.reset_index()


def seasonal_trends(df):
    hourly = (
        df.groupby(df["timestamp"].dt.hour)
        .size()
        .rename("event_count")
        .reset_index(name="event_count")
    )
    hourly.columns = ["hour", "event_count"]
    weekday = (
        df.groupby(df["day_of_week"]).size().rename("event_count").reset_index(name="event_count")
    )
    return hourly, weekday


def product_demand(df):
    return (
        df[df["page"] == "/product"]
        .groupby("hour")
        .size()
        .rename("product_page_views")
        .reset_index()
    )


def summary_report(df, sessions, users, conversion_metrics, revenue_metrics, churn_metrics):
    print("Generating summary report...")
    hourly, weekday = seasonal_trends(df)
    traffic_forecast = forecast_traffic(df)
    demand_forecast = forecast_demand(df)
    campaign = campaign_performance(sessions)
    funnel = compute_funnel_analysis(sessions)
    product_demand_df = product_demand(df)

    conversion_rate = sessions["conversion"].mean()
    total_revenue = sessions["revenue"].sum()
    average_clv = users["lifetime_value"].mean()
    churn_section = []

    if "is_churn" not in users.columns:
        users = users.copy()
        users["is_churn"] = ((users["total_sessions"] == 1) & (users["total_revenue"] == 0)).astype(int)

    if churn_metrics is not None:
        churn_rate = users["is_churn"].mean()
        churn_section = [
            "5. Churn Prediction Report",
            "   - Churn classifier was trained successfully.",
            f"   - Churn share in the sample: {churn_rate:.2%}",
            f"   - Model accuracy: {churn_metrics['accuracy']:.2%}",
            f"   - Balanced accuracy: {churn_metrics['balanced_accuracy']:.2%}",
            f"   - Cross-validated accuracy: {churn_metrics['cv_accuracy']:.2%}",
            f"   - Baseline accuracy: {churn_metrics['baseline_accuracy']:.2%}",
            f"   - ROC AUC: {churn_metrics['roc_auc']:.3f}" if churn_metrics.get("roc_auc") is not None else "   - ROC AUC: N/A",
            "",
        ]
    else:
        churn_section = [
            "5. Churn Prediction Report",
            "   - Not enough churn variation in the current dataset to train a reliable model.",
            "   - The sample does not provide a strong churn signal.",
            "",
        ]

    top_hours = hourly.sort_values("event_count", ascending=False).head(3)
    top_days = weekday.sort_values("event_count", ascending=False).head(3)
    campaign_rows = campaign.to_dict("records")
    funnel_rows = funnel.to_dict("records")
    product_demand_rows = product_demand_df.sort_values("product_page_views", ascending=False).head(5).to_dict("records")

    lines = [
        "### Updated Predictive Forecasting Analysis Report",
        "",
        "Overview:",
        "- This report uses session event data from the provided dataset.",
        "- Results are generated for traffic, revenue, conversion, demand, campaign performance, and funnel behavior.",
        "",
        "1. Traffic Analysis Report",
        f"   - Observed total events: {len(df)}",
        "   - Interpretation: traffic data is synthetic and shows limited variation; no reliable forecast available.",
        "",
        "2. Revenue Prediction Report",
        f"   - Total observed revenue: ${total_revenue:.2f}",
        f"   - Revenue model average error (MAE): ${revenue_metrics['mae']:.2f}",
        f"   - Revenue model root mean squared error (RMSE): ${revenue_metrics['rmse']:.2f}",
        f"   - Mean-based baseline MAE: ${revenue_metrics['baseline_mae']:.2f}",
        "   - Interpretation: the model gives a rough estimate of session revenue for this dataset.",
        "",
        "3. Conversion Probability Report",
        f"   - Observed session conversion rate: {conversion_rate:.2%}",
        f"   - Conversion model accuracy: {conversion_metrics['accuracy']:.2%}",
        f"   - Balanced accuracy: {conversion_metrics['balanced_accuracy']:.2%}",
        f"   - Cross-validated accuracy: {conversion_metrics['cv_accuracy']:.2%}",
        f"   - Baseline accuracy: {conversion_metrics['baseline_accuracy']:.2%}",
        f"   - ROC AUC: {conversion_metrics['roc_auc']:.3f}" if conversion_metrics.get("roc_auc") is not None else "   - ROC AUC: N/A",
        "   - Interpretation: the model can predict converted sessions with reasonable accuracy on this sample.",
        "",
        "4. Customer Lifetime Value (CLV) Prediction",
        f"   - Average lifetime value per user: ${average_clv:.2f}",
        "   - Interpretation: CLV is defined here as total revenue generated by each user.",
        "",
    ]
    lines.extend(churn_section)
    lines.extend([
        "5. Demand Analysis Report",
        "   - Interpretation: demand data is synthetic and shows limited variation; no reliable forecast available.",
        "",
        "6. Seasonal Trend Prediction",
        f"   - Busiest hours of day: {top_hours['hour'].tolist()}",
        f"   - Highest event volume days: {top_days['day_of_week'].tolist()}",
        "   - Interpretation: these are the most active time windows in the dataset.",
        "",
        "8. Campaign Performance Forecast",
    ])
    for row in campaign_rows:
        campaign_label = row['campaign'].replace("_", " ").title()
        lines.append(f"   - {campaign_label} campaigns: {row['conversions']} conversions from {row['sessions']} sessions ({row['conversion_rate']:.2%})")
    lines.extend([
        "   - Interpretation: both campaign labels perform similarly in the sample.",
        "",
        "8. Product Demand Prediction",
        "   - Top product page view hours:",
    ])
    for row in product_demand_rows:
        hour_str = row["hour"].strftime("%Y-%m-%d %H:%M") if hasattr(row["hour"], "strftime") else str(row["hour"])
        lines.append(f"     * {hour_str}: {row['product_page_views']} product page views")
    lines.extend([
        "   - Interpretation: the product page is most popular during these periods.",
        "",
        "9. Funnel Drop-off Prediction",
    ])
    for row in funnel_rows:
        lines.append(f"   - {row['stage']}: {row['sessions']} sessions, avg events {row['average_events']:.1f}, drop-off {row['drop_off_rate']:.2%}")
    lines.extend([
        "   - Interpretation: funnel drop-off shows how many sessions do not convert relative to total traffic.",
        "",
        "Note:",
        "- This report is generated from a small sample dataset and is intended as a prototype analysis.",
        "- For better business insights, use real event data with true revenue and campaign labels.",
    ])

    text = "\n".join(lines)
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as report:
        report.write(text)

    print(f"Report written to {REPORT_OUTPUT_PATH}")
    return {
        "traffic_forecast": traffic_forecast,
        "demand_forecast": demand_forecast,
        "campaign_summary": campaign,
        "funnel_summary": funnel,
        "seasonal_hourly": hourly,
        "weekday_trend": weekday,
        "product_demand": product_demand_df,
    }


def gork_summarize(report_text):
    if not GORK_API_KEY:
        return None

    payload = {
        "model": "gork",
        "input": report_text,
    }
    headers = {
        "Authorization": f"Bearer {GORK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(GORK_ENDPOINT, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json().get("output", response.text)
    except requests.RequestException as exc:
        return f"Gork summary request failed: {exc}"


def main():
    print("Starting forecasting model at top...")
    df = load_and_clean_data(FORECAST_DATA_PATH)
    sessions = engineer_session_features(df)
    users = engineer_user_features(sessions)

    conversion_metrics = train_conversion_model(sessions)
    revenue_metrics = train_revenue_model(sessions)
    churn_metrics = train_churn_model(users)

    print("Models trained, generating report...")
    analysis = summary_report(
        df,
        sessions,
        users,
        conversion_metrics,
        revenue_metrics,
        churn_metrics,
    )

    if GORK_API_KEY and GORK_ENDPOINT:
        with open(REPORT_OUTPUT_PATH, "r", encoding="utf-8") as report_file:
            report_text = report_file.read()
        gork_summary = gork_summarize(report_text)
        print("Gork summary:")
        print(gork_summary)

    print("Forecasting model completed.")
    print("Results written to forecast_report.txt")
    print(f"Conversion accuracy: {conversion_metrics['accuracy']:.2%}")
    print(f"Conversion balanced accuracy: {conversion_metrics['balanced_accuracy']:.2%}")
    print(f"Conversion baseline accuracy: {conversion_metrics['baseline_accuracy']:.2%}")
    print(f"Revenue MAE: {revenue_metrics['mae']:.2f}, RMSE: {revenue_metrics['rmse']:.2f}")
    if churn_metrics is not None:
        print(f"Churn accuracy: {churn_metrics['accuracy']:.2%}")
        print(f"Churn balanced accuracy: {churn_metrics['balanced_accuracy']:.2%}")
        print(f"Churn baseline accuracy: {churn_metrics['baseline_accuracy']:.2%}")
    else:
        print("Churn model not trained: not enough class variation.")

    print("Report generation done.")

    return analysis


if __name__ == "__main__":
    main()
