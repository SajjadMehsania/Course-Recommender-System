import pandas as pd
from prophet import Prophet
import os

df = pd.read_csv("data/courses.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

agg = df.groupby(['course_id', 'timestamp']).size().reset_index(name='enrollments')

forecast_summary = []

os.makedirs("forecasts", exist_ok=True)

for course_id in agg['course_id'].unique():
    sub_df = agg[agg['course_id'] == course_id][['timestamp', 'enrollments']]
    sub_df = sub_df.rename(columns={'timestamp': 'ds', 'enrollments': 'y'})

    if len(sub_df) < 10:
        continue

    model = Prophet()
    model.fit(sub_df)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    future_forecast = forecast.tail(30)
    mean_yhat = future_forecast['yhat'].mean()

    forecast_summary.append({
        'course_id': course_id,
        'predicted_enrollment': round(mean_yhat, 2)
    })

summary_df = pd.DataFrame(forecast_summary)
summary_df.to_csv("forecasts/forecast_summary.csv", index=False)

print("✅ Forecast completed and saved to forecasts/forecast_summary.csv")
