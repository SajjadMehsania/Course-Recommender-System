# import pandas as pd
# df = pd.read_csv("data/courses.csv")
# print("Sample data:\n", df.head())

# from sklearn.preprocessing import LabelEncoder

# user_enc = LabelEncoder()
# course_enc = LabelEncoder()

# df['user'] = user_enc.fit_transform(df['user_id'])
# df['course'] = course_enc.fit_transform(df['course_id'])

# print("\nEncoded data:\n", df[['user_id', 'course_id', 'user', 'course']].head())

# from models.neural_model import build_model
# import numpy as np

# num_users = df['user'].nunique()
# num_courses = df['course'].nunique()

# model = build_model(num_users, num_courses)

# X = [df['user'].values, df['course'].values]
# y = df['rating'].values

# model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# target_user_id = 12

# all_courses = np.array([i for i in range(num_courses)])

# user_array = np.full(shape=(num_courses,), fill_value=target_user_id)

# predictions = model.predict([user_array, all_courses], verbose=0)

# top_indices = predictions.reshape(-1).argsort()[::-1]

# recommended_courses = course_enc.inverse_transform(top_indices[:5])

# print(f"\nTop 5 recommended courses for user_id {user_enc.inverse_transform([target_user_id])[0]}:")
# print(recommended_courses)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from models.neural_model import build_model

df = pd.read_csv("data/courses.csv")
print("Sample data:\n", df.head())

user_enc = LabelEncoder()
course_enc = LabelEncoder()

df['user'] = user_enc.fit_transform(df['user_id'])
df['course'] = course_enc.fit_transform(df['course_id'])

print("\nEncoded data:\n", df[['user_id', 'course_id', 'user', 'course']].head())

num_users = df['user'].nunique()
num_courses = df['course'].nunique()

model = build_model(num_users, num_courses)

X = [df['user'].values, df['course'].values]
y = df['rating'].values

model.fit(X, y, epochs=10, batch_size=32, verbose=1)

target_user_id = 10

all_courses = np.array([i for i in range(num_courses)])
user_array = np.full(shape=(num_courses,), fill_value=target_user_id)

predictions = model.predict([user_array, all_courses], verbose=0).reshape(-1)

decoded_course_ids = course_enc.inverse_transform(all_courses)
original_user_id = user_enc.inverse_transform([target_user_id])[0]
user_name = df[df['user_id'] == original_user_id]['user_name'].iloc[0]

forecast_df = pd.read_csv("forecasts/forecast_summary.csv")

rec_df = pd.DataFrame({
    'course_id': decoded_course_ids,
    'predicted_rating': predictions
})

rec_df = rec_df.merge(forecast_df, on='course_id', how='left')
rec_df['predicted_enrollment'] = rec_df['predicted_enrollment'].fillna(0)

min_enroll = rec_df['predicted_enrollment'].min()
max_enroll = rec_df['predicted_enrollment'].max()
rec_df['popularity_score'] = (rec_df['predicted_enrollment'] - min_enroll) / (max_enroll - min_enroll + 1e-6)

rec_df['final_score'] = 0.7 * rec_df['predicted_rating'] + 0.3 * rec_df['popularity_score']

top_courses = rec_df.sort_values('final_score', ascending=False).head(5)

top_courses = top_courses.merge(
    df[['course_id', 'course_name']].drop_duplicates(),
    on='course_id',
    how='left'
)

print(f"\nTop 5 recommended courses for {user_name} (user_id {original_user_id}):")
for _, row in top_courses.iterrows():
    print(f"- {row['course_name']} (ID: {row['course_id']}), Rating: {row['predicted_rating']:.2f}, Popularity: {row['popularity_score']:.2f}")
