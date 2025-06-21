import pandas as pd
import random
from datetime import datetime, timedelta

num_users = 100
num_courses = 50
num_rows = 5000

user_names = [f"User_{i}" for i in range(1, num_users + 1)]
user_id_name_map = {i + 1: user_names[i] for i in range(num_users)}  

course_ids = [i for i in range(101, 101 + num_courses)]
course_names = [f"Course_{i}" for i in range(1, num_courses + 1)]
course_id_name_map = {course_ids[i]: course_names[i] for i in range(num_courses)}  

data = []
start_date = datetime(2024, 1, 1)

for _ in range(num_rows):
    user_id = random.randint(1, num_users)
    course_id = random.choice(course_ids)
    rating = random.randint(1, 5)

    days_offset = random.randint(0, 180)
    date = start_date + timedelta(days=days_offset)
    timestamp = date.strftime('%Y-%m-%d')

    user_name = user_id_name_map[user_id]
    course_name = course_id_name_map[course_id]

    data.append([user_id, user_name, course_id, course_name, rating, timestamp])

df = pd.DataFrame(data, columns=["user_id", "user_name", "course_id", "course_name", "rating", "timestamp"])
df.to_csv("data/courses.csv", index=False)

print("✅ Generated 1000 rows in data/courses.csv with user_name and course_name")
