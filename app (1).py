import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Instagram Likes Predictor")

df = pd.read_csv("lection_updated.csv")
df["datetime"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert("Asia/Bahrain")
df["post_hour"] = df["datetime"].dt.hour
df["post_day"] = df["datetime"].dt.day_name()

X = df[["post_hour", "post_day", "Type"]]
y = df["Likes"]
X = pd.get_dummies(X, columns=["post_day", "Type"], dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

post_hour = st.selectbox("Post hour", range(24))
post_day = st.selectbox("Post day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
post_type = st.selectbox("Post type", sorted(df["Type"].dropna().unique()))

if st.button("Predict Likes"):
    new_post = pd.DataFrame({"post_hour": [post_hour], "post_day": [post_day], "Type": [post_type]})
    new_post = pd.get_dummies(new_post, columns=["post_day", "Type"], dtype=int)
    new_post = new_post.reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(new_post)[0]
    st.success(f"Predicted likes: {max(0, round(prediction))}")
