import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime

st.title('Instagram Likes Predictor')

df = pd.read_csv('lection_updated.csv')
X = df[['post_hour', 'post_day', 'Type']]
y = df['Likes']
X = pd.get_dummies(X, columns=['post_day', 'Type'], dtype=int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

post_date = st.date_input('Post date')
post_time = st.time_input('Post time')
post_type = st.selectbox('Post type', sorted(df['Type'].unique()))

if st.button('Predict Likes'):
    new_post = pd.DataFrame({
        'post_hour': [post_time.hour],
        'post_day': [post_date.strftime('%A')],
        'Type': [post_type]
    })
    new_post = pd.get_dummies(new_post, columns=['post_day', 'Type'], dtype=int)
    new_post = new_post.reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(new_post)[0]
    st.success(f'Predicted likes: {max(0, round(prediction)):,}')
