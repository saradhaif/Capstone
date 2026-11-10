import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title('Instagram Likes Predictor')

df = pd.read_csv('lection_updated.csv')
X = pd.get_dummies(df[['post_day', 'Type']], dtype=int)
y = df['Likes']

model = LinearRegression()
model.fit(X, y)

post_date = st.date_input('Post date')
post_type = st.selectbox('Post type', ['Carousel', 'Reel', 'Post'])

if st.button('Predict Likes'):
    new_post = pd.DataFrame({'post_day': [post_date.strftime('%A')], 'Type': [post_type]})
    new_X = pd.get_dummies(new_post, dtype=int).reindex(columns=X.columns, fill_value=0)
    prediction = model.predict(new_X)[0]
    st.success(f'Predicted likes: {max(0, round(prediction)):,}')
