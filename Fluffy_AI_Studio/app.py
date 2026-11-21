from __future__ import annotations

import base64
import io
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Fluffy AI Studio', page_icon='🧶', layout='wide')
st.markdown('''<style>
.stApp {background:linear-gradient(160deg,#fffaf8,#f8eff5);}
.block-container{max-width:1250px;padding-top:1.7rem;}
[data-testid="stSidebar"]{background:#f4e8ef;}
h1,h2,h3{color:#55374e;}
div.stButton>button[kind="primary"]{background:#79526d;border-color:#79526d;}
.hero{background:linear-gradient(110deg,#efdeea,#fff1e8);border:1px solid #e6d4df;border-radius:22px;padding:24px 30px;margin-bottom:20px;}
.hero h1{margin:0;font-size:2.5rem;}.hero p{color:#6b5766;margin:10px 0 0;}
</style>''',unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading prediction models…')
def prediction_resources():
    import joblib
    import tensorflow as tf
    from sentence_transformers import SentenceTransformer
    likes = tf.keras.models.load_model(ROOT/'models'/'likes_model.keras', compile=False)
    comments = tf.keras.models.load_model(ROOT/'models'/'comments_model.keras', compile=False)
    imputer = joblib.load(ROOT/'preprocessing'/'num_imputer.pkl')
    scaler = joblib.load(ROOT/'preprocessing'/'scaler.pkl')
    encoder = joblib.load(ROOT/'preprocessing'/'encoder.pkl')
    vision = tf.keras.applications.MobileNetV2(weights='imagenet',include_top=False,pooling='avg',input_shape=(224,224,3))
    vision.trainable = False
    text_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return likes, comments, imputer, scaler, encoder, vision, text_model

NUM_COLS = ['followers','caption_length','word_count','hashtag_count','post_hour']
CAT_COLS = ['Type','post_day']

def predict_engagement(photo: Image.Image, caption: str, followers: int, hour: int, post_type: str, day: str):
    import tensorflow as tf
    likes, comments, imputer, scaler, encoder, vision, text_model = prediction_resources()
    image = photo.convert('RGB').resize((224,224))
    arr = np.asarray(image,dtype=np.float32)[None,...]
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    img_vector = vision.predict(arr,verbose=0).astype('float32')
    text_vector = np.asarray(text_model.encode([caption]),dtype='float32')
    row = pd.DataFrame([{
        'followers':followers, 'caption_length':len(caption),
        'word_count':len(caption.split()), 'hashtag_count':caption.count('#'),
        'post_hour':hour, 'Type':post_type, 'post_day':day,
    }])
    numeric = scaler.transform(imputer.transform(row[NUM_COLS])).astype('float32')
    categorical = encoder.transform(row[CAT_COLS].fillna('Missing')).astype('float32')
    inputs = {'image':img_vector,'caption':text_vector,'numerical':numeric,'categorical':categorical}
    results=[]
    for model in (likes,comments):
        log_prediction = float(model.predict(inputs,verbose=0).reshape(-1)[0])
        results.append(max(0,round(float(np.expm1(np.clip(log_prediction,-20,20))))))
    return results


def get_api_key():
    try:
        secret = st.secrets.get('OPENROUTER_API_KEY','')
    except Exception:
        secret = ''
    return secret or os.getenv('OPENROUTER_API_KEY','')


def ask_openrouter(prompt, image=None, model='openrouter/free', timeout=90):
    key = get_api_key()
    if not key:
        raise ValueError('Set OPENROUTER_API_KEY in .streamlit/secrets.toml or an environment variable.')
    content = prompt
    if image is not None:
        image = image.convert('RGB')
        buf=io.BytesIO()
        image.save(buf,format='JPEG',quality=85)
        encoded=base64.b64encode(buf.getvalue()).decode('ascii')
        content=[{'type':'text','text':prompt},{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+encoded}}]
    response=requests.post('https://openrouter.ai/api/v1/chat/completions',headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},json={'model':model,'messages':[{'role':'user','content':content}]},timeout=timeout)
    response.raise_for_status()
    data=response.json()
    return data['choices'][0]['message']['content']


def is_listening(port):
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('127.0.0.1',port))==0

@st.cache_resource(show_spinner=False)
def start_engine():
    source=ROOT/'crochet_engine'
    if not (source/'index.html').exists():
        return None,'3D engine files are missing.'
    port=int(os.getenv('CROCHETPARADE_LOCAL_PORT','8766'))
    if not is_listening(port):
        kwargs={'cwd':str(source),'stdout':subprocess.DEVNULL,'stderr':subprocess.DEVNULL}
        if os.name=='nt':kwargs['creationflags']=getattr(subprocess,'CREATE_NO_WINDOW',0)
        proc=subprocess.Popen([sys.executable,'-m','http.server',str(port),'--bind','127.0.0.1'],**kwargs)
        for _ in range(30):
            if is_listening(port):break
            if proc.poll() is not None:return None,'3D file server exited unexpectedly.'
            time.sleep(0.15)
    if not is_listening(port):return None,'3D file server could not start.'
    return f'http://127.0.0.1:{port}/',None

with st.sidebar:
    st.title('🧶 Fluffy AI Studio')
    st.caption('Your crochet business creative toolkit')
    section=st.radio('Explore', ['🏠 Home','📈 Engagement Predictor','✍️ Caption Generator','📷 Photo Advisor','🧵 3D Crochet Designer'])
    st.divider()
    st.caption('AI estimates are experimental and are not guaranteed engagement outcomes.')

if section=='🏠 Home':
    st.markdown('<div class="hero"><h1>Make it. Style it. Share it.</h1><p>One creative workspace for crochet makers: plan content, explore photography, predict engagement and visualise patterns in 3D.</p></div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        st.subheader('📈 Engagement Predictor')
        st.write('Predict likes and comments for a proposed Instagram post using the photo, caption, account size and posting details.')
        st.subheader('📷 Photo Advisor')
        st.write('Get actionable product-photography suggestions from a vision AI.')
    with b:
        st.subheader('✍️ Caption Generator')
        st.write('Create short crochet captions in your chosen tone and language.')
        st.subheader('🧵 3D Crochet Designer')
        st.write('Use the original CrochetPARADE editor to explore crochet patterns and 3D structures.')

elif section=='📈 Engagement Predictor':
    st.title('📈 Instagram Engagement Predictor')
    st.write('See estimated likes and comments **before publishing** your crochet post.')
    st.info('The prediction models were trained on historical Instagram data. Actual results can differ substantially due to reach, trends and algorithm changes.')
    with st.form('prediction'):
        left,right=st.columns([1,1])
        with left:
            uploaded=st.file_uploader('Upload your post image',type=['png','jpg','jpeg','webp'],key='prediction_image')
            caption=st.text_area('Instagram caption',height=160,placeholder='Handmade with love 🧶✨')
            followers=st.number_input('Follower count',min_value=0,max_value=100000000,value=5000,step=100)
        with right:
            post_type=st.selectbox('Post type',['Carousel','Post','Reel'])
            day=st.selectbox('Posting day',['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
            hour=st.slider('Posting hour (24-hour clock)',0,23,18)
            st.caption('For carousels or Reels, the uploaded image represents a single cover/thumbnail, as in the training data.')
        submitted=st.form_submit_button('✨ Predict engagement',type='primary',use_container_width=True)
    if uploaded:
        st.image(uploaded,caption='Your selected post image',width=260)
    if submitted:
        if not uploaded:
            st.warning('Upload a photo to generate predictions.')
        else:
            try:
                with st.spinner('Extracting image and caption features…'):
                    photo=Image.open(uploaded)
                    likes,comments=predict_engagement(photo,caption,followers,hour,post_type,day)
                a,b=st.columns(2)
                a.metric('❤️ Predicted likes',f'{likes:,}')
                b.metric('💬 Predicted comments',f'{comments:,}')
                st.caption('Rounded, nonnegative model estimates. No uncertainty interval is available from the saved models.')
            except Exception as exc:
                st.error(f'Prediction failed: {exc}')
                st.info('On first run, MobileNetV2 and the multilingual sentence encoder may need to download pretrained weights.')

elif section=='✍️ Caption Generator':
    st.title('✍️ AI Caption Generator')
    st.write('Generate a crochet-product caption using your original OpenRouter feature.')
    with st.form('caption_form'):
        product=st.text_input('Describe your product',placeholder='Pink crochet bunny with a bow')
        tone=st.selectbox('Tone',['Cute and playful','Elegant','Warm and friendly','Minimal','Fun and catchy'])
        language=st.selectbox('Language',['English','Arabic','English and Arabic'])
        price=st.text_input('Price (optional)',placeholder='20 BD')
        go=st.form_submit_button('✨ Generate caption',type='primary')
    if go:
        if not product.strip():st.warning('Describe the product first.')
        else:
            prompt=f'''Generate a creative Instagram caption for a crochet business. Product: {product}. Tone: {tone}. Language: {language}. Price: {price or 'Not specified'}. Keep it short, natural and engaging. Include suitable emojis and do not invent product details. If price is unspecified, do not mention a price.'''
            try:
                with st.spinner('Writing your caption…'):
                    result=ask_openrouter(prompt)
                st.subheader('Your caption')
                st.write(result)
                st.code(result,language=None)
            except Exception as exc:st.error(f'Caption generation failed: {exc}')

elif section=='📷 Photo Advisor':
    st.title('📷 AI Photo Advisor')
    st.write('Get practical photography feedback for your crochet product photos.')
    photo_file=st.file_uploader('Upload a crochet product photo',type=['png','jpg','jpeg','webp'],key='advisor_image')
    if photo_file:st.image(photo_file,width=360)
    if st.button('🔎 Analyze my photo',type='primary'):
        if not photo_file:st.warning('Upload a photo first.')
        else:
            prompt='''You are a product photography advisor for crochet businesses. Analyze this Instagram product photo. Evaluate: (1) background and distractions, (2) lighting and shadows, (3) product positioning and camera angle, (4) human presence and scale, (5) composition and styling. Give 3 to 5 specific actionable recommendations. Explain what you can actually observe. Do not assume people or props are always better. Do not claim any change will increase likes.'''
            try:
                with st.spinner('Analyzing composition and lighting…'):
                    result=ask_openrouter(prompt,Image.open(photo_file),model=os.getenv('OPENROUTER_VISION_MODEL','inclusionai/ling-3.2-flash:free'),timeout=120)
                st.subheader('Photography feedback')
                st.write(result)
            except Exception as exc:
                st.error(f'Photo analysis failed: {exc}')
                st.info('If the free vision model is unavailable, configure OPENROUTER_VISION_MODEL with a vision-capable model available to your account.')

elif section=='🧵 3D Crochet Designer':
    st.title('🧵 3D Crochet Designer')
    st.write('The original CrochetPARADE pattern editor and 3D renderer are preserved from your uploaded ZIP.')
    remote=os.getenv('CROCHETPARADE_URL','').strip()
    if remote:
        if remote.startswith(('https://','http://')):
            st.caption('Using externally hosted CrochetPARADE engine.')
            components.iframe(remote,height=980,scrolling=True)
        else:st.error('CROCHETPARADE_URL must start with https:// or http://')
    else:
        url,error=start_engine()
        if error:st.error(error)
        else:
            st.caption('Local CrochetPARADE editor')
            components.iframe(url,height=980,scrolling=True)
            st.info('For a remote Streamlit deployment, host the crochet_engine directory as a static website and set CROCHETPARADE_URL to its public URL. A browser cannot access the remote server’s localhost iframe.')
    st.caption('CrochetPARADE is an open-source project. Its original license and attribution are retained in crochet_engine/.')
