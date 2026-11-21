# 🧶 Fluffy AI Studio

An integrated Streamlit application built from the uploaded capstone notebook and CrochetPARADE Studio ZIP.

## Features
- Predict **likes and comments** from a post photo, caption, followers, post type, posting day, and posting hour.
- Generate captions using OpenRouter.
- Get AI product-photo advice using an OpenRouter vision model.
- Explore patterns using the **original CrochetPARADE 3D editor**.

## Run locally (Windows, macOS, Linux)
1. Extract the ZIP and open a terminal inside `Fluffy_AI_Studio`.
2. Recommended: Python **3.11** in a virtual environment.
3. Install dependencies: `python -m pip install -r requirements.txt`.
4. To use the two API tools, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, then insert your OpenRouter key. **Never commit or share secrets.toml.** Alternatively, set the `OPENROUTER_API_KEY` environment variable.
5. Start the app: `streamlit run app.py`.
6. Open the URL printed by Streamlit (typically `http://localhost:8501`).

The first prediction can be slow: MobileNetV2 ImageNet weights and the multilingual MiniLM sentence model may download automatically and be cached. Internet is required for that first download. No retraining is performed.

## Prediction pipeline
Exactly follows the final notebook's feature preparation:
- MobileNetV2 (`weights='imagenet'`, `include_top=False`, `pooling='avg'`) → 1280-dimensional image embedding.
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` → 384-dimensional caption embedding.
- `num_imputer.pkl` + `scaler.pkl` on `followers`, `caption_length`, `word_count`, `hashtag_count`, `post_hour`.
- `encoder.pkl` on `Type` and `post_day` (3 + 7 one-hot features).
- Saved Keras model outputs are on the log1p scale; app applies `expm1`, clips to nonnegative, and rounds.

**Important:** `Type='Reel'` and `Type='Carousel'` use a single uploaded cover image as a proxy. Model estimates are experimental and should not be interpreted as guaranteed outcomes. The notebook's comments-evaluation cell uses `y_test` instead of `y_2_test`; that affects reported comments metrics, not the saved comments model's inference.

## 3D engine: local versus hosted
The 3D editor's original JavaScript, WASM, and supporting files are in `crochet_engine/` and are served locally using a background `http.server` on port 8766. This **works for local usage** because your browser and Streamlit server run on the same machine.

For **Streamlit Community Cloud or any remote server**, the local `127.0.0.1` iframe cannot be reached by your browser. Publish `crochet_engine/` to a static web host that supports `.wasm`, allows embedding, and respects the upstream license; then set `CROCHETPARADE_URL` to its public HTTPS URL. The static host should use the correct WASM MIME type. The original upstream project's license is retained in `crochet_engine/LICENSE`.

## API notes
Caption generation uses the notebook's `openrouter/free` routing model. Photo advice uses `inclusionai/ling-3.2-flash:free` by default. Free model availability may change; optionally set `OPENROUTER_VISION_MODEL` in the environment to a supported vision-capable model. API features need a valid OpenRouter key and internet access.

## Security
Never upload `.streamlit/secrets.toml` to GitHub. Saved `.pkl` files should only be loaded from trusted sources (these are the user-supplied artifacts). The app does not persist uploaded photos.
