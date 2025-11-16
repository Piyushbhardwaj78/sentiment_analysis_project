# Customer Sentiment Analysis from Big Data - Sample Python Project

This is a complete starter project for **Customer Sentiment Analysis** using Python.
It includes:
- Sample dataset (data/sample_reviews.csv)
- Training script (train_model.py) — trains TF-IDF + LogisticRegression and saves a pipeline
- Streamlit demo app (app_streamlit.py) — interactive web UI to analyze single reviews
- Utility cleaning (utils.py)
- Requirements (requirements.txt)
- Short report (report.md)

## How to run (locally)
1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate    # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train_model.py
   ```
   This will create `model/sentiment_pipeline.joblib`.
4. Run the Streamlit app:
   ```bash
   streamlit run app_streamlit.py
   ```
5. Open the app in a browser (Streamlit will show the local URL).

## Notes
- The sample dataset is small; replace `data/sample_reviews.csv` with a larger labelled dataset for better results.
- You can extend the pipeline with data cleaning, class balancing, hyperparameter tuning, or a transformer-based model (BERT).
