# Book Recommendation System

This is a content-based Book Recommendation System built using Python and Machine Learning concepts.

## Project Description
The system recommends books based on textual similarity.  
For learning purposes, a dataset of **50 books was manually created directly inside the code** instead of using an external CSV file.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Streamlit

## Methodology
- Text data is transformed using **TF-IDF Vectorizer**
- Similarity between books is calculated using **Cosine Similarity**
- Users can interact with the system through a **Streamlit web app**

## Features
- Recommends similar books based on user input
- Simple and interactive interface
- Demonstrates core concepts of recommendation systems

## How to Run
```bash
streamlit run app.py