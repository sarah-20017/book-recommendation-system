import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

BOOKS_DATA = [
    {"id": 1, "title": "The Hobbit", "author": "J.R.R. Tolkien", "genre": "Fantasy", "rating": 4.7,
     "description": "A fantasy adventure", "year": 1937, "pages": 310},
    {"id": 2, "title": "Dune", "author": "Frank Herbert", "genre": "Science Fiction", "rating": 4.6,
     "description": "Epic space opera", "year": 1965, "pages": 682},
    {"id": 3, "title": "Pride and Prejudice", "author": "Jane Austen", "genre": "Romance", "rating": 4.6,
     "description": "Classic romance", "year": 1813, "pages": 279},
    {"id": 4, "title": "The Lord of the Rings", "author": "J.R.R. Tolkien", "genre": "Fantasy", "rating": 4.8,
     "description": "Epic fantasy", "year": 1954, "pages": 1178},
    {"id": 5, "title": "1984", "author": "George Orwell", "genre": "Dystopian", "rating": 4.5,
     "description": "Dystopian novel", "year": 1949, "pages": 328},
    {"id": 6, "title": "To Kill a Mockingbird", "author": "Harper Lee", "genre": "Fiction", "rating": 4.8,
     "description": "Coming-of-age story", "year": 1960, "pages": 324},
    {"id": 7, "title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "genre": "Fiction", "rating": 4.5,
     "description": "Jazz Age tale", "year": 1925, "pages": 180},
    {"id": 8, "title": "Foundation", "author": "Isaac Asimov", "genre": "Science Fiction", "rating": 4.5,
     "description": "Sci-fi epic", "year": 1951, "pages": 255},
    {"id": 9, "title": "Harry Potter", "author": "J.K. Rowling", "genre": "Fantasy", "rating": 4.7,
     "description": "Wizard school", "year": 1997, "pages": 309},
    {"id": 10, "title": "Brave New World", "author": "Aldous Huxley", "genre": "Dystopian", "rating": 4.4,
     "description": "Future society", "year": 1932, "pages": 268},
]


class BookRecommendationEngine:
    def __init__(self, books_data):
        self.books_df = pd.DataFrame(books_data)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        self.prepare_data()

    def prepare_data(self):
        self.books_df['combined_features'] = (
                self.books_df['title'] + ' ' +
                self.books_df['author'] + ' ' +
                self.books_df['genre']
        ).str.lower()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.books_df['combined_features'])

    def get_recommendations(self, book_title, n=5):
        match = self.books_df[self.books_df['title'].str.lower() == book_title.lower()]
        if match.empty:
            return None, f"Book '{book_title}' not found"
        idx = match.index[0]
        scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0]
        similar_idx = np.argsort(scores)[::-1][1:n + 1]
        recs = self.books_df.iloc[similar_idx].copy()
        recs['score'] = scores[similar_idx] * 100
        return recs, None

    def get_popular(self, n=10):
        return self.books_df.nlargest(n, 'rating')

    def get_stats(self):
        return {
            'total': len(self.books_df),
            'avg_rating': self.books_df['rating'].mean(),
            'genres': self.books_df['genre'].nunique(),
        }


if 'engine' not in st.session_state:
    st.session_state.engine = BookRecommendationEngine(BOOKS_DATA)

engine = st.session_state.engine
stats = engine.get_stats()

st.title("📚 Book Recommendation System")

tab1, tab2, tab3 = st.tabs(["Home", "Recommendations", "Popular"])

with tab1:
    st.header("Welcome!")
    col1, col2, col3 = st.columns(3)
    col1.metric("Books", stats['total'])
    col2.metric("Avg Rating", f"{stats['avg_rating']:.2f}")
    col3.metric("Genres", stats['genres'])

with tab2:
    st.header("Get Recommendations")
    book = st.text_input("Enter book title:", placeholder="The Hobbit, Dune, 1984...")
    if st.button("Search"):
        if book:
            recs, error = engine.get_recommendations(book, 5)
            if error:
                st.error(error)
            else:
                for i, (_, b) in enumerate(recs.iterrows(), 1):
                    st.write(f"{i}. **{b['title']}** by {b['author']} ⭐ {b['rating']}")

with tab3:
    st.header("Popular Books")
    for i, (_, b) in enumerate(engine.get_popular(5).iterrows(), 1):
        st.write(f"{i}. **{b['title']}** by {b['author']} ⭐ {b['rating']}")
