#TRUST IN  GOD WITH ALL YOUR HEART
#BOOK RECOMMENDATION SYSTEM


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

# ------------------ BOOK DATA ------------------
BOOKS_DATA = [
    {"id": 1, "title": "The Hobbit", "author": "J.R.R. Tolkien", "genre": "Fantasy", "rating": 4.7, "description": "A fantasy adventure", "year": 1937, "pages": 310},
    {"id": 2, "title": "Dune", "author": "Frank Herbert", "genre": "Science Fiction", "rating": 4.6, "description": "Epic space opera", "year": 1965, "pages": 682},
    {"id": 3, "title": "Pride and Prejudice", "author": "Jane Austen", "genre": "Romance", "rating": 4.6, "description": "Classic romance", "year": 1813, "pages": 279},
    {"id": 4, "title": "The Lord of the Rings", "author": "J.R.R. Tolkien", "genre": "Fantasy", "rating": 4.8, "description": "Epic fantasy", "year": 1954, "pages": 1178},
    {"id": 5, "title": "1984", "author": "George Orwell", "genre": "Dystopian", "rating": 4.5, "description": "Dystopian society", "year": 1949, "pages": 328},
    {"id": 6, "title": "To Kill a Mockingbird", "author": "Harper Lee", "genre": "Fiction", "rating": 4.8, "description": "Coming-of-age story", "year": 1960, "pages": 324},
    {"id": 7, "title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "genre": "Fiction", "rating": 4.5, "description": "Jazz Age tale", "year": 1925, "pages": 180},
    {"id": 8, "title": "Foundation", "author": "Isaac Asimov", "genre": "Science Fiction", "rating": 4.5, "description": "Sci-fi epic", "year": 1951, "pages": 255},
    {"id": 9, "title": "Harry Potter and the Sorcerer's Stone", "author": "J.K. Rowling", "genre": "Fantasy", "rating": 4.7, "description": "Wizard school", "year": 1997, "pages": 309},
    {"id": 10, "title": "Brave New World", "author": "Aldous Huxley", "genre": "Dystopian", "rating": 4.4, "description": "Future society", "year": 1932, "pages": 268},
    {"id": 11, "title": "Jane Eyre", "author": "Charlotte Brontë", "genre": "Romance", "rating": 4.5, "description": "Classic love story", "year": 1847, "pages": 500},
    {"id": 12, "title": "The Catcher in the Rye", "author": "J.D. Salinger", "genre": "Fiction", "rating": 4.0, "description": "Teen angst story", "year": 1951, "pages": 277},
    {"id": 13, "title": "The Chronicles of Narnia", "author": "C.S. Lewis", "genre": "Fantasy", "rating": 4.6, "description": "Magical adventures", "year": 1950, "pages": 767},
    {"id": 14, "title": "The Da Vinci Code", "author": "Dan Brown", "genre": "Thriller", "rating": 4.2, "description": "Mystery thriller", "year": 2003, "pages": 454},
    {"id": 15, "title": "Angels & Demons", "author": "Dan Brown", "genre": "Thriller", "rating": 4.1, "description": "Secret societies", "year": 2000, "pages": 616},
    {"id": 16, "title": "The Alchemist", "author": "Paulo Coelho", "genre": "Fiction", "rating": 4.6, "description": "Spiritual journey", "year": 1988, "pages": 208},
    {"id": 17, "title": "Moby Dick", "author": "Herman Melville", "genre": "Fiction", "rating": 4.0, "description": "Whale hunting adventure", "year": 1851, "pages": 635},
    {"id": 18, "title": "Wuthering Heights", "author": "Emily Brontë", "genre": "Romance", "rating": 4.3, "description": "Gothic love story", "year": 1847, "pages": 416},
    {"id": 19, "title": "The Shining", "author": "Stephen King", "genre": "Horror", "rating": 4.6, "description": "Haunted hotel thriller", "year": 1977, "pages": 447},
    {"id": 20, "title": "It", "author": "Stephen King", "genre": "Horror", "rating": 4.5, "description": "Evil clown terror", "year": 1986, "pages": 1138},
    {"id": 21, "title": "The Girl with the Dragon Tattoo", "author": "Stieg Larsson", "genre": "Mystery", "rating": 4.3, "description": "Crime investigation", "year": 2005, "pages": 465},
    {"id": 22, "title": "Gone Girl", "author": "Gillian Flynn", "genre": "Thriller", "rating": 4.1, "description": "Psychological thriller", "year": 2012, "pages": 422},
    {"id": 23, "title": "The Hunger Games", "author": "Suzanne Collins", "genre": "Dystopian", "rating": 4.6, "description": "Survival games", "year": 2008, "pages": 374},
    {"id": 24, "title": "Catching Fire", "author": "Suzanne Collins", "genre": "Dystopian", "rating": 4.5, "description": "Survival continues", "year": 2009, "pages": 391},
    {"id": 25, "title": "Mockingjay", "author": "Suzanne Collins", "genre": "Dystopian", "rating": 4.4, "description": "Final battle", "year": 2010, "pages": 390},
    {"id": 26, "title": "The Fault in Our Stars", "author": "John Green", "genre": "Romance", "rating": 4.5, "description": "Teen romance and tragedy", "year": 2012, "pages": 313},
    {"id": 27, "title": "Looking for Alaska", "author": "John Green", "genre": "Romance", "rating": 4.3, "description": "Coming-of-age story", "year": 2005, "pages": 221},
    {"id": 28, "title": "Twilight", "author": "Stephenie Meyer", "genre": "Romance", "rating": 4.1, "description": "Vampire romance", "year": 2005, "pages": 498},
    {"id": 29, "title": "New Moon", "author": "Stephenie Meyer", "genre": "Romance", "rating": 4.0, "description": "Vampire saga continues", "year": 2006, "pages": 563},
    {"id": 30, "title": "Eclipse", "author": "Stephenie Meyer", "genre": "Romance", "rating": 4.1, "description": "Love triangle", "year": 2007, "pages": 629},
    {"id": 31, "title": "Breaking Dawn", "author": "Stephenie Meyer", "genre": "Romance", "rating": 4.0, "description": "Final book in saga", "year": 2008, "pages": 756},
    {"id": 32, "title": "The Kite Runner", "author": "Khaled Hosseini", "genre": "Fiction", "rating": 4.7, "description": "Friendship and betrayal", "year": 2003, "pages": 371},
    {"id": 33, "title": "A Thousand Splendid Suns", "author": "Khaled Hosseini", "genre": "Fiction", "rating": 4.6, "description": "Mother-daughter saga", "year": 2007, "pages": 372},
    {"id": 34, "title": "And Then There Were None", "author": "Agatha Christie", "genre": "Mystery", "rating": 4.6, "description": "Murder mystery", "year": 1939, "pages": 272},
    {"id": 35, "title": "Murder on the Orient Express", "author": "Agatha Christie", "genre": "Mystery", "rating": 4.5, "description": "Famous detective novel", "year": 1934, "pages": 256},
    {"id": 36, "title": "The Secret Garden", "author": "Frances Hodgson Burnett", "genre": "Fiction", "rating": 4.4, "description": "Children's classic", "year": 1911, "pages": 331},
    {"id": 37, "title": "Anne of Green Gables", "author": "L.M. Montgomery", "genre": "Fiction", "rating": 4.5, "description": "Orphan girl adventures", "year": 1908, "pages": 320},
    {"id": 38, "title": "Little Women", "author": "Louisa May Alcott", "genre": "Fiction", "rating": 4.6, "description": "Sisters' story", "year": 1868, "pages": 449},
    {"id": 39, "title": "Frankenstein", "author": "Mary Shelley", "genre": "Horror", "rating": 4.3, "description": "Science and horror", "year": 1818, "pages": 280},
    {"id": 40, "title": "Dracula", "author": "Bram Stoker", "genre": "Horror", "rating": 4.2, "description": "Vampire classic", "year": 1897, "pages": 418},
    {"id": 41, "title": "Les Misérables", "author": "Victor Hugo", "genre": "Fiction", "rating": 4.7, "description": "Revolutionary France", "year": 1862, "pages": 1463},
    {"id": 42, "title": "War and Peace", "author": "Leo Tolstoy", "genre": "Fiction", "rating": 4.6, "description": "Historical epic", "year": 1869, "pages": 1225},
    {"id": 43, "title": "Crime and Punishment", "author": "Fyodor Dostoevsky", "genre": "Fiction", "rating": 4.6, "description": "Psychological novel", "year": 1866, "pages": 671},
    {"id": 44, "title": "The Brothers Karamazov", "author": "Fyodor Dostoevsky", "genre": "Fiction", "rating": 4.7, "description": "Philosophical novel", "year": 1880, "pages": 824},
    {"id": 45, "title": "The Odyssey", "author": "Homer", "genre": "Classic", "rating": 4.5, "description": "Epic Greek poem", "year": -800, "pages": 541},
    {"id": 46, "title": "The Iliad", "author": "Homer", "genre": "Classic", "rating": 4.5, "description": "Trojan War epic", "year": -750, "pages": 704},
    {"id": 47, "title": "Macbeth", "author": "William Shakespeare", "genre": "Classic", "rating": 4.6, "description": "Tragic play", "year": 1606, "pages": 160},
    {"id": 48, "title": "Hamlet", "author": "William Shakespeare", "genre": "Classic", "rating": 4.7, "description": "Tragedy play", "year": 1603, "pages": 200},
    {"id": 49, "title": "Romeo and Juliet", "author": "William Shakespeare", "genre": "Classic", "rating": 4.6, "description": "Romantic tragedy", "year": 1597, "pages": 160},
    {"id": 50, "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle", "genre": "Mystery", "rating": 4.7, "description": "Detective stories", "year": 1892, "pages": 307},

]

# ------------------ ENGINE ------------------
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

# ------------------ STREAMLIT ------------------
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
    selected_genre = st.selectbox(
        "Filter by genre",
        ["All"] + sorted(engine.books_df["genre"].unique().tolist())
    )
    book = st.text_input("Enter book title:", placeholder="The Hobbit, Dune, 1984...")
    if st.button("Search"):
        if book:
            recs, error = engine.get_recommendations(book, 5)
            if error:
                st.error(error)
            else:
                if selected_genre != "All":
                    recs = recs[recs["genre"] == selected_genre]
                for i, (_, b) in enumerate(recs.iterrows(), 1):
                    st.write(f"{i}. **{b['title']}** by {b['author']} ⭐ {b['rating']} ({b['genre']})")

with tab3:
    st.header("Popular Books")
    search = st.text_input("Search by title or author in popular books")
    popular = engine.get_popular(10)
    if search:
        popular = popular[
            popular["title"].str.contains(search, case=False) |
            popular["author"].str.contains(search, case=False)
        ]
    for i, (_, b) in enumerate(popular.iterrows(), 1):
        st.write(f"{i}. **{b['title']}** by {b['author']} ⭐ {b['rating']} ({b['genre']})")


