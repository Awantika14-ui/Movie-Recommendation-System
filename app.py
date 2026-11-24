import streamlit as st
import pandas as pd
import requests
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load and process movie datasets
@st.cache_data
def load_movies():
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = pd.read_csv("tmdb_5000_movies.csv")

    movies = movies.merge(credits, on="title")

    movies = movies[
        ["movie_id", "title", "overview", "genres",
         "keywords", "cast", "crew"]
    ]

    def extract_names(value):
        try:
            return [item["name"] for item in ast.literal_eval(value)]
        except (ValueError, TypeError):
            return []

    movies["genres"] = movies["genres"].apply(extract_names)
    movies["keywords"] = movies["keywords"].apply(extract_names)

    movies["cast"] = movies["cast"].apply(
        lambda x: extract_names(x)[:3]
    )

    movies["crew"] = movies["crew"].apply(
        lambda x: [
            item["name"]
            for item in ast.literal_eval(x)
            if item.get("job") == "Director"
        ] if isinstance(x, str) else []
    )

    movies["tags"] = (
        movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    movies["tags"] = movies["tags"].apply(
        lambda x: " ".join(x)
    )

    return movies[["movie_id", "title", "overview", "tags"]]


# Build TF-IDF model
@st.cache_resource
def build_model(tags):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(tags)
    return tfidf_matrix


movies = load_movies()
tfidf_matrix = build_model(movies["tags"])

# Recommend similar movies
def get_recommendations(title):
    matches = movies.index[movies["title"] == title]

    if len(matches) == 0:
        return pd.DataFrame()

    idx = matches[0]

    similarity_scores = linear_kernel(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[-11:-1][::-1]

    return movies.iloc[similar_indices][["title", "movie_id"]]



# Fetch movie poster from TMDB
def fetch_poster(movie_id):
    try:
        api_key = st.secrets["TMDB_API_KEY"]
    except Exception:
        return None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    try:
        response = requests.get(
            url,
            params={"api_key": api_key},
            timeout=10
        )
        response.raise_for_status()

        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except requests.RequestException:
        return None

    return None


# Streamlit interface
st.title("Movie Recommendation System")
st.write("Discover movies similar to your favourite films!")

selected_movie = st.selectbox(
    "Select a movie:",
    movies["title"].dropna().unique()
)

if st.button("Recommend"):
    recommendations = get_recommendations(selected_movie)

    if recommendations.empty:
        st.warning("No recommendations found.")
    else:
        st.subheader("Top 10 Recommended Movies")

        for i in range(0, len(recommendations), 5):
            cols = st.columns(5)

            for col, (_, movie) in zip(
                cols,
                recommendations.iloc[i:i + 5].iterrows()
            ):
                with col:
                    poster_url = fetch_poster(movie["movie_id"])

                    if poster_url:
                        st.image(poster_url, width=130)
                    else:
                        st.caption("Poster unavailable")

                    st.write(movie["title"])
