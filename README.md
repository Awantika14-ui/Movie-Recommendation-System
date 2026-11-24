# Movie Recommendation System

A content-based Movie Recommendation System built using Python and Streamlit that suggests similar movies based on the user's selected movie.

## Features

- Select a movie from the available movie collection.
- Get the top 10 similar movie recommendations.
- Uses TF-IDF vectorization to process movie information.
- Uses cosine similarity to identify similar movies.
- Fetches movie posters using the TMDB API.
- Simple and interactive Streamlit interface.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Requests
- TMDB API

## Recommendation Approach

This project uses content-based filtering.

1. Movie information such as genres, keywords, cast, and director is processed.
2. TF-IDF converts movie descriptions into numerical features.
3. Cosine similarity measures the similarity between movies.
4. The system displays the top 10 similar movies.

## Project Structure

```text
Movie-Recommendation-System/
│
├── app.py
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
├── requirements.txt
├── .gitignore
├── README.md
└── .streamlit/
    └── secrets.toml (not uploaded)
```

## Installation and Setup

1. Clone this repository:

```bash
git clone https://github.com/Awantika14-ui/Movie-Recommendation-System.git
```

2. Navigate to the project folder:

```bash
cd Movie-Recommendation-System
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Create a `.streamlit/secrets.toml` file and add your TMDB API key:

```toml
TMDB_API_KEY = "YOUR_TMDB_API_KEY"
```

5. Run the application:

```bash
streamlit run app.py
```

## Dataset

The project uses the TMDB 5000 Movies and Credits datasets.

## API Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.

## Purpose

Developed as a learning and portfolio project to explore Python, machine learning concepts, content-based filtering, and interactive web applications.