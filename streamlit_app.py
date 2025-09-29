# COVID-19 Metadata - Assignment
# Solution maker: Morgan Muuo Mutuku
# Single-file Jupyter script that performs:
# Part 1: Data loading and basic exploration
# Part 2: Data cleaning and preparation
# Part 3: Data analysis and visualization
# Part 4: Simple Streamlit app generator


# --------------------
# 0. Setup: imports and installations 
# --------------------
%pip install setuptools

import sys
import subprocess
import pkg_resources

required = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'wordcloud': 'wordcloud',
    'streamlit': 'streamlit',
}

# install missing packages
for pkg, import_name in required.items():
    try:
        pkg_resources.get_distribution(pkg)
    except Exception:
        print(f"Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

# Now import
import os
from collections import Counter
import re
from datetime import datetime

%pip install pandas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)

# Part 1: Data Loading and Basic Exploration

# --------------------
# 1. Load file into pandas DataFrame
# --------------------
DATA_PATH = 'metadata.csv'  

# Try to read; provide a helpful error if not found
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {DATA_PATH}")
except FileNotFoundError:
    raise FileNotFoundError(f"{DATA_PATH} not found. Please place metadata.csv in the working directory or update DATA_PATH.")


# Quick peek
print('\n--- First five rows ---')
display(df.head())

print('\n--- DataFrame shape (rows, columns) ---')
print(df.shape)

print('\n--- DataFrame info ---')
df.info()

print('\n--- Missing values per column ---')
missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_table = pd.concat([missing, missing_pct], axis=1)
missing_table.columns = ['missing_count', 'missing_pct']
display(missing_table)

print('\n--- Basic statistics for numerical columns ---')
display(df.describe(include=[np.number]).T)


# Part 2: Data Cleaning and Preparation
# - Dropping columns with >50% missing values


# --------------------
# helper: find first available column name from common choices
# --------------------

def find_column(df, candidates):
    """Return first column name from candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
        # try lowercase match
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

# Guess common column names
TITLE_COL = find_column(df, ['title', 'Title', 'paper_title'])
ABSTRACT_COL = find_column(df, ['abstract', 'Abstract', 'paper_abstract'])
JOURNAL_COL = find_column(df, ['journal', 'source', 'venue', 'publisher'])
DATE_COL = find_column(df, ['publication_date', 'pub_date', 'date', 'published', 'date_published'])

print('\nDetected columns:')
print('TITLE_COL =', TITLE_COL)
print('ABSTRACT_COL =', ABSTRACT_COL)
print('JOURNAL_COL =', JOURNAL_COL)
print('DATE_COL =', DATE_COL)

# --------------------
# Identify columns with many missing values
# --------------------
THRESHOLD = 0.5  # drop columns with >50% missing
n_rows = len(df)
cols_to_drop = [col for col, pct in (df.isna().sum() / n_rows).items() if pct > THRESHOLD]
print(f"\nColumns with >{int(THRESHOLD*100)}% missing (will drop):", cols_to_drop)

# create a cleaned copy to work on
df_clean = df.copy()
if cols_to_drop:
    df_clean = df_clean.drop(columns=cols_to_drop)

# --------------------
# Convert date columns to datetime
# --------------------
if DATE_COL is not None and DATE_COL in df_clean.columns:
    df_clean[DATE_COL] = pd.to_datetime(df_clean[DATE_COL], errors='coerce')
    # create year
    df_clean['publication_year'] = df_clean[DATE_COL].dt.year
else:
    print('\nNo date column detected: publication_year will be missing. If you have a date column, rename it to one of: publication_date, pub_date, date, published, date_published')

# --------------------
# Fill missing values for remaining columns
# --------------------
for col in df_clean.columns:
    if df_clean[col].isna().sum() == 0:
        continue
    if pd.api.types.is_numeric_dtype(df_clean[col]):
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
    elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
        continue
    else:
        df_clean[col] = df_clean[col].fillna('Unknown')

# --------------------
# Create text-derived features: word counts
# --------------------

def word_count_text(s):
    if pd.isna(s):
        return 0
    # simple tokenization
    tokens = re.findall(r"\w+", str(s))
    return len(tokens)

if TITLE_COL:
    df_clean['title_word_count'] = df_clean[TITLE_COL].apply(word_count_text)
else:
    df_clean['title_word_count'] = 0

if ABSTRACT_COL:
    df_clean['abstract_word_count'] = df_clean[ABSTRACT_COL].apply(word_count_text)
else:
    df_clean['abstract_word_count'] = 0

# Save cleaned dataset
CLEANED_PATH = 'metadata_cleaned.csv'
df_clean.to_csv(CLEANED_PATH, index=False)
print(f"\nCleaned data saved to {CLEANED_PATH}. Shape: {df_clean.shape}")


# Part 3: Data Analysis and Visualization
# We'll create several charts and save them to `figures/`.

# --------------------
# 1) Count papers by publication year
# --------------------
if 'publication_year' in df_clean.columns:
    year_counts = df_clean['publication_year'].value_counts().sort_index()
    print('\nPublication years found (counts):')
    display(year_counts.head(20))

    plt.figure(figsize=(10,5))
    year_counts.plot(kind='line', marker='o')
    plt.title('Number of publications by year')
    plt.xlabel('Year')
    plt.ylabel('Number of papers')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/publications_by_year.png')
    plt.show()
else:
    print('\npublication_year not available — skipping time series plot.')

# --------------------
# 2) Top journals publishing COVID-19 research
# --------------------
if JOURNAL_COL and JOURNAL_COL in df_clean.columns:
    top_journals = df_clean[JOURNAL_COL].value_counts().head(20)
    print('\nTop journals / sources:')
    display(top_journals)

    plt.figure(figsize=(10,6))
    top_journals.sort_values().plot(kind='barh')
    plt.title('Top publishing journals/sources')
    plt.xlabel('Number of papers')
    plt.tight_layout()
    plt.savefig('figures/top_journals.png')
    plt.show()
else:
    print('\nNo journal/source column detected — skipping top journals chart.')

# --------------------
# 3) Most frequent words in titles (simple frequency)
# --------------------
STOPWORDS_SET = set(STOPWORDS)
# Add a few domain-specific stopwords often found in COVID datasets
extra_stop = {'covid', 'covid19', 'covid-19', 'sarscov2', 'study', 'analysis', 'preprint', 'review', 'coronavirus'}
STOPWORDS_SET.update(extra_stop)

if TITLE_COL and TITLE_COL in df_clean.columns:
    # combine titles
    titles = df_clean[TITLE_COL].dropna().astype(str).str.lower()
    all_words = []
    for t in titles:
        tokens = re.findall(r"\w+", t)
        tokens = [tok for tok in tokens if tok not in STOPWORDS_SET and len(tok) > 2]
        all_words.extend(tokens)

    counter = Counter(all_words)
    most_common = counter.most_common(30)
    print('\nMost common words in titles:')
    display(pd.DataFrame(most_common, columns=['word', 'count']))

    # Barplot for top 20
    top_words = pd.DataFrame(most_common[:20], columns=['word', 'count']).set_index('word')
    plt.figure(figsize=(10,6))
    top_words.sort_values('count').plot(kind='barh', legend=False)
    plt.title('Top words in titles')
    plt.xlabel('Frequency')
    plt.tight_layout()
    plt.savefig('figures/top_title_words.png')
    plt.show()

    # Word cloud
    wc_text = ' '.join(all_words)
    if wc_text.strip():
        wc = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS_SET).generate(wc_text)
        plt.figure(figsize=(12,6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Titles')
        plt.tight_layout()
        plt.savefig('figures/wordcloud_titles.png')
        plt.show()
    else:
        print('Not enough title text for a word cloud.')
else:
    print('\nNo title column detected — skipping title frequency and word cloud.')

# --------------------
# 4) Distribution of paper counts by source (if available)
# --------------------
if JOURNAL_COL and JOURNAL_COL in df_clean.columns:
    source_counts = df_clean[JOURNAL_COL].value_counts()
    plt.figure(figsize=(10,5))
    # plot top 40 or all if fewer
    top_n = min(40, len(source_counts))
    source_counts.iloc[:top_n].plot(kind='bar')
    plt.title('Paper counts by source (top {})'.format(top_n))
    plt.xlabel('Source')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('figures/paper_counts_by_source.png')
    plt.show()


# Part 4: Streamlit Application
# We'll create a `streamlit_app.py` next to this notebook. The app reads the cleaned CSV and offers
# interactive filters and displays the charts and a data sample.

# Create streamlit app file
#streamlit_code = r'''

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from collections import Counter

st.set_page_config (page_title='COVID-19 Metadata Explorer', layout='wide')

st.title('COVID-19 Metadata Explorer')

st.markdown('''**Solution maker:** Morgan Muuo Mutuku''')

DATA_PATH = 'metadata_cleaned.csv'

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Could not load cleaned data at {DATA_PATH}. Run the Jupyter script first to produce it. Error: {e}")
    st.stop()

# widgets
min_year = int(df['publication_year'].min()) if 'publication_year' in df.columns and not df['publication_year'].isna().all() else 0
max_year = int(df['publication_year'].max()) if 'publication_year' in df.columns and not df['publication_year'].isna().all() else 0

if min_year and max_year:
    year_slider = st.slider('Publication year', min_value=min_year, max_value=max_year, value=(min_year, max_year))
    df = df[(df['publication_year'] >= year_slider[0]) & (df['publication_year'] <= year_slider[1])]

journal_col_candidates = ['journal', 'source', 'publisher', 'venue']
journal_col = None
for c in journal_col_candidates:
    if c in df.columns:
        journal_col = c
        break

if journal_col:
    journals = ['All'] + sorted(df[journal_col].dropna().unique().tolist())
    selected_journal = st.selectbox('Filter by journal/source', journals)
    if selected_journal and selected_journal != 'All':
        df = df[df[journal_col] == selected_journal]

# Show counts
st.subheader('Summary')
col1, col2 = st.columns(2)
with col1:
    st.metric('Papers (filtered)', len(df))
with col2:
    if 'publication_year' in df.columns:
        st.metric('Year range', f"{int(df['publication_year'].min())} - {int(df['publication_year'].max())}")

# Charts
st.subheader('Publications by year')
if 'publication_year' in df.columns:
    year_counts = df['publication_year'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8,4))
    year_counts.plot(kind='line', marker='o', ax=ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of papers')
    st.pyplot(fig)
else:
    st.info('No publication_year column available.')

# Top journals
if journal_col:
    st.subheader('Top journals / sources (filtered)')
    top_j = df[journal_col].value_counts().head(20)
    fig2, ax2 = plt.subplots(figsize=(8,5))
    top_j.sort_values().plot(kind='barh', ax=ax2)
    ax2.set_xlabel('Number of papers')
    st.pyplot(fig2)

# Word cloud for titles
st.subheader('Word cloud of titles')
TITLE_COL = None
for c in ['title', 'Title', 'paper_title']:
    if c in df.columns:
        TITLE_COL = c
        break

if TITLE_COL:
    titles = df[TITLE_COL].dropna().astype(str).str.lower()
    STOPWORDS_SET = set(STOPWORDS)
    extra = {'covid','covid-19','covid19','sarscov2','study','analysis','coronavirus'}
    STOPWORDS_SET.update(extra)
    all_words = []
    for t in titles:
        tokens = re.findall(r"\w+", t)
        tokens = [tok for tok in tokens if tok not in STOPWORDS_SET and len(tok) > 2]
        all_words.extend(tokens)
    wc_text = ' '.join(all_words)
    if wc_text.strip():
        wc = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS_SET).generate(wc_text)
        fig3, ax3 = plt.subplots(figsize=(10,5))
        ax3.imshow(wc, interpolation='bilinear')
        ax3.axis('off')
        st.pyplot(fig3)
    else:
        st.info('Not enough text for a word cloud.')

# Data sample
st.subheader('Data sample (first 200 rows)')
st.dataframe(df.head(200))


with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(streamlit_code)

print('\nStreamlit app file written to streamlit_app.py')
print('Run it in a terminal with: streamlit run streamlit_app.py')

print('\nAll done. Figures saved in the `figures/` folder. Cleaned data saved to metadata_cleaned.csv.\n') 


