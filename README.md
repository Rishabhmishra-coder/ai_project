# ==== Cell 1 ====
# Import Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
NETFLIX_RED = '#E50914'
NETFLIX_BLACK = '#221f1f'


# ==== Cell 2 ====
# Load Dataset
df = pd.read_csv('netflix_titles.csv')


# ==== Cell 3 ====
# Dataset First Look
df.head()


# ==== Cell 4 ====
# Dataset Rows & Columns count
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# ==== Cell 5 ====
# Dataset Info
df.info()


# ==== Cell 6 ====
# Dataset Duplicate Value Count
print("Fully duplicate rows:", df.duplicated().sum())
print("Duplicate titles:", df['title'].duplicated().sum())
print("Duplicate show_id:", df['show_id'].duplicated().sum())


# ==== Cell 7 ====
# Missing Values/Null Values Count
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct (%)': missing_pct})
missing_df[missing_df.missing_count > 0]


# ==== Cell 8 ====
# Visualizing the missing values
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='rocket')
plt.title('Missing Value Map')
plt.show()


# ==== Cell 9 ====
# Dataset Columns
df.columns.tolist()


# ==== Cell 10 ====
# Dataset Describe
df.describe(include='all').T


# ==== Cell 11 ====
# Check Unique Values for each variable.
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")


# ==== Cell 12 ====
# Write your code to make your dataset analysis ready.

def clean_data(data):
    data = data.copy()
    # Categorical/text columns: missing usually means "not credited", not an error -> fill 'Unknown'
    for col in ['director', 'cast', 'country']:
        data[col] = data[col].fillna('Unknown')
        data[col] = data[col].replace('', 'Unknown')
    # Rating: only 7 missing rows -> impute with the mode
    data['rating'] = data['rating'].fillna(data['rating'].mode()[0])
    # date_added: only 10 missing rows -> safe to drop
    data = data.dropna(subset=['date_added'])
    # Remove any exact duplicate rows
    data = data.drop_duplicates()
    return data.reset_index(drop=True)


def engineer_features(data):
    data = data.copy()
    # Parse the date Netflix added the title
    data['date_added_parsed'] = pd.to_datetime(data['date_added'].str.strip(), errors='coerce')
    data['year_added'] = data['date_added_parsed'].dt.year
    data['month_added'] = data['date_added_parsed'].dt.month

    # Split the mixed-unit `duration` column into a type flag + numeric value
    data['duration_type'] = np.where(data['duration'].str.contains('Season', case=False, na=False),
                                      'Season(s)', 'Minutes')
    data['duration_int'] = data['duration'].str.extract(r'(\d+)').astype(float)

    # First-listed country / genre, useful for grouping and plotting
    data['primary_country'] = data['country'].apply(lambda x: str(x).split(',')[0].strip())
    data['primary_genre'] = data['listed_in'].apply(lambda x: str(x).split(',')[0].strip())

    # How many years after release Netflix added the title (content "freshness"); clip negative
    # values (a handful of data-entry quirks where release_year > year_added) at 0
    data['content_age_at_add'] = (data['year_added'] - data['release_year']).clip(lower=0)

    return data

df = clean_data(df)
df = engineer_features(df)
print("Shape after cleaning + feature engineering:", df.shape)
df[['title', 'duration_type', 'duration_int', 'primary_country', 'primary_genre', 'content_age_at_add']].head()


# ==== Cell 13 ====
# Chart - 1 visualization code
type_counts = df['type'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
        colors=[NETFLIX_RED, NETFLIX_BLACK], explode=(0.03, 0.03), startangle=90)
plt.title('Share of Content: Movies vs TV Shows')
plt.show()
print(type_counts)


# ==== Cell 14 ====
# Chart - 2 visualization code
trend = df.groupby(['year_added', 'type']).size().unstack(fill_value=0)

trend.plot(kind='line', marker='o', color=[NETFLIX_RED, NETFLIX_BLACK], figsize=(10, 5))
plt.title('Titles Added to Netflix Per Year, by Type')
plt.ylabel('Number of Titles Added')
plt.xlabel('Year Added')
plt.show()

tv_share = (trend['TV Show'] / (trend['Movie'] + trend['TV Show'])).round(3)
print("TV Show share of additions by year:\n", tv_share)


# ==== Cell 15 ====
# Chart - 3 visualization code
country_series = df['country'].str.split(',').explode().str.strip()
top_countries = country_series[~country_series.isin(['', 'Unknown'])].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='Reds_r')
plt.title('Top 10 Countries Producing Netflix Content')
plt.xlabel('Number of Titles')
plt.show()
print(top_countries)


# ==== Cell 16 ====
# Chart - 4 visualization code
genre_series = df['listed_in'].str.split(',').explode().str.strip()
top_genres = genre_series.value_counts().head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='mako')
plt.title('Top 15 Genres on Netflix')
plt.xlabel('Number of Titles')
plt.show()
print(top_genres)


# ==== Cell 17 ====
# Chart - 5 visualization code
plt.figure(figsize=(10, 5))
order = df['rating'].value_counts().index
sns.countplot(data=df, y='rating', order=order, palette='rocket')
plt.title('Distribution of Content Ratings')
plt.xlabel('Number of Titles')
plt.show()
print(df['rating'].value_counts())


# ==== Cell 18 ====
# Chart - 6 visualization code
movie_durations = df[df['type'] == 'Movie']['duration_int'].dropna()

plt.figure(figsize=(9, 5))
sns.histplot(movie_durations, bins=30, color=NETFLIX_RED, kde=True)
plt.title('Distribution of Movie Duration (minutes)')
plt.xlabel('Duration (minutes)')
plt.show()
print(movie_durations.describe())


# ==== Cell 19 ====
# Chart - 7 visualization code
tv_seasons = df[df['type'] == 'TV Show']['duration_int'].dropna()

plt.figure(figsize=(9, 5))
sns.countplot(x=tv_seasons, color=NETFLIX_BLACK)
plt.title('Number of Seasons per TV Show')
plt.xlabel('Seasons')
plt.show()
print(tv_seasons.value_counts().sort_index())


# ==== Cell 20 ====
# Chart - 8 visualization code
top_directors = df[df['director'] != 'Unknown']['director'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_directors.values, y=top_directors.index, palette='Reds_r')
plt.title('Top 10 Directors by Number of Titles on Netflix')
plt.xlabel('Number of Titles')
plt.show()
print(top_directors)


# ==== Cell 21 ====
# Chart - 9 visualization code
cast_series = df[df['cast'] != 'Unknown']['cast'].str.split(',').explode().str.strip()
top_cast = cast_series.value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_cast.values, y=top_cast.index, palette='mako')
plt.title('Top 10 Most Frequently Credited Cast Members')
plt.xlabel('Number of Titles')
plt.show()
print(top_cast)


# ==== Cell 22 ====
# Chart - 10 visualization code
plt.figure(figsize=(9, 5))
sns.histplot(df['release_year'], bins=30, color=NETFLIX_RED)
plt.title('Distribution of Release Year')
plt.xlabel('Release Year')
plt.show()
print(df['release_year'].describe())


# ==== Cell 23 ====
# Chart - 11 visualization code
month_counts = df['month_added'].value_counts().sort_index()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure(figsize=(10, 5))
sns.barplot(x=[month_names[int(m)-1] for m in month_counts.index], y=month_counts.values, color=NETFLIX_RED)
plt.title('Titles Added to Netflix by Month (All Years Combined)')
plt.ylabel('Number of Titles Added')
plt.show()
print(month_counts)


# ==== Cell 24 ====
# Chart - 12 visualization code
top5_countries = df['primary_country'].value_counts().head(5).index
subset = df[df['primary_country'].isin(top5_countries)]
ct = pd.crosstab(subset['primary_country'], subset['type'], normalize='index')

ct.plot(kind='bar', stacked=True, color=[NETFLIX_RED, NETFLIX_BLACK], figsize=(9, 5))
plt.title('Movie vs TV Show Mix in Top 5 Countries')
plt.ylabel('Proportion of Titles')
plt.xticks(rotation=0)
plt.show()
print(ct.round(3))


# ==== Cell 25 ====
# Chart - 13 visualization code
rating_type = pd.crosstab(df['rating'], df['type'])

rating_type.plot(kind='bar', color=[NETFLIX_RED, NETFLIX_BLACK], figsize=(11, 5))
plt.title('Content Rating by Type (Movie vs TV Show)')
plt.ylabel('Number of Titles')
plt.xticks(rotation=45)
plt.show()
print(rating_type)


# ==== Cell 26 ====
# Correlation Heatmap visualization code
numeric_cols = ['release_year', 'year_added', 'month_added', 'duration_int', 'content_age_at_add']
corr = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()


# ==== Cell 27 ====
# Pair Plot visualization code
sample_df = df[numeric_cols + ['type']].dropna().sample(n=min(1000, len(df)), random_state=42)

sns.pairplot(sample_df, hue='type', palette=[NETFLIX_RED, NETFLIX_BLACK], diag_kind='hist', plot_kws={'alpha': 0.5, 's': 15})
plt.suptitle('Pair Plot of Numeric Features by Content Type', y=1.02)
plt.show()


# ==== Cell 28 ====
# Perform Statistical Test to obtain P-Value

df_h1 = df.dropna(subset=['year_added']).copy()
df_h1['period'] = np.where(df_h1['year_added'] >= 2019, 'Recent (2019-2021)', 'Earlier (<=2018)')

contingency_1 = pd.crosstab(df_h1['period'], df_h1['type'])
print(contingency_1)

chi2_1, p_1, dof_1, expected_1 = stats.chi2_contingency(contingency_1)
print(f"\nChi-square statistic: {chi2_1:.3f}")
print(f"Degrees of freedom: {dof_1}")
print(f"P-value: {p_1:.5f}")

alpha = 0.05
if p_1 < alpha:
    print(f"\nSince p-value ({p_1:.5f}) < alpha ({alpha}), we REJECT the null hypothesis.")
else:
    print(f"\nSince p-value ({p_1:.5f}) >= alpha ({alpha}), we FAIL TO REJECT the null hypothesis.")


# ==== Cell 29 ====
# Perform Statistical Test to obtain P-Value

top5_countries = df['primary_country'].value_counts().head(5).index
df_h2 = df[df['primary_country'].isin(top5_countries)].copy()
df_h2['mature'] = df_h2['rating'].isin(['TV-MA', 'R', 'NC-17']).map({True: 'Mature', False: 'Not Mature'})

contingency_2 = pd.crosstab(df_h2['primary_country'], df_h2['mature'])
print(contingency_2)

chi2_2, p_2, dof_2, expected_2 = stats.chi2_contingency(contingency_2)
print(f"\nChi-square statistic: {chi2_2:.3f}")
print(f"Degrees of freedom: {dof_2}")
print(f"P-value: {p_2:.2e}")

if p_2 < alpha:
    print(f"\nSince p-value ({p_2:.2e}) < alpha ({alpha}), we REJECT the null hypothesis.")
else:
    print(f"\nSince p-value ({p_2:.2e}) >= alpha ({alpha}), we FAIL TO REJECT the null hypothesis.")


# ==== Cell 30 ====
# Perform Statistical Test to obtain P-Value

movies_h3 = df[df['type'] == 'Movie'].copy()
top_ratings = movies_h3['rating'].value_counts().head(4).index
groups = [movies_h3[movies_h3['rating'] == r]['duration_int'].dropna() for r in top_ratings]

f_stat, p_3 = stats.f_oneway(*groups)
print("Ratings compared:", list(top_ratings))
for r, g in zip(top_ratings, groups):
    print(f"  {r}: mean duration = {g.mean():.1f} min (n={g.count()})")

print(f"\nF-statistic: {f_stat:.3f}")
print(f"P-value: {p_3:.2e}")

if p_3 < alpha:
    print(f"\nSince p-value ({p_3:.2e}) < alpha ({alpha}), we REJECT the null hypothesis.")
else:
    print(f"\nSince p-value ({p_3:.2e}) >= alpha ({alpha}), we FAIL TO REJECT the null hypothesis.")


# ==== Cell 31 ====
# Handling Missing Values & Missing Value Imputation

def clean_data(data):
    data = data.copy()
    for col in ['director', 'cast', 'country']:
        data[col] = data[col].fillna('Unknown')
        data[col] = data[col].replace('', 'Unknown')
    data['rating'] = data['rating'].fillna(data['rating'].mode()[0])
    data = data.dropna(subset=['date_added'])
    data = data.drop_duplicates()
    return data.reset_index(drop=True)

df = clean_data(df)
print("Shape after cleaning:", df.shape)
print(df.isnull().sum())


# ==== Cell 32 ====
# Handling Outliers & Outlier treatments

def engineer_features(data):
    data = data.copy()
    data['date_added_parsed'] = pd.to_datetime(data['date_added'].str.strip(), errors='coerce')
    data['year_added'] = data['date_added_parsed'].dt.year
    data['month_added'] = data['date_added_parsed'].dt.month
    data['duration_type'] = np.where(data['duration'].str.contains('Season', case=False, na=False),
                                      'Season(s)', 'Minutes')
    data['duration_int'] = data['duration'].str.extract(r'(\d+)').astype(float)
    data['primary_country'] = data['country'].apply(lambda x: str(x).split(',')[0].strip())
    data['primary_genre'] = data['listed_in'].apply(lambda x: str(x).split(',')[0].strip())
    # Clip a handful of negative content-age values (release_year briefly exceeding year_added
    # due to pre-release/press-screening dates) at 0 rather than dropping those rows
    data['content_age_at_add'] = (data['year_added'] - data['release_year']).clip(lower=0)
    return data

df = engineer_features(df)

# Visual outlier check on release_year and duration_int
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x=df['release_year'], ax=axes[0], color=NETFLIX_RED)
axes[0].set_title('Release Year - Outlier Check')
sns.boxplot(x=df[df['type']=='Movie']['duration_int'], ax=axes[1], color=NETFLIX_BLACK)
axes[1].set_title('Movie Duration - Outlier Check')
plt.tight_layout()
plt.show()


# ==== Cell 33 ====
# Encode your categorical columns

from sklearn.preprocessing import LabelEncoder

# `type` is a simple binary categorical column used later for cluster-profiling / cross-tabs;
# label-encode it for any numeric summary that needs it (clustering itself uses text features, see below)
le_type = LabelEncoder()
df['type_encoded'] = le_type.fit_transform(df['type'])
print(dict(zip(le_type.classes_, le_type.transform(le_type.classes_))))


# ==== Cell 34 ====
# Expand Contraction

# NOTE: the `contractions` package is not available in this offline environment, so a small
# manual contraction map is used instead — sufficient for short movie/TV descriptions.
CONTRACTIONS_MAP = {
    "don't": "do not", "can't": "cannot", "won't": "will not", "it's": "it is",
    "i'm": "i am", "he's": "he is", "she's": "she is", "that's": "that is",
    "there's": "there is", "what's": "what is", "n't": " not", "'re": " are",
    "'s": " is", "'d": " would", "'ll": " will", "'ve": " have",
}

def expand_contractions(text):
    text = text.lower()
    for k, v in CONTRACTIONS_MAP.items():
        text = text.replace(k, v)
    return text

df['soup_raw'] = (
    df['listed_in'].astype(str) + ' ' +
    df['description'].astype(str) + ' ' +
    df['director'].astype(str) + ' ' +
    df['type'].astype(str) + ' ' +
    df['cast'].astype(str).apply(lambda x: ' '.join(x.split(',')[:3]))  # top 3 cast members only
)

df['soup_step1'] = df['soup_raw'].apply(expand_contractions)
df[['title', 'soup_step1']].head(3)


# ==== Cell 35 ====
# Lower Casing
# (lower-casing is already folded into expand_contractions() above, since contraction
# matching itself needs lowercase text -- shown here again explicitly for clarity)
df['soup_step2'] = df['soup_step1'].str.lower()
df[['title', 'soup_step2']].head(3)


# ==== Cell 36 ====
# Remove Punctuations
import re

def remove_punctuation(text):
    return re.sub(r'[^a-z0-9\s]', ' ', text)

df['soup_step3'] = df['soup_step2'].apply(remove_punctuation)
df[['title', 'soup_step3']].head(3)


# ==== Cell 37 ====
# Remove URLs & Remove words and digits contain digits

def remove_urls_and_digits(text):
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\b\w*\d\w*\b', ' ', text)   # drop any token containing a digit
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['soup_step4'] = df['soup_step3'].apply(remove_urls_and_digits)
df[['title', 'soup_step4']].head(3)


# ==== Cell 38 ====
# Remove Stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)

def remove_stopwords(text):
    return ' '.join(w for w in text.split() if w not in STOPWORDS and len(w) > 2)

df['soup_step5'] = df['soup_step4'].apply(remove_stopwords)
df[['title', 'soup_step5']].head(3)


# ==== Cell 39 ====
# Remove White spaces
df['soup_step5'] = df['soup_step5'].str.strip().str.replace(r'\s+', ' ', regex=True)


# ==== Cell 40 ====
# Rephrase Text
# No further rephrasing/spelling-correction step is applied -- movie/TV synopses in this dataset
# are already professionally written, well-formed English, so this step is a no-op here.
df['soup_step6'] = df['soup_step5']


# ==== Cell 41 ====
# Tokenization
df['tokens'] = df['soup_step6'].apply(str.split)
df[['title', 'tokens']].head(3)


# ==== Cell 42 ====
# Normalizing Text (i.e., Stemming, Lemmatization etc.)

# NOTE: NLTK is not available in this offline environment, so a small rule-based
# suffix-stripping stemmer is used as a lightweight substitute for a proper Porter/Snowball stemmer.
def simple_stem(word):
    for suf in ['ational', 'tional', 'edly', 'ing', 'ies', 'ied', 'ly', 'ed', 'es', 's']:
        if word.endswith(suf) and len(word) - len(suf) > 3:
            return word[:-len(suf)]
    return word

df['tokens_stemmed'] = df['tokens'].apply(lambda toks: [simple_stem(t) for t in toks])
df['soup_clean'] = df['tokens_stemmed'].apply(' '.join)
df[['title', 'soup_clean']].head(3)


# ==== Cell 43 ====
# POS Taging
# A full POS-tagging step is skipped: it typically feeds into lemmatization (choosing the right
# lemma per part of speech), but since NLTK's tagger/lemmatizer aren't available here, POS tags
# would have no downstream use in this pipeline. TF-IDF vectorization below treats all remaining
# tokens as a flat bag/n-gram of words regardless of part of speech.
print("POS tagging skipped -- not required for the TF-IDF vectorization pipeline used below.")


# ==== Cell 44 ====
# Vectorizing Text
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['soup_clean'])
print("TF-IDF matrix shape:", tfidf_matrix.shape)


# ==== Cell 45 ====
# Manipulate Features to minimize feature correlation and create new features

# Combine the cleaned text signal (soup_clean / tfidf_matrix) with a couple of numeric/categorical
# features that are informative but shouldn't dominate the (very high-dimensional) text signal.
# We keep the numeric side lightweight: content_age_at_add and duration_int summarize metadata
# without duplicating information already present in the text (release_year is highly correlated
# with content_age_at_add, so only one of the pair is kept -- see the Chart 14 correlation heatmap).
print("Numeric side-features considered: duration_int, content_age_at_add")
print(df[['duration_int', 'content_age_at_add']].describe())


# ==== Cell 46 ====
# Select your features wisely to avoid overfitting

# The primary clustering feature space is the TF-IDF matrix (already computed above).
# We reduce it with TruncatedSVD below rather than hand-picking a subset of TF-IDF columns,
# since SVD keeps the components that explain the most variance across ALL 5,000 terms at once
# (a data-driven form of feature selection for sparse text features).
FEATURES_FOR_CLUSTERING = "TF-IDF matrix (5000 terms) -> reduced via TruncatedSVD"
print(FEATURES_FOR_CLUSTERING)


# ==== Cell 47 ====
# Transform Your data
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=50, random_state=42)
reduced_features = svd.fit_transform(tfidf_matrix)
print("Reduced shape:", reduced_features.shape)
print(f"Explained variance (50 components): {svd.explained_variance_ratio_.sum():.2%}")


# ==== Cell 48 ====
# Scaling your data
from sklearn.preprocessing import Normalizer

# L2-normalize each row of the SVD-reduced matrix. This is the classic "spherical k-means"
# trick for text: after normalizing, Euclidean distance between rows becomes proportional to
# cosine distance, which is the natural similarity measure for TF-IDF/SVD text vectors.
normalizer = Normalizer(copy=False)
X = normalizer.fit_transform(reduced_features)
print("Final feature matrix for clustering:", X.shape)


# ==== Cell 49 ====
# Dimensionality Reduction (If needed)
# (Already performed above via TruncatedSVD -- shown again here for template completeness.)
print("Dimensionality reduction already applied: TF-IDF (5000-dim, sparse) -> TruncatedSVD (50-dim, dense)")
print(f"Variance explained by 50 SVD components: {svd.explained_variance_ratio_.sum():.2%}")


# ==== Cell 50 ====
# Split your data to train and test. Choose Splitting ratio wisely.

# This is an UNSUPERVISED clustering problem -- there is no target/label column to predict,
# so a conventional train/test split is not applicable here (see the written answer below).
print("No train/test split performed: unsupervised clustering has no ground-truth label to hold out.")


# ==== Cell 51 ====
# Handling Imbalanced Dataset (If needed)
# Not applicable: this is an unsupervised clustering task with no target labels to balance.
print("Not applicable for this unsupervised clustering task.")


# ==== Cell 52 ====
# ML Model - 1 Implementation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Fit the Algorithm
# First, use the Elbow method + Silhouette score to choose k
K_range = range(2, 11)
inertias, sil_scores, db_scores = [], [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels, sample_size=2000, random_state=42))
    db_scores.append(davies_bouldin_score(X, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(list(K_range), inertias, marker='o', color=NETFLIX_RED)
axes[0].set_title('Elbow Method (Inertia)')
axes[0].set_xlabel('k')
axes[1].plot(list(K_range), sil_scores, marker='o', color=NETFLIX_BLACK)
axes[1].set_title('Silhouette Score by k')
axes[1].set_xlabel('k')
plt.tight_layout()
plt.show()

best_k = list(K_range)[int(np.argmax(sil_scores))]
print("Chosen k (highest silhouette score):", best_k)

# Predict on the model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['kmeans_cluster'] = kmeans.fit_predict(X)
print(df['kmeans_cluster'].value_counts().sort_index())


# ==== Cell 53 ====
# Visualizing evaluation Metric Score chart
sil_kmeans = silhouette_score(X, df['kmeans_cluster'], sample_size=2000, random_state=42)
db_kmeans = davies_bouldin_score(X, df['kmeans_cluster'])

metrics_df = pd.DataFrame({'Metric': ['Silhouette Score (higher=better)', 'Davies-Bouldin Index (lower=better)'],
                            'Value': [sil_kmeans, db_kmeans]})
plt.figure(figsize=(6, 4))
sns.barplot(data=metrics_df, x='Metric', y='Value', color=NETFLIX_RED)
plt.title('K-Means Evaluation Metrics')
plt.xticks(rotation=15)
plt.show()
print(metrics_df)

# Model explainability: top TF-IDF terms per cluster (used in Section 7's explainability answer)
terms = np.array(tfidf.get_feature_names_out())
cluster_rows = []
for c in sorted(df['kmeans_cluster'].unique()):
    idx = df[df['kmeans_cluster'] == c].index
    mean_tfidf = np.asarray(tfidf_matrix[idx].mean(axis=0)).flatten()
    top_terms = terms[mean_tfidf.argsort()[::-1][:8]]
    cluster_rows.append({
        'cluster': c,
        'size': len(idx),
        'top_terms': ', '.join(top_terms),
        'dominant_type': df.loc[idx, 'type'].mode()[0],
        'dominant_country': df.loc[idx, 'primary_country'].mode()[0],
        'dominant_rating': df.loc[idx, 'rating'].mode()[0],
    })
cluster_summary = pd.DataFrame(cluster_rows)
cluster_summary


# ==== Cell 54 ====
# ML Model - 1 Implementation with hyperparameter optimization techniques
# (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

# K-Means' main hyperparameter is k (n_clusters), already tuned above via the Elbow/Silhouette
# grid search over k=2..10. Here we additionally grid-search the `init` strategy and `n_init` count.
best_score = -1
best_params = None
for init in ['k-means++', 'random']:
    for n_init in [5, 10, 20]:
        km = KMeans(n_clusters=best_k, init=init, n_init=n_init, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=2000, random_state=42)
        if score > best_score:
            best_score, best_params = score, {'init': init, 'n_init': n_init}

print("Best KMeans params:", best_params, "Silhouette:", round(best_score, 4))

# Fit the Algorithm with the best found params
kmeans = KMeans(n_clusters=best_k, random_state=42, **best_params)
# Predict on the model
df['kmeans_cluster'] = kmeans.fit_predict(X)


# ==== Cell 55 ====
# Visualizing evaluation Metric Score chart
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

# Dendrogram on a sample for readability
sample_idx = np.random.RandomState(42).choice(len(X), size=min(300, len(X)), replace=False)
Z = linkage(X[sample_idx], method='ward')

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram (sample of 300 titles)')
plt.xlabel('Titles')
plt.ylabel('Distance (Ward)')
plt.show()

agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
df['hier_cluster'] = agg.fit_predict(X)

sil_agg = silhouette_score(X, df['hier_cluster'], sample_size=2000, random_state=42)
db_agg = davies_bouldin_score(X, df['hier_cluster'])
print(f"Agglomerative (ward) -- Silhouette: {sil_agg:.4f}, Davies-Bouldin: {db_agg:.4f}")
print(df['hier_cluster'].value_counts().sort_index())


# ==== Cell 56 ====
# ML Model - 2 Implementation with hyperparameter optimization techniques
# (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

# Agglomerative Clustering's key hyperparameter is the linkage method; grid-search it directly
# (again via Silhouette Score, since there's no label to cross-validate against).
sample_idx2 = np.random.RandomState(42).choice(len(X), size=min(1500, len(X)), replace=False)
best_linkage, best_link_score = None, -1
for link in ['ward', 'average', 'complete']:
    agg_test = AgglomerativeClustering(n_clusters=best_k, linkage=link)
    labels_test = agg_test.fit_predict(X[sample_idx2])
    score = silhouette_score(X[sample_idx2], labels_test)
    print(f"linkage={link}: silhouette={score:.4f}")
    if score > best_link_score:
        best_link_score, best_linkage = score, link

print("\nBest linkage method:", best_linkage)

# Fit the Algorithm
agg = AgglomerativeClustering(n_clusters=best_k, linkage=best_linkage)
# Predict on the model
df['hier_cluster'] = agg.fit_predict(X)


# ==== Cell 57 ====
# ML Model - 3 Implementation
from sklearn.cluster import DBSCAN

# Fit the Algorithm -- grid search eps / min_samples (DBSCAN's key hyperparameters)
best_eps, best_ms, best_db_score, best_n_found = None, None, -2, 0
for eps in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    for ms in [5, 10, 15]:
        db_test = DBSCAN(eps=eps, min_samples=ms)
        labels_test = db_test.fit_predict(X)
        n_found = len(set(labels_test)) - (1 if -1 in labels_test else 0)
        if n_found < 2:
            continue
        score = silhouette_score(X, labels_test, sample_size=2000, random_state=42)
        if score > best_db_score:
            best_db_score, best_eps, best_ms, best_n_found = score, eps, ms, n_found

print(f"Best DBSCAN params: eps={best_eps}, min_samples={best_ms} -> {best_n_found} clusters, silhouette={best_db_score:.4f}")

# Predict on the model
dbscan = DBSCAN(eps=best_eps, min_samples=best_ms)
df['dbscan_cluster'] = dbscan.fit_predict(X)

noise_pct = (df['dbscan_cluster'] == -1).mean() * 100
print(f"Noise points flagged: {noise_pct:.1f}% of titles")
print(df['dbscan_cluster'].value_counts().head(10))


# ==== Cell 58 ====
# Visualizing evaluation Metric Score chart
valid_dbscan = df['dbscan_cluster'].nunique() > 2
if valid_dbscan:
    sil_dbscan = silhouette_score(X, df['dbscan_cluster'], sample_size=2000, random_state=42)
    db_dbscan = davies_bouldin_score(X, df['dbscan_cluster'])
else:
    sil_dbscan, db_dbscan = np.nan, np.nan

comparison_df = pd.DataFrame({
    'Algorithm': ['K-Means', 'Agglomerative', 'DBSCAN'],
    'Silhouette Score': [sil_kmeans, sil_agg, sil_dbscan],
    'Davies-Bouldin Index': [db_kmeans, db_agg, db_dbscan],
})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=comparison_df, x='Algorithm', y='Silhouette Score', ax=axes[0], color=NETFLIX_RED)
axes[0].set_title('Silhouette Score by Algorithm (higher = better)')
sns.barplot(data=comparison_df, x='Algorithm', y='Davies-Bouldin Index', ax=axes[1], color=NETFLIX_BLACK)
axes[1].set_title('Davies-Bouldin Index by Algorithm (lower = better)')
plt.tight_layout()
plt.show()
print(comparison_df)


# ==== Cell 59 ====
# ML Model - 3 Implementation with hyperparameter optimization techniques
# (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

# (Hyperparameter tuning for DBSCAN -- eps & min_samples -- was already performed via the grid
# search above, since those ARE DBSCAN's primary hyperparameters; shown again here for completeness.)
print(f"Tuned DBSCAN hyperparameters: eps={best_eps}, min_samples={best_ms}")
print(f"Resulting silhouette score: {best_db_score:.4f}, clusters found: {best_n_found}, noise: {noise_pct:.1f}%")


# ==== Cell 60 ====
# Save the File
import joblib

pipeline_artifacts = {
    'tfidf_vectorizer': tfidf,
    'svd': svd,
    'normalizer': normalizer,
    'kmeans_model': kmeans,
    'best_k': best_k,
}
joblib.dump(pipeline_artifacts, 'netflix_cluster_pipeline.joblib')
print("Saved pipeline to netflix_cluster_pipeline.joblib")


# ==== Cell 61 ====
# Load the File and predict unseen data.
loaded = joblib.load('netflix_cluster_pipeline.joblib')

# A couple of unseen, hand-written "new title" descriptions to sanity-check the pipeline
unseen_titles = pd.DataFrame({
    'soup_clean': [
        'stand up comedy special comedian jokes live audience',
        'international drama family relationships secrets small town',
    ]
})

unseen_tfidf = loaded['tfidf_vectorizer'].transform(unseen_titles['soup_clean'])
unseen_reduced = loaded['svd'].transform(unseen_tfidf)
unseen_X = loaded['normalizer'].transform(unseen_reduced)
unseen_clusters = loaded['kmeans_model'].predict(unseen_X)

for desc, cluster in zip(unseen_titles['soup_clean'], unseen_clusters):
    print(f"'{desc}' -> predicted cluster {cluster}")
