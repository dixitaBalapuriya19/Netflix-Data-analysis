import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
import numpy as np


# NOTE: clear DATA
df=pd.read_csv('netflix_titles.csv')
df['date_added']=pd.to_datetime(df['date_added'],format='mixed')
df['director'] = df['director'].fillna('unknown')
df['cast'] = df['cast'].fillna('unknown')
df['country'] = df['country'].fillna('unknown')

df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
df['date_added'] = df['date_added'].fillna(df['date_added'].mode()[0])
df['duration_minutes'] = df['duration'].str.extract(r'(\d+)') 
df['duration_minutes'] = pd.to_numeric(df['duration_minutes'], errors='coerce')

mean_movie_duration = df[df['type'] == 'Movie']['duration_minutes'].mean()

df.loc[(df['type'] == 'Movie') & (df['duration_minutes'].isna()), 'duration_minutes'] = mean_movie_duration

df['duration'] = df['duration'].fillna(mean_movie_duration)

df.to_csv("netflix_cleaned.csv", index=False)

df = pd.read_csv("./netflix_cleaned.csv")

df['date_added']=pd.to_datetime(df['date_added'],format='mixed')
df['date_added']

# =======================
#  Section 1: Content Overview
# =======================
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Movies vs TV Shows
sns.countplot(data=df, x='type', palette='pastel', ax=axes[0,0],hue="type")
axes[0,0].set_title("Movies vs TV Shows")
axes[0,0].set_ylabel("Count")

# 2. Content over years
df['year_added'] = df['date_added'].dt.year
yearly_counts = df['year_added'].value_counts().sort_index()
sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker="o", color="red", ax=axes[0,1],hue=yearly_counts.values)
axes[0,1].set_title("Content Added Over the Years")

# 3. By month
df['month_added'] = df['date_added'].dt.month_name()
month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
month_counts = df['month_added'].value_counts().reindex(month_order)
sns.barplot(x=month_counts.index, y=month_counts.values, palette='viridis', ax=axes[1,0],hue=month_counts.values)
axes[1,0].set_title("Additions by Month")
axes[1,0].tick_params(axis='x', rotation=45)

# 4. Top countries
country_counts = df['country'].fillna('Unknown').value_counts().head(10)
sns.barplot(x=country_counts.values, y=country_counts.index, palette='coolwarm', ax=axes[1,1],hue=country_counts.index)
axes[1,1].set_title("Top 10 Producing Countries")

plt.tight_layout()
plt.show()

# =======================
#  Section 2: Genre & Rating
# =======================
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Top genres
genre_counts = Counter()
for genres in df['listed_in'].fillna(''):
    for genre in genres.split(', '):
        genre_counts[genre] += 1
genre_df = pd.DataFrame(genre_counts.items(), columns=['Genre', 'Count']).sort_values(by='Count', ascending=False).head(10)
sns.barplot(x='Count', y='Genre', data=genre_df, palette='mako', ax=axes[0,0],hue="Genre")
axes[0,0].set_title("Top Genres")

# 2. Ratings
df['rating'] = df['rating'].apply(lambda x: np.nan if isinstance(x, str) and 'min' in x.lower() else x)
df['rating'] = df['rating'].fillna('Unknown')
rating_counts = df['rating'].value_counts().head(10)
sns.barplot(x=rating_counts.values, y=rating_counts.index, palette='cubehelix', ax=axes[0,1],hue=rating_counts.index)
axes[0,1].set_title("Top Ratings")

# 3. Movie durations
movies = df[df['type'] == 'Movie']
sns.histplot(movies['duration_minutes'], bins=30, kde=True, color='blue', ax=axes[1,0])
axes[1,0].set_title("Movie Duration Distribution")

# 4. TV Show seasons
tv_shows = df[df['type'] == 'TV Show'].copy()
tv_shows['seasons'] = tv_shows['duration'].str.extract(r'(\d+)').astype(float)
sns.countplot(x='seasons', data=tv_shows, palette='Set2', ax=axes[1,1],hue="seasons")
axes[1,1].set_title("TV Show Seasons Distribution")

plt.tight_layout()
plt.show()

# =======================
#  Section 3: People & Creators
# =======================
fig3, axes = plt.subplots(1, 2, figsize=(16, 6))

# Directors
director_counts = Counter()
for d in df['director'].fillna('Unknown'):
    for name in str(d).split(', '):
        if name.strip().lower() != 'unknown':
            director_counts[name] += 1
director_df = pd.DataFrame(director_counts.items(), columns=['Director', 'Count']).sort_values(by='Count', ascending=False).head(10)
sns.barplot(x='Count', y='Director', data=director_df, palette='crest', ax=axes[0],hue="Director")
axes[0].set_title("Top 10 Directors")

# Actors
actor_counts = Counter()
for c in df['cast'].fillna('Unknown'):
    for actor in str(c).split(', '):
        if actor.strip().lower() != 'unknown':
            actor_counts[actor] += 1
actor_df = pd.DataFrame(actor_counts.items(), columns=['Actor', 'Count']).sort_values(by='Count', ascending=False).head(10)
sns.barplot(x='Count', y='Actor', data=actor_df, palette='mako', ax=axes[1],hue="Actor")
axes[1].set_title("Top 10 Actors")

plt.tight_layout()
plt.show()

# =======================
#  Section 4: Temporal Patterns
# =======================
fig4, axes = plt.subplots(1, 3, figsize=(18, 5))

# Top date
date_counts = df.groupby('date_added').size().reset_index(name='Count').sort_values(by='Count', ascending=False).head(5)
sns.barplot(x='date_added', y='Count', data=date_counts, palette='viridis', ax=axes[0],hue="Count")
axes[0].set_title("Top Dates for Additions")
axes[0].tick_params(axis='x', rotation=45)

# Month pattern
sns.barplot(x=month_counts.index, y=month_counts.values, palette='coolwarm', ax=axes[1],hue=month_counts.values)
axes[1].set_title("Additions by Month")
axes[1].tick_params(axis='x', rotation=45)

# Seasons
season_map = {
    'December': 'Winter', 'January': 'Winter', 'February': 'Winter',
    'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
    'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
    'September': 'Fall', 'October': 'Fall', 'November': 'Fall'
}
df['season'] = df['month_added'].map(season_map)
season_counts = df['season'].value_counts()
sns.barplot(x=season_counts.index, y=season_counts.values, palette='pastel', ax=axes[2],hue=season_counts.values)
axes[2].set_title("Additions by Season")

plt.tight_layout()
plt.show()

# =======================
#  Section 5: Correlation & Insights
# =======================
fig5, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Type vs Duration
sns.boxplot(data=movies, x='type', y='duration_minutes', palette='Set2', ax=axes[0,0],hue="duration_minutes",legend=False)
axes[0,0].set_title("Type vs Duration (Movies)")

# 2. Genre avg duration
genre_durations = defaultdict(list)
for _, row in movies.iterrows():
    for genre in str(row['listed_in']).split(', '):
        if not pd.isna(row['duration_minutes']):
            genre_durations[genre].append(row['duration_minutes'])
genre_avg = {g: sum(v)/len(v) for g,v in genre_durations.items() if len(v) > 5}
genre_avg_df = pd.DataFrame(list(genre_avg.items()), columns=['Genre', 'Avg_Duration']).sort_values(by='Avg_Duration', ascending=False).head(10)
sns.barplot(data=genre_avg_df, x='Avg_Duration', y='Genre', palette='mako', ax=axes[0,1],hue="Genre")
axes[0,1].set_title("Genres with Longest Avg Durations")

# 3. Rating vs Duration
rating_duration = movies.groupby('rating')['duration_minutes'].mean().reset_index().sort_values(by='duration_minutes', ascending=False)
sns.barplot(data=rating_duration, x='duration_minutes', y='rating', palette='coolwarm', ax=axes[1,0],hue="rating")
axes[1,0].set_title("Avg Movie Duration by Rating")

# 4. Genre popularity by country (Top 3 countries)
top_countries = df['country'].value_counts().head(3).index
country_genre = df[df['country'].isin(top_countries)].groupby(['country', 'listed_in']).size().reset_index(name='Count')
sns.barplot(data=country_genre, x='country', y='Count', hue='listed_in', dodge=False, ax=axes[1,1])
axes[1,1].set_title("Genre Popularity in Top 3 Countries")
axes[1,1].legend([],[], frameon=False)  # hide legend for clarity

plt.tight_layout()
plt.show()