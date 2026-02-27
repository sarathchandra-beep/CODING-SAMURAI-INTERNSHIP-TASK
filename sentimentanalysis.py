import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("sentimentdataset.csv")

print("Dataset Loaded Successfully ✅")
print(df.head())

positive_keywords = [
    'success', 'joy', 'wonder', 'spark',
    'vibrancy', 'triumph', 'zest',
    'thrill', 'tenderness', 'touched',
    'excited', 'gratitude', 'happy'
]

negative_keywords = [
    'sorrow', 'suffering', 'shame',
    'yearning', 'fear', 'anger',
    'sad', 'solitude', 'blues',
    'terrible', 'bad'
]

def smart_map(sent):
    sent_lower = str(sent).lower()

    if any(word in sent_lower for word in positive_keywords):
        return "Positive"
    elif any(word in sent_lower for word in negative_keywords):
        return "Negative"
    else:
        return "Neutral"

df['Main_Sentiment'] = df['Sentiment'].apply(smart_map)

print("\nMain Sentiment Distribution:")
print(df['Main_Sentiment'].value_counts())


plt.figure(figsize=(6,4))
df['Main_Sentiment'].value_counts().plot(kind='bar')
plt.title("Overall Main Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

engagement = df.groupby('Main_Sentiment')[['Likes','Retweets']].mean()
print("\nAverage Engagement by Sentiment:")
print(engagement)

# Likes by Sentiment
plt.figure(figsize=(6,4))
df.groupby('Main_Sentiment')['Likes'].mean().plot(kind='bar')
plt.title("Average Likes by Sentiment")
plt.ylabel("Average Likes")
plt.tight_layout()
plt.show()

# Retweets by Sentiment
plt.figure(figsize=(6,4))
df.groupby('Main_Sentiment')['Retweets'].mean().plot(kind='bar')
plt.title("Average Retweets by Sentiment")
plt.ylabel("Average Retweets")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,5))
pd.crosstab(df['Platform'], df['Main_Sentiment']).plot(kind='bar')
plt.title("Platform vs Main Sentiment")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
pd.crosstab(df['Country'], df['Main_Sentiment']).plot(kind='bar', stacked=True)
plt.title("Country-wise Sentiment Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


hour_engagement = df.groupby('Hour')['Likes'].mean()

plt.figure(figsize=(8,5))
hour_engagement.plot()
plt.title("Average Likes by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Likes")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,4))
sns.heatmap(
    df[['Likes','Retweets','Hour']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nAnalysis Completed Successfully 🚀")