### Importing packages ###
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


##### THE NAME OF THE DATASET DIFFERS FROM THE KAGGLE'S ONE #####
### Reading the data ###
raw_data = pd.read_csv('Womens_clothing.csv')
print(raw_data.head())

# Sampling 3000 reviews randomly
my_sample = raw_data.sample(n=3000, random_state=42)
print(my_sample.head())



### Data Preprocessing ###
# Dropping missing values
my_sample = my_sample.dropna(subset=['Review Text'])

# Converting reviews to lowercase
my_sample['Review Text'] = my_sample['Review Text'].str.lower()

# Removing special characters and numbers
my_sample['Review Text'] = my_sample['Review Text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# Performing tokenization
my_sample['Review Text'] = my_sample['Review Text'].apply(lambda x: nltk.word_tokenize(x))

# Removing stopwords
stop_words = set(stopwords.words('english'))
my_sample['Review Text'] = my_sample['Review Text'].apply(lambda x: [word for word in x if word not in stop_words])

# Performing lemmatization
lemmatizer = WordNetLemmatizer()
my_sample['Review Text'] = my_sample['Review Text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Merging the tokens back into sentences
my_sample['Review Text'] = my_sample['Review Text'].apply(lambda x: ' '.join(x))



### Classification ###
# Splitting the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(my_sample['Review Text'], my_sample['Recommended IND'], test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Logistic Regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Evaluating the performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)

## ROC curve
y_pred_proba = classifier.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)

# Plotting the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()

print("AUC Score:", auc_score)



### Clustering ###
## The Elbow Method
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(my_sample['Review Text'])

max_clusters = 10
wcss = []

for num_clusters in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.plot(range(1, max_clusters + 1), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()

## K-means with 8 clusters
num_clusters = 8  # Number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(X.toarray())

# Adding cluster labels to the dataset
labels = kmeans.labels_
my_sample['Cluster'] = labels

# Visualizing the clusters
plt.figure()
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels)
plt.title('K-means Clustering')
plt.xlabel('PC 1')
plt.ylabel('PC 2')

# Making a legend
legend_elements = []
for label in set(labels):
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}', markerfacecolor=scatter.to_rgba(label)))
plt.legend(handles=legend_elements)

plt.show()



### Topic modelling ###
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(my_sample['Review Text'])

# Performing LDA
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Extracting important topics and their top words
num_top_words = 10
feature_names = vectorizer.get_feature_names_out()

topics = {}
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
    topics[f"Topic {topic_idx+1}"] = top_words

# Visualisation of important topics
for topic, top_words in topics.items():
    print(f"{topic}: {', '.join(top_words)}")

    # Generating a word cloud for each topic
    wordcloud = WordCloud(background_color='white').generate(" ".join(top_words))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(topic)
    plt.axis('off')
    plt.show()

## Topic distributions
topic_distributions = lda.transform(X)
topic_proportions = topic_distributions.sum(axis=0) / topic_distributions.sum()

# Visualizing topic distributions
plt.bar(range(num_topics), topic_proportions)
plt.xlabel('Topic')
plt.ylabel('Proportion')
plt.title('Topic Distributions')
plt.xticks(range(num_topics), ['Topic {}'.format(i+1) for i in range(num_topics)])
plt.show()



### Analysis of mutual similarity ###
X = vectorizer.fit_transform(my_sample['Review Text'])

# Similarity matrix
similarity_matrix = cosine_similarity(X)



### Collocation Analysis ###
# Creating bigrams
vectorizer = TfidfVectorizer(ngram_range=(2, 2))

# TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(my_sample['Review Text'])

bigrams = vectorizer.get_feature_names_out()

# Average TF-IDF score for each bigram
avg_tfidf_scores = tfidf_matrix.mean(axis=0).tolist()[0]

bigrams_df = pd.DataFrame({'Bigram': bigrams, 'TF-IDF Score': avg_tfidf_scores})

# Sorting bigrams in descending order
sorted_bigrams_df = bigrams_df.sort_values(by='TF-IDF Score', ascending=False)

# Top collocations
top_collocations = sorted_bigrams_df.head(15)
print(top_collocations)


### Analysis of mutual similarity ###
X = vectorizer.fit_transform(my_sample['Review Text'])

# Similarity matrix
similarity_matrix = cosine_similarity(X)




