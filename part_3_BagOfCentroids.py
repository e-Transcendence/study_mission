import pygame
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from study_mission.part_2_word2vec import review_to_wordlist

model = Word2Vec.load("D:\All_project\study_mission\\300features_40minwords_10context")
train = pd.read_csv("D:\All_project\study_mission\\labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("D:\All_project\study_mission\\testData.tsv", header=0, delimiter="\t", quoting=3)

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0
num_clusters = int(word_vectors.shape[0] / 5)
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip(model.wv.index2word, idx))

# For the first 10 clusters
for cluster in range(0, 10):
    #
    # Print the cluster number
    print("\nCluster %d" % cluster)
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in range(0, len(word_centroid_map.values())):
        if (list(word_centroid_map.values())[i] == cluster):
            words.append(list(word_centroid_map.keys())[i])
    print(words)


def create_bag_of_centroids(wordlist, word_centroid_map):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(list(word_centroid_map.values())) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))


print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review,remove_stopwords=True))


# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros((train["review"].size, num_clusters),dtype="float32")

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review,word_centroid_map)
    counter += 1

# Repeat for test reviews
test_centroids = np.zeros((test["review"].size, num_clusters),dtype="float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review,word_centroid_map)
    counter += 1

# Fit a random forest and extract predictions
forest = RandomForestClassifier(n_estimators=100)

# Fitting the forest may take a few minutes
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids, train["sentiment"])
result = forest.predict(test_centroids)

# Write the test results
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("BagOfCentroids.csv", index=False, quoting=3)

pygame.mixer.init()
track = pygame.mixer.music.load('D:\All_project\stock_prediction\src\叮.mp3')
for i in range(2):
    pygame.mixer.music.play()
    time.sleep(1)
    pygame.mixer.music.stop()