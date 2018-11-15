from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import json
import string
import matplotlib.pyplot as plt 
import json_lines
import itertools
import collections
from collections import Counter

with open('result.jl', 'rb') as f:
    entities = [x for x in json_lines.reader(f)]
    stories = [x for x in entities if str(x['pageType']) == "story" and str(x['storyType']) == '/book/Harry-Potter/']
    storiesNotPotter = [x for x in entities if str(x['pageType']) == "story" and str(x['storyType']) != '/book/Harry-Potter/']
    reviews = [x for x in entities if str(x['pageType']) == "review"]
    users = [x for x in entities if str(x['pageType']) == "user"]
    
    print(len(stories),len(storiesNotPotter),len(reviews),len(users))

#generate character id mappings
characterFreqs = Counter({})
for x in stories:
    characterFreqs += Counter(x['characters'])
characterFreqs = dict(characterFreqs)
charList = list(characterFreqs.keys())
idToChar = {charList.index(x): x for x in charList}
charToId = {x : charList.index(x) for x in charList}

def foldDict(chars):
    charsMult = [(str(charToId[char]) + ' ') * freq for char, freq in chars.items()]
    charsStr = ''
    for charStr in charsMult:
        charsStr += charStr
    return charsStr

#get all the characters present
charList = [ foldDict(story['characters']) for story in stories]

#make the lda model
n_features = 5000
n_topics = 100
n_top_words = 10

train = charList
tf_vectorizer = CountVectorizer(max_df=0.8, min_df=10,
                                max_features=n_features,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(train)
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=20,
                                learning_method='online',
                                learning_offset=50.)

ldaalloc = lda.fit(tf)

tf_feature_names = tf_vectorizer.get_feature_names()

import numpy as np
def buildMatrix(model, num_topics, feature_names):
    matrix = np.zeros([num_topics, len(feature_names)])
    for topic_idx, topic in enumerate(model.components_):
        matrix[topic_idx] = np.array(topic)
    return matrix

LDATopicMatrix = np.transpose(buildMatrix(lda, n_topics, tf_feature_names))

charIdToFeatureId = {}
featureIdToCharId = {}

featureId = 0
for charId in tf_feature_names:
    charId = int(charId)
    charIdToFeatureId[charId] = featureId
    featureIdToCharId[featureId] = charId
    featureId += 1

def convertCharDicToFeatureVec(story, feature_names):
    totalCharFreq = 0
    featureVec = np.zeros((len(feature_names)))
    chars = dict(story['characters'])
    for char, freq in chars.items():
        totalCharFreq += freq
    for char, freq in chars.items():
        if(charToId[char] in charIdToFeatureId):
            featureId = charIdToFeatureId[charToId[char]]
            featureVec[featureId] = freq
    return featureVec
    
def buildDocCharacterMatrix(feature_names):
    matrix = np.zeros([len(stories), len(feature_names)])
    storyId = 0
    for story in stories:
        matrix[storyId] = convertCharDicToFeatureVec(story, feature_names)
        storyId += 1

    return matrix
    

docCharMatrix = buildDocCharacterMatrix(tf_feature_names)

def magnitude(vec):
    return np.linalg.norm(vec)

def cosine(storyVec, TopicVec):
    ANorm = magnitude(storyVec)
    BNorm = magnitude(TopicVec)
    dot = np.dot(storyVec, TopicVec)
    return (dot / (ANorm*BNorm))

def getTopTopicsForStory(story, TopicMatrix, featureNamesList):
    storyFeatureVec = convertCharDicToFeatureVec(story, featureNamesList)
    predictedTopicScores = np.zeros(len(TopicMatrix[0]))
    
    for i in range(len(TopicMatrix[0])):
        topicVec = TopicMatrix[:,i]
        predictedTopicScores[i] = cosine(storyFeatureVec, topicVec)
    
    bestTopic = predictedTopicScores.argmax()
    bestTopicScore = predictedTopicScores[bestTopic]
    
    return (bestTopic, bestTopicScore)

story = stories[0]
print(getTopTopicsForStory(story, LDATopicMatrix, tf_feature_names))