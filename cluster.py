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
    #print(featureId,charId, idToChar[int(charId)])
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

storyLinkToIdDict = {}
IdToStoryDict = {}
storyId = 0
for story in stories:
    #matrix[storyId] = convertCharDicToFeatureVec(story, feature_names)
    storyLinkToIdDict[story['storyLink']] = storyId
    IdToStoryDict[storyId] = story
    storyId += 1
    
def buildDocCharacterMatrix(feature_names):
    matrix = np.zeros([len(stories), len(feature_names)])
    #print(len(feature_names), len(charToId))
    
    for story in stories:
        storyId = storyLinkToIdDict[story['storyLink']]
        matrix[storyId] = convertCharDicToFeatureVec(story, feature_names)
        
    #matrix = (matrix / matrix.sum())
    return matrix;

docCharMatrix = buildDocCharacterMatrix(tf_feature_names)

storyTopicScores = np.matmul(docCharMatrix, LDATopicMatrix)
predictedTopics = [storyTopicScores[i].argmax() for i in range(len(storyTopicScores))] 

def getTopicScoresForStory(story, TopicMatrix, featureNamesList):
    storyFeatureVec = convertCharDicToFeatureVec(story, featureNamesList)
    predictedTopicScores = np.zeros(len(TopicMatrix[0]))
    
    for i in range(len(TopicMatrix[0])):
        topicVec = TopicMatrix[:,i]
        predictedTopicScores[i] = np.dot(storyFeatureVec, topicVec)
        
    return predictedTopicScores

def magnitude(vec):
    return np.linalg.norm(vec)

def cosine(storyVec, TopicVec):
    ANorm = magnitude(storyVec)
    BNorm = magnitude(TopicVec)
    dot = np.dot(storyVec, TopicVec)
    if((ANorm*BNorm) == 0):
        return 0
    res = (dot / (ANorm*BNorm))
    return res

def getScoresForUser(user):
    storyIds = []
    for favorite in user['favorites']:
        if(favorite['favStory'] in storyLinkToIdDict):
            storyIds.append(storyLinkToIdDict[favorite['favStory']])
    topicScores = np.zeros(n_topics)
    for id in storyIds:
        topicScores += getTopicScoresForStory(IdToStoryDict[id], LDATopicMatrix, tf_feature_names)
    print(len(LDATopicMatrix), len(LDATopicMatrix[0]))
    topicScores = topicScores / magnitude(topicScores)
    storySimilarities = np.zeros(len(storyTopicScores))
    for i in range(len(storyTopicScores)):
        otherStoryTopicScores = storyTopicScores[i]
        storySimilarities[i] = cosine(topicScores, otherStoryTopicScores)
    return storySimilarities
    
print(getScoresForUser(users[0]))
