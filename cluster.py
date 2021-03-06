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
import numpy as np
from sklearn.preprocessing import normalize

class cluster():
    def __init__(self, stories):
        self.stories = stories
        #generate character id mappings
        characterFreqs = Counter({})
        for x in self.stories:
            characterFreqs += Counter(x['characters'])
        characterFreqs = dict(characterFreqs)
        charList = list(characterFreqs.keys())
        self.idToChar = {charList.index(x): x for x in charList}
        self.charToId = {x : charList.index(x) for x in charList}

        def foldDict(chars):
            charsMult = [(str(self.charToId[char]) + ' ') * freq for char, freq in chars.items()]
            charsStr = ''
            for charStr in charsMult:
                charsStr += charStr
            return charsStr

        #get all the characters present
        charList = [ foldDict(story['characters']) for story in self.stories]

        #make the lda model
        n_features = 5000
        self.n_topics = 100
        n_top_words = 10

        train = charList
        tf_vectorizer = CountVectorizer(max_df=0.8, min_df=10,
                                        max_features=n_features,
                                        stop_words='english')

        tf = tf_vectorizer.fit_transform(train)
        lda = LatentDirichletAllocation(n_components=self.n_topics, max_iter=20,
                                        learning_method='online',
                                        learning_offset=50.)

        ldaalloc = lda.fit(tf)

        self.tf_feature_names = tf_vectorizer.get_feature_names()
        
        #build the term to topic score matrix
        self.LDATopicMatrix = np.zeros([self.n_topics, len(self.tf_feature_names)])
        for topic_idx, topic in enumerate(lda.components_):
            self.LDATopicMatrix[topic_idx] = np.array(topic)
        self.LDATopicMatrix = self.LDATopicMatrix.transpose()

        self.charIdToFeatureId = {}
        self.featureIdToCharId = {}

        featureId = 0
        #make the dictionary to go from character id to feature id
        #features have some characters left out if they are too frequent or infrequent
        for charId in self.tf_feature_names:
            charId = int(charId)
            self.charIdToFeatureId[charId] = featureId
            self.featureIdToCharId[featureId] = charId
            featureId += 1

        self.storyLinkToIdDict = {}
        self.IdToStoryDict = {}

        #create a dict between storied and their id
        storyId = 0
        for story in self.stories:
            self.storyLinkToIdDict[story['storyLink']] = storyId
            self.IdToStoryDict[storyId] = story
            storyId += 1
            
        def buildDocCharacterMatrix(feature_names):
            matrix = np.zeros([len(self.stories), len(feature_names)])
            for story in self.stories:
                storyId = self.storyLinkToIdDict[story['storyLink']]
                matrix[storyId] = self.convertCharDicToFeatureVec(story, feature_names)
            matrix = matrix / np.linalg.norm(matrix)
            return matrix

        #make a matrix that holds the character frequencies in each story
        docCharMatrix = buildDocCharacterMatrix(self.tf_feature_names)

        self.storyTopicScores = np.matmul(docCharMatrix, self.LDATopicMatrix)

    def convertCharDicToFeatureVec(self, story, feature_names):
            totalCharFreq = 0
            featureVec = np.zeros((len(feature_names)))
            chars = dict(story['characters'])
            for char, freq in chars.items():
                totalCharFreq += freq
            for char, freq in chars.items():
                if(self.charToId[char] in self.charIdToFeatureId):
                    featureId = self.charIdToFeatureId[self.charToId[char]]
                    featureVec[featureId] = freq
            return featureVec

    def getTopicScoresForStory(self, story, TopicMatrix, featureNamesList):
        storyFeatureVec = self.convertCharDicToFeatureVec(story, featureNamesList)
        predictedTopicScores = np.zeros(len(TopicMatrix[0]))
        
        for i in range(len(TopicMatrix[0])):
            topicVec = TopicMatrix[:,i]
            predictedTopicScores[i] = np.dot(storyFeatureVec, topicVec)
            
        return predictedTopicScores

    def getStoryIdsFromUser(self, user):
        #get the users favorite stories
        storyIds = []
        for favorite in user['favorites']:
            if(favorite['favStory'] in clusterRecommender.storyLinkToIdDict):
                storyIds.append(clusterRecommender.storyLinkToIdDict[favorite['favStory']])
        return storyIds

    def getScoresForStories(self, storyIds):
        '''
        Takes in a list of story ids and returns the similarity to all other stories based on the clustering
        '''
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
        

        topicScores = np.zeros(self.n_topics)
        #get the topic scores for each story
        for id in storyIds:
            topicScores += self.getTopicScoresForStory(self.IdToStoryDict[id], self.LDATopicMatrix, self.tf_feature_names)
        
        #printprint(topicScores, storyIds,magnitude(topicScores))
        #print(topicScores, magnitude(topicScores))
        topicScores = topicScores / magnitude(topicScores)

        #find all story sim scores
        storySimilarities = np.zeros(len(self.storyTopicScores))
        for i in range(len(self.storyTopicScores)):
            otherStoryTopicScores = self.storyTopicScores[i]
            storySimilarities[i] = cosine(topicScores, otherStoryTopicScores)
        return storySimilarities
            
with open('result.jl', 'rb') as f:
    entities = [x for x in json_lines.reader(f)]
    stories = [x for x in entities if str(x['pageType']) == "story"]
    users = [x for x in entities if str(x['pageType']) == "user"]

#print(getScoresForUser(self.users[0]))

clusterRecommender = cluster(stories)
import time
import math

def checkRMSE():
    def RMSE(predScores, actualScores):
        rmse = 0
        n = len(predScores)
        #print(predScores, n)

        if(n == 0):
            return "invalid"
        for i in range(n):
            predScore = predScores[i]
            if(np.isnan(predScore)):
                predScore = .5
            actualScore = actualScores[i]
            sqrDiff = math.pow(actualScore - predScore, 2)
            rmse += sqrDiff
            #print(rmse, sqrDiff)
        return math.sqrt(rmse/n)

    allPredictions = []
    for user in users:
        storyIds = clusterRecommender.getStoryIdsFromUser(user)
        if(len(storyIds) != 0):
            predictions = clusterRecommender.getScoresForStories(storyIds)
            #print(list(predictions[storyIds]))
            [allPredictions.append(x) for x in list(predictions[storyIds])]

    print(RMSE(allPredictions,np.ones(len(allPredictions))))

checkRMSE()
