# In[141]:
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise import KNNBasic
from surprise import KNNWithMeans
import json
import string
import matplotlib.pyplot as plt 
import json_lines
import itertools
import pandas as pd
import operator
import collections
from collections import Counter
import numpy as np



class surpriseRecommender():
    def __init__(self, stories, reviews, users):

        self.stories = stories
        self.reviews = reviews
        self.users = users

        self.storyLinkToIdDict = {}
        self.IdToStoryDict = {}

        #create a dict between storied and their id
        self.lastStoryId = 0
        for story in stories:
            self.storyLinkToIdDict[story['storyLink']] = self.lastStoryId
            self.IdToStoryDict[self.lastStoryId] = story
            self.lastStoryId += 1
            
        self.userLinkToIdDict = {}
        self.IdToUserDict = {}

        self.lastUserId = 0
        for user in users:
            self.userLinkToIdDict[user['name']] = self.lastUserId
            self.IdToUserDict[self.lastUserId] = user
            self.lastUserId += 1

        for review in reviews:
            if(review['reviewer'] not in self.userLinkToIdDict):
                self.userLinkToIdDict[review['reviewer']] = self.lastUserId
                self.IdToUserDict[self.lastUserId] = review['reviewer']
                self.lastUserId+=1

        self.reviewLinkToIdDict = {}
        self.IdToReviewDict = {}
        self.lastReviewId = 0
        for review in reviews:
            self.reviewLinkToIdDict[review['reviewOf'] + '|' + review['reviewer']] = self.lastReviewId
            self.IdToReviewDict[self.lastReviewId] = review
            self.lastReviewId += 1

        # ## make scores dict

        storyReviewDic = Counter({})
        storyScores = {}

        cnt = 0
        self.minScore = 0
        self.maxScore = 0
        for review in reviews:
            if(review['reviewOf'] in self.storyLinkToIdDict):
                userId = self.userLinkToIdDict[review['reviewer']]
                storyId = self.storyLinkToIdDict[review['reviewOf']]
                score = review['sentimentScore']
                self.minScore = min(score, self.minScore)
                self.maxScore = max(score,self. maxScore)
                storyScores[(userId, storyId)] = {
                            "storyId" : storyId, 
                            "userId" : userId,
                            "score" :  score
                        }
                cnt += 1 
        print(self.minScore, self.maxScore)
        # ### add in favorites data
        # bias favorites over reviews
        for user in users:
            userId = self.userLinkToIdDict[user['name']]
            
            for favorite in user['favorites']:
                if(favorite['favStory'] in self.storyLinkToIdDict):
                    
                    storyId = self.storyLinkToIdDict[favorite['favStory']]
                    score = 10
                    if((userId, storyId) not in storyScores):
                        storyScores[(userId, storyId)] = {
                            "storyId" : storyId, 
                            "userId" : userId,
                            "score" :  0
                        }
                    storyScores[(userId, storyId)]['score'] += score

        self.inputScores = []
        for score, body in storyScores.items():
            self.inputScores.append(body)

    def train(self):
        df = pd.DataFrame(self.inputScores)
        reader = Reader(rating_scale=(self.minScore, self.maxScore))
        data = Dataset.load_from_df(df[['userId', 'storyId', 'score']], reader)
        trainset = data.build_full_trainset()
        self.algo = SVD()
        self.algo.fit(trainset)

    def getTopPredictions(self, userId):
        df = pd.DataFrame(self.inputScores)
        df_filtered = df.query('userId==' + str(userId))
        #print(df_filtered)
        test_items = []
        for story in self.stories:
            storyId = self.storyLinkToIdDict[story['storyLink']]
            test_items.append({
                        "storyId" : storyId, 
                        "userId" : userId,
                        "score" :  0
                    })
        df = pd.DataFrame(test_items)
        #remove values the user already knows
        mask = np.logical_not(df['storyId'].isin(set(df_filtered['storyId'])))
        df = df[mask]
    
        reader = Reader(rating_scale=(self.minScore, self.maxScore))
        data = Dataset.load_from_df(df[['userId', 'storyId', 'score']], reader)
        trainset = data.build_full_trainset()
        testset = trainset.build_testset()
        predictions = self.algo.test(testset)
        
        scores = {}
        for uid, iid, true_r, est, _ in predictions:
            scores[self.IdToStoryDict[iid]['storyLink']] =  est

        return scores

    def predict(self, user):
        #if we havent seen this user before
        if(user['name'] not in self.userLinkToIdDict):
            self.userLinkToIdDict[user['name']] = self.lastUserId
            self.IdToUserDict[self.lastUserId] = user
            self.lastUserId += 1
            self.train()
        return self.getTopPredictions(self.userLinkToIdDict[user['name']])

#%%
'''
with open('result.jl', 'rb') as f:
    Fentities = [x for x in json_lines.reader(f)][:100000]
    Fstories = [x for x in Fentities if str(x['pageType']) == "story" and str(x['storyType']) == '/book/Harry-Potter/']
    Freviews = [x for x in Fentities if str(x['pageType']) == "review"]
    Fusers = [x for x in Fentities if str(x['pageType']) == "user"]
    heldoutUser = Fusers[0]
    Fusers = Fusers[1:]

recommender = surpriseRecommender(Fstories, Freviews, Fusers)
recommender.train()
'''
#trainedAlgo = train(inputScores)
#getTopNPredictions(trainedAlgo, inputScores, 500, 10)
#predict(heldoutUser)
