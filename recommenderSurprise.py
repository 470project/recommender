
# coding: utf-8

# # import

# In[141]:


#from surprise import NormalPredictor
#import surprise


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

with open('result.jl', 'rb') as f:
    entities = [x for x in json_lines.reader(f)]
    stories = [x for x in entities if str(x['pageType']) == "story" and str(x['storyType']) == '/book/Harry-Potter/']
    reviews = [x for x in entities if str(x['pageType']) == "review"]
    users = [x for x in entities if str(x['pageType']) == "user"]
    heldoutUser = users[0]
    users = users[1:]
    print(len(stories),len(reviews),len(users))

class surpriseRecommender():
    def __init__():

    # # construct dicts

    storyLinkToIdDict = {}
    IdToStoryDict = {}

    #create a dict between storied and their id
    storyId = 0
    for story in stories:
        storyLinkToIdDict[story['storyLink']] = storyId
        IdToStoryDict[storyId] = story
        storyId += 1
        
    userLinkToIdDict = {}
    IdToUserDict = {}

    userId = 0
    for user in users:
        userLinkToIdDict[user['name']] = userId
        IdToUserDict[userId] = user
        userId += 1

    for review in reviews:
        if(review['reviewer'] not in userLinkToIdDict):
            userLinkToIdDict[review['reviewer']] = userId
            IdToUserDict[userId] = review['reviewer']
            userId+=1

    reviewLinkToIdDict = {}
    IdToReviewDict = {}
    reviewId = 0
    for review in reviews:
        reviewLinkToIdDict[review['reviewOf'] + '|' + review['reviewer']] = reviewId
        IdToReviewDict[reviewId] = review
        reviewId += 1


# ## make scores dict

storyReviewDic = Counter({})
storyScores = {}

cnt = 0
for review in reviews:
    if(review['reviewOf'] in storyLinkToIdDict):
        #storyReviewDic[storyLinkToIdDict[review['reviewOf']]] += review['sentimentScore']
        userId = userLinkToIdDict[review['reviewer']]
        storyId = storyLinkToIdDict[review['reviewOf']]
        score = review['sentimentScore']
        storyScores[(userId, storyId)] = {
                    "storyId" : storyId, 
                    "userId" : userId,
                    "score" :  score
                }
        cnt += 1 


# ### add in favorites data

for user in users:
    userId = userLinkToIdDict[user['name']]
    
    for favorite in user['favorites']:
        if(favorite['favStory'] in storyLinkToIdDict):
            
            storyId = storyLinkToIdDict[favorite['favStory']]
            score = 2
            if((userId, storyId) not in storyScores):
                storyScores[(userId, storyId)] = {
                    "storyId" : storyId, 
                    "userId" : userId,
                    "score" :  0
                }
            storyScores[(userId, storyId)]['score'] += score

inputScores = []
userList = set()
storyList = set()
for score, body in storyScores.items():
    inputScores.append(body)
    userList.add(body['userId'])
    userList.add(body['storyId'])

def train(examples):
    df = pd.DataFrame(examples)
    reader = Reader(rating_scale=(-5, 10))
    data = Dataset.load_from_df(df[['userId', 'storyId', 'score']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    testset = trainset.build_anti_testset()
    #predictions = algo.test(testset)
    #cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return algo    

trainedAlgo = train(inputScores)

def getTopNPredictions(algo, examples, userId, n = 10):
    df = pd.DataFrame(examples)
    df_filtered = df.query('userId==' + str(userId))
    #print(df_filtered)
    test_items = []
    for story in stories:
        storyId = storyLinkToIdDict[story['storyLink']]
        test_items.append({
                    "storyId" : storyId, 
                    "userId" : userId,
                    "score" :  0
                })
    df = pd.DataFrame(test_items)
    #remove values the user already knows
    mask = np.logical_not(df['storyId'].isin(set(df_filtered['storyId'])))
    df = df[mask]
    
    reader = Reader(rating_scale=(-5, 10))
    data = Dataset.load_from_df(df[['userId', 'storyId', 'score']], reader)
    trainset = data.build_full_trainset()
    testset = trainset.build_testset()
    predictions = algo.test(testset)
    
    top_n = []
    for uid, iid, true_r, est, _ in predictions:
        top_n.append((IdToStoryDict[iid]['storyLink'], iid, est))
    
    top_n.sort(key=lambda x: x[2], reverse=True)
    top_n = top_n[:n]
    return top_n
    
getTopNPredictions(trainedAlgo, inputScores, 500, 10)

def predict(user, n = 10):
    global userId, inputScores, trainedAlgo 
    if(user['name'] not in userLinkToIdDict):
        print('bad', userId)
        userLinkToIdDict[user['name']] = userId
        IdToUserDict[userId] = user
        userId += 1
        trainedAlgo = train(inputScores)
    uid = userLinkToIdDict[heldoutUser['name']]
    return getTopNPredictions(trainedAlgo, inputScores, userLinkToIdDict[user['name']], n)

predict(heldoutUser)