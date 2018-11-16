from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise import KNNBasic
import json
import string
import matplotlib.pyplot as plt 
import json_lines
import itertools
import pandas as pd
with open('result.jl', 'rb') as f:
    entities = [x for x in json_lines.reader(f)]
    stories = [x for x in entities if str(x['pageType']) == "story" and str(x['storyType']) == '/book/Harry-Potter/']
    reviews = [x for x in entities if str(x['pageType']) == "review"]
    users = [x for x in entities if str(x['pageType']) == "user"]
    
    print(len(stories),len(reviews),len(users))

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

storyScores = []
for user in users:
    userId = userLinkToIdDict[user['name']]
    
    for favorite in user['favorites']:
        if(favorite['favStory'] in storyLinkToIdDict):
            
            storyId = storyLinkToIdDict[favorite['favStory']]
            score = 1
            storyScores.append(
                {
                    "storyId" : storyId, 
                    "userId" : userId,
                    "score" :  score
                })

df = pd.DataFrame(storyScores)
print(df)
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(df[['userId', 'storyId', 'score']], reader)
#cross_validate(NormalPredictor(), data, cv=2)
# Use the famous SVD algorithm.
algo = SVD()
#algo = KNNBasic()
# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

uid = str(0)  # raw user id (as in the ratings file). They are **strings**!
iid = str(71)  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, verbose=True)
