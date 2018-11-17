#%%
import json
import string
import matplotlib.pyplot as plt 
import json_lines
import itertools
class pageRankRecommender():
    def __init__(self, stories, users):
        with open('PRresult.json') as f:
            PRresults = json.load(f)

        userLinkToIdDict = {}
        IdToUserDict = {}

        lastUserId = 0
        for user in users:
            userLinkToIdDict[user['name']] = lastUserId
            IdToUserDict[lastUserId] = user
            lastUserId += 1

        storyLinkToIdDict = {}
        IdToStoryDict = {}

        #create a dict between storied and their id
        lastStoryId = 0
        for story in stories:
            storyLinkToIdDict[story['storyLink']] = lastStoryId
            IdToStoryDict[lastStoryId] = story
            lastStoryId += 1

        self.storyLinkToScores = {}

        for res in PRresults:
            score = res['score']
            link = res['link']

            if(link in userLinkToIdDict):
                user = IdToUserDict[userLinkToIdDict[link]]
                for story in user['stories']:
                    self.storyLinkToScores[story] = score

    def predict(self):
        return self.storyLinkToScores

'''
with open('result.jl', 'rb') as f:
    entities = [x for x in json_lines.reader(f)]
    stories = [x for x in entities if str(x['pageType']) == "story" and str(x['storyType']) == '/book/Harry-Potter/']
    users = [x for x in entities if str(x['pageType']) == "user"]

PRRec = pageRankRecommender(stories, users)
print(len(PRRec.predict()))
'''