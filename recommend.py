'''
* Recommender 
* jaccard similarity and then grab the highest ratings from the most similar users as long as those have never been rated by the target user.
'''

#read json file
import json

#importing the global variables
from app import userFavs
from app import topStories

#user that we are matching
matchUser = ["/u/1138361/iheartmwpp", "/u/8545331/Professor-Flourish-and-Blotts", "/u/4286546/Missbexiee", "/u/1697963/lydiamaartin", "/u/609412/Crystallic-Rain"]


'''
#userFavs saves each users favorite authors
userFavs = {}
topStories = {}

with open('result.jl') as f:
    for line in f:
        j = json.loads(line)
        if j["pageType"] == "user":
            favAuthors = []
            favs = j["favorites"]
            for elem in favs:
                favAuthors.append(elem["favAuthor"])
            userFavs[j["name"]] = set(favAuthors)

        if j["pageType"] == "story":
            favs = int(j["otherInfo"]["favorites"])
            author = j["author"]
            link = j["storyLink"]
            if author not in topStories:
                topStories[author] = (link, int(favs))
            else:
                #if the current top story for the author has less favorites than the new story then make the new story the top story. else don't change anything.
                if int(topStories[author][1]) < int(favs):
                    topStories[author] = (link, int(favs))
'''
            



#Find similar users
jaccardDict = {}

for key in userFavs:
    cInter = 0
    cUnion = len(matchUser)
    for author in userFavs[key]:
        if author in matchUser:
            cInter+=1
        else:
            cUnion+=1
            
    jaccardDict[key] = cInter/cUnion

#Sorting Jaccard Dictionary    
sortedJaccard = sorted(jaccardDict.items(), key=lambda kv: kv[1], reverse=True)
    
authorsToLookAt = []
authorStoryScore = {}

#top twenty most similar
print("Most similar users")
for i in range(20):
    favList = userFavs[sortedJaccard[i][0]]
    for elem in favList:
        if elem not in matchUser:
            if elem in authorStoryScore:
                #adds the similarity score to the previous score that way authors that show up multiple times have their weight increased.
                #Not the best way to add but yolo
                newSim = authorStoryScore[elem][0] + sortedJaccard[i][1]
                authorStoryScore[elem] = (newSim, "")
            else:
                authorsToLookAt.append(elem)
                authorStoryScore[elem] = (sortedJaccard[i][1], "")        #saves the similarity score from the user and leaves the storylink blank for now.

for elem in set(authorsToLookAt):
    if elem in topStories:
        simScore = authorStoryScore[elem][0]
        authorStoryScore[elem] = (simScore, topStories[elem][0])

print("Printing Author similarityScore story link.")    #if there is no story link then the story link area will be an empty string.
print("")
for key in authorStoryScore:
    print(key, authorStoryScore[key][0], authorStoryScore[key][1])
    print("")


