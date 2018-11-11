'''
* Recommender 
* jaccard similarity and then grab the highest ratings from the most similar users as long as those have never been rated by the target user.
'''

#read json file
import json

#user that we are matching
matchUser = ["/u/1138361/iheartmwpp", "/u/8545331/Professor-Flourish-and-Blotts", "/u/4286546/Missbexiee", "/u/1697963/lydiamaartin", "/u/609412/Crystallic-Rain"]

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
    

topTen = []
authorsToLookAt = []
#top twenty most similar
print("Most similar users")
for i in range(20):
    topTen.append(sortedJaccard[i])
    print(sortedJaccard[i][0], sortedJaccard[i][1])
    favList = userFavs[sortedJaccard[i][0]]
    for elem in favList:
        if elem not in matchUser:
            authorsToLookAt.append(elem)

print("")
print("Authors to check out")

for elem in set(authorsToLookAt):
    print(elem)

print("")
print("Stories to check out")
#printing out the top stories from the authors that are recommended.
for elem in set(authorsToLookAt):
    if elem in topStories:
        print(topStories[elem][0], topStories[elem][1])

