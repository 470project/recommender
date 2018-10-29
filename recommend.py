'''
* Recommender 
* Pearson correlation for similarity and then the grab the highest ratings from the most similar users as long as those have never been rated by the target user.
'''

#read json file
import json

#user that we are matching
matchUser = ["/u/1138361/iheartmwpp", "/u/8545331/Professor-Flourish-and-Blotts", "/u/49515/Iniga", "/u/4286546/Missbexiee", "/u/1697963/lydiamaartin", "/u/609412/Crystallic-Rain"]

#userFavs saves each users favorite authors
userFavs = {}

with open('result.jl') as f:
    for line in f:
        j = json.loads(line)
        if j["pageType"] == "user":
            favAuthors = []
            favs = j["favorites"]
            for elem in favs:
                favAuthors.append(elem["favAuthor"])
            userFavs[j["name"]] = set(favAuthors)


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
#top ten most similar
for i in range(10):
    topTen.append(sortedJaccard[i])
    print(sortedJaccard[i][0], sortedJaccard[i][1])

    
#Get top 10 users and get the stories that they rated highly and the other user has not rated.