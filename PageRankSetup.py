import json
import string
import matplotlib.pyplot as plt 
import json_lines
import itertools

with open('result.jl', 'rb') as f:
    users = [x for x in json_lines.reader(f) if str(x['pageType']) == "user"]
    
    print(len(users))

import numpy as np
idDict = {}
idCnt = 0

for x in users:
    if(x['name'] in idDict or len(x['favorites']) == 0):
        continue
    idDict[idCnt] = x['name']
    idCnt += 1

nameToIdDict = {v: k for k, v in idDict.items()}

userFavorites = {}
for user in users:
    if(user['name'] not in nameToIdDict):
        continue
    userId = nameToIdDict[user['name']]
    userFavorites[userId] = []
    for favorite in user['favorites']:
        auther = favorite['favAuthor']
        if(auther in nameToIdDict):
            autherId = nameToIdDict[auther]
            if(autherId not in userFavorites[userId]):
                userFavorites[userId].append(autherId)

#print(userFavorites)
                
adjacency_matrix = np.zeros([len(idDict), len(idDict)])

for userId, favorites in userFavorites.items():
    adjacency_matrix[userId, userId] = 1
    for favId in favorites:
        adjacency_matrix[userId, favId] = 1

import networkx as nx
rows, cols = np.where(adjacency_matrix == 1)
edges = zip(rows.tolist(), cols.tolist())
gr = nx.Graph()
gr.add_edges_from(edges)
pr = nx.pagerank(gr, alpha=0.9)

import json

prJsonFriendly = [{"userId" : id, "score" : score, "link": idDict[id]} for id, score in pr.items()]

with open('PRresult.json', 'w') as f:
    f.write(json.dumps(prJsonFriendly))