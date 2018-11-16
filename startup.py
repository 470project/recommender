
'''
Imports the global variable from the init file
Should be declared as:
global userFavs = {}
global topStories = {}
'''

from app import userFavs
from app import topStories

#read json file
import json

#userFavs saves each users favorite authors
#userFavs = {}
#topStories = {}

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