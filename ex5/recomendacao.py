import pandas as pd
dataFile='ratings.csv'
data=pd.read_csv(dataFile,sep=",",header=0,names=["userId","movieId","rating","timestamp"])

data.head()

movieFile='movies.csv'
movies=pd.read_csv(movieFile,sep=",",header=0,names=["movieId","title","genres"])

movies.head()

def movieMeta(movieId):
    title = movies.at[movieId,"title"]
    genre = movies.at[movieId,"genres"]
    return title, genre

data = data[data["movieId"].isin(movies.index)]

def faveMovies(user,N):
    
    userRatings = data[data["userId"]==user]
    
    sortedRatings = pd.DataFrame.sort_values(userRatings,['rating'],ascending=[0])[:N]
    
    sortedRatings["title"] = sortedRatings["movieId"].apply(movieMeta)
    return sortedRatings

print(faveMovies(34,10))

data.shape

usersPerMOVIE = data.movieId.value_counts()
usersPerMOVIE.head(10)

usersPerMOVIE.shape

MOVIESsPerUser = data.userId.value_counts()
MOVIESsPerUser.head()

MOVIESsPerUser.shape
data = data[data["movieId"].isin(usersPerMOVIE[usersPerMOVIE>10].index)]

userItemRatingMatrix=pd.pivot_table(data, values='rating',
                                    index=['userId'], columns=['movieId'])

userItemRatingMatrix.head()
userItemRatingMatrix.shape

user1 = 1
user2 = 20
user1Ratings = userItemRatingMatrix.transpose()[user1]
user1Ratings.head(100)

user2Ratings = userItemRatingMatrix.transpose()[user2]

from scipy.spatial.distance import hamming 
hamming(user1Ratings,user2Ratings)

import numpy as np
def distance(user1,user2):
        try:
            
            user1Ratings = userItemRatingMatrix.transpose()[user1]
            user2Ratings = userItemRatingMatrix.transpose()[user2]
            distance = hamming(user1Ratings,user2Ratings)
        except: 
            distance = np.NaN
        return distance

user = 6
allUsers = pd.DataFrame(userItemRatingMatrix.index)
allUsers = allUsers[allUsers.userId!=user]
allUsers.head(10)

allUsers["distance"] = allUsers["userId"].apply(lambda x: distance(user,x))

allUsers.head()
K = 10
KnearestUsers = allUsers.sort_values(["distance"],ascending=True)["userId"][:K]

def nearestNeighbors(user,K=10):
    allUsers = pd.DataFrame(userItemRatingMatrix.index)
    allUsers = allUsers[allUsers.userId!=user]
    allUsers["distance"] = allUsers["userId"].apply(lambda x: distance(user,x))
    KnearestUsers = allUsers.sort_values(["distance"],ascending=True)["userId"][:K]
    return KnearestUsers

KnearestUsers = nearestNeighbors(user)

print(KnearestUsers)

NNRatings = userItemRatingMatrix[userItemRatingMatrix.index.isin(KnearestUsers)]
avgRating = NNRatings.apply(np.nanmean).dropna()
avgRating.head()

moviesAlreadyWatched = userItemRatingMatrix.transpose()[user].dropna().index
avgRating = avgRating[~avgRating.index.isin(moviesAlreadyWatched)]

N=3
topNMOVIEIDs = avgRating.sort_values(ascending=False).index[:N]
pd.Series(topNMOVIEIDs).apply(movieMeta)

def topN(user,N=3):
    KnearestUsers = nearestNeighbors(user)
    NNRatings = userItemRatingMatrix[userItemRatingMatrix.index.isin(KnearestUsers)]
    avgRating = NNRatings.apply(np.nanmean).dropna()
    moviesAlreadyWatched = userItemRatingMatrix.transpose()[user].dropna().index
    avgRating = avgRating[~avgRating.index.isin(moviesAlreadyWatched)]
    ratingPredictedValue = avgRating.sort_values(ascending=False)
    
    topNMOVIEIDs = avgRating.sort_values(ascending=False).index[:N]
    recommendation = pd.DataFrame(topNMOVIEIDs) 
    recommendation["title"] = recommendation["movieId"].apply(movieMeta)
    recommendation["Prediction"] = ratingPredictedValue.values[:N]
    return recommendation

faveMovies(3,10)

print(topN(6,70))