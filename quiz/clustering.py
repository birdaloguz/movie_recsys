import pandas as pd
import numpy as np
from scipy import sparse
import scipy
import csv
from sklearn.metrics.pairwise import cosine_similarity

df_movies_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/movies.csv", skiprows=[0], names=["movie_id", "title", "genres"]).drop(columns=['genres'])
df_ratings_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/ratings.csv", skiprows=[0],  names=["user_id", "movie_id", "rating", "timestamp"]).drop(columns=['timestamp']).head(1000000)
df_link_org = pd.read_csv("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/links.csv", skiprows=[0],  names=["movie_id", "imdb_id", "tmdb_id"]).drop(columns=['tmdb_id'])

df_movies_org = df_movies_org[df_movies_org["movie_id"].isin(df_ratings_org.movie_id.unique())]
df_movies_org = df_movies_org.reset_index()

matrix_df = df_ratings_org.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
um_matrix = scipy.sparse.csr_matrix(matrix_df.values)

df_ratings = df_ratings_org[['movie_id', 'rating']].groupby(['movie_id']).size().reset_index(name='counts')
df_ratings = df_ratings.sort_values(by=['counts'], ascending=False)
popular = df_ratings

#model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
#model_knn.fit(um_matrix)


#Create triplets from pairwise matches
pairwise_distances = cosine_similarity(um_matrix)
movies = [i for i in range(0, len(df_movies_org))]
triplets = []
for idx, value in enumerate(pairwise_distances):
    #get closest pairs to create triplets
    max = np.max(np.delete(pairwise_distances[idx], idx))
    index = [i for i in np.argpartition(pairwise_distances[idx], -20)[-20:] if i!=idx]
    if len(index)>=2 and index[0] in movies and index[1] in movies and idx in movies:
        triplets.append([df_movies_org.iloc[idx]["movie_id"], df_movies_org.iloc[index[0]]["movie_id"], df_movies_org.iloc[index[1]]["movie_id"]])
        movies = [i for i in movies if i not in [idx, index[0], index[1]]]
    elif len(index)>=3 and index[0] in movies and index[2] in movies and idx in movies:
        triplets.append([df_movies_org.iloc[idx]["movie_id"], df_movies_org.iloc[index[0]]["movie_id"], df_movies_org.iloc[index[2]]["movie_id"]])
        movies = [i for i in movies if i not in [idx, index[0], index[2]]]
    elif len(index) >= 3 and index[1] in movies and index[2] in movies and idx in movies:
        triplets.append([df_movies_org.iloc[idx]["movie_id"], df_movies_org.iloc[index[1]]["movie_id"], df_movies_org.iloc[index[2]]["movie_id"]])
        movies = [i for i in movies if i not in [idx, index[1], index[2]]]
    elif len(index) >= 3 and index[0] in movies and index[3] in movies and idx in movies:
        triplets.append([df_movies_org.iloc[idx]["movie_id"], df_movies_org.iloc[index[0]]["movie_id"], df_movies_org.iloc[index[3]]["movie_id"]])
        movies = [i for i in movies if i not in [idx, index[0], index[3]]]
    elif len(index) >= 3 and index[1] in movies and index[3] in movies and idx in movies:
        triplets.append([df_movies_org.iloc[idx]["movie_id"], df_movies_org.iloc[index[1]]["movie_id"], df_movies_org.iloc[index[3]]["movie_id"]])
        movies = [i for i in movies if i not in [idx, index[1], index[3]]]
    elif len(index) >= 3 and index[2] in movies and index[3] in movies and idx in movies:
        triplets.append([df_movies_org.iloc[idx]["movie_id"], df_movies_org.iloc[index[2]]["movie_id"], df_movies_org.iloc[index[3]]["movie_id"]])
        movies = [i for i in movies if i not in [idx, index[2], index[3]]]
    else:
        index = [i for i in index if i in movies]
        if len(index)>1 and idx in movies:
            triplets.append([df_movies_org.iloc[idx]["movie_id"], df_movies_org.iloc[index[0]]["movie_id"], df_movies_org.iloc[index[1]]["movie_id"]])
            movies = [i for i in movies if i not in [idx, index[0], index[1]]]

for i in range(0, len(movies), 3):
    if(len(movies[i:i + 3])==3):
        t_3 = [df_movies_org.iloc[x]["movie_id"] for x in movies[i:i + 3]]
        triplets.append(t_3)

with open("/home/binglidev001/movie_recsys/dataset/ml-20m/ml-20m/triplets.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(triplets)

#n_csklearn.metrics.pairwise.cosine_similarity(lusters=10
#kmeans = KMeans(n_clusters=n_clusters, random_state=0)
#clusters = kmeans.fit_predict(um_matrix)
#alldistances = kmeans.fit_transform(um_matrix)
#alldistances = [min(i) for i in alldistances]
#df_movies_org["cluster"]=np.array(clusters)
#df_movies_org["distances"]=alldistances
#df_movies_org = pd.merge(df_movies_org, popular, how='left', on=['movie_id'])
#df_movies_org.to_csv("clustered_movies.csv", index=False)
