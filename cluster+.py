import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn import decomposition
from scipy.interpolate import spline
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
import sys

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 2000)


def readfile(rows):
    global movies, ratings, tags, g_scores, g_tags, links, data_list, data_list_name
    data_list_name = ["movies", "ratings", "tags", "g_scores", "g_tags", "links"]
    movies = pd.read_csv('input/movies.csv', nrows=1000000)
    ratings = pd.read_csv('input/ratings.csv', nrows=rows)
    #tags = pd.read_csv('input/tags.csv', nrows=rows)
    #g_scores = pd.read_csv('input/genome-scores.csv', nrows=rows)
    #g_tags = pd.read_csv('input/genome-tags.csv', nrows=rows)
    #links = pd.read_csv('input/links.csv', nrows=rows)
    #data_list = [movies, ratings, tags, g_scores, g_tags, links]
    data_list = [movies, ratings]

def get_year(name):
    try:
        return int(name.split(')')[-2][-4:])
    except:
        pass
        #print (name)
        #return int(9999)

def f(category_list):
    n_categories = len(category_list)
    return pd.Series(dict(zip(category_list, [1]*n_categories)))   


if __name__ == "__main__":
    testing = False
    if testing:
        readfile(1000000)
    else:
        readfile(30000000)

    for i in range(len(data_list)):
        data_list[i] = data_list[i].dropna(axis=0, how='any')
    
    movies['year'] = movies.apply(lambda row: get_year(row.title), axis=1)
    #movies = movies.concat(movies['year'])
    movies = movies.set_index(movies['movieId'])
    
    #table1 : movie and genre
    t1 = movies.genres.dropna().str.split('|').apply(f)
    t1 = t1.set_index(movies.movieId)
    t1 = pd.concat([movies, t1], axis = 1)
    t1 = t1.drop('genres',1)
    #print(t1.head(20))

    #table2 : user and rating
    t2 = ratings
    t2 = t2.set_index(ratings.movieId)
    print (t2.head(20))
    print (t2.shape)
    print ("drop num: " + str(len(t2[t2.rating < 3.6].index)))
    #t2 = t2.drop(t2[t2.rating < 3.6].index)
    lst = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    for i in lst:
        t2 = t2[t2.rating != i]
    print (t2.head(20))
    print (t2.shape)
    #sys.exit()
    
    #table3: user and rating per category
    #print(t1.columns[3:])
    t3 = pd.DataFrame(columns = [t1.columns[3:]])
    t3 = t2.join(t3)
    #print (t3.head(10))

    for i in t3.columns[4:]:
        t3[i] = t1[i][t3['movieId']]*t3['rating']
    
    t3 = t3.groupby('userId').mean().reset_index().fillna(-1)
    t3 = t3.drop(['movieId','timestamp'],1)
    #print ("\n\navg rating of movie per genre per user")
    #print(t3.head(10))

    #table4: number of film per catagory
    t4 = pd.DataFrame(columns = [t1.columns[3:]])
    t4 = t2.join(t4)
    
    for i in t4.columns[4:]:
        t4[i] = t1[i][t4['movieId']]*1

    t4 = t4.groupby('userId').sum().reset_index().fillna(0)
    t4 = t4.drop(['movieId','rating','timestamp','(no genres listed)'],1)
    t5 = pd.DataFrame(t4.userId)
    t4 = t4.drop(['userId'],1)
    #t4.set_index(t4.userId)
    t4['total'] = t4.sum(axis=1)
    print (t4.head(20))
    #print (t4['sum'].head(5))
    #print (t4['Action'].head(5))
    for i in t4.columns[:]:
        t4[i] = t4[i]/t4['total']
    t4 = t4.drop(['total'],1)
    t4 = pd.concat([t5, t4], axis=1)
    t4.set_index(t4.userId)
    print ("\n\nnumber of movie per genre per user")
    print (t4.head(20))

    lst=[1,104,116,180,208,311,384,388,394,424,572,586,632,637,768,775,903,910,1115,1244,1411,1516,1588,1931,1937, 3134,11016, 11902, 15087, 17848, 31379, 33607, 35634, 41241, 47986, 53021, 73701, 74277, 74318, 83560, 96769, 96799, 97922, 103218, 104658, 105145, 105235, 108839, 113216, 116522, 117653, 119414, 124362, 125545,1, 5, 13, 14, 16, 31, 53, 73, 95, 104, 116, 120, 122, 125, 148, 161, 180, 182, 194, 197, 198, 229, 255, 261, 265, 276,  87, 301, 310, 338]
    lst = sorted(set(lst))
    print ("\n\n\n\n")
    print(t4.loc[t4['userId'].isin(lst)])
    print ("\n\n\n\n")

    sys.exit()
    t4_label = t4.iloc[:, :1]
    t4_data = t4.iloc[:, 1:]
    
    print (t4_label.head(10))
    print (t4_data.head(10))

    y = t4_label.as_matrix()
    X = t4_data.as_matrix()
    
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    #X = min_max_scaler.fit_transform(X.T, norm='l2')
    #poly = PolynomialFeatures(degree=3, interaction_only=True)
    X = min_max_scaler.fit_transform(X.T)
    X=X.T
    df = pd.DataFrame(X)

    print (df.head(10))
    '''
    '''
    X= preprocessing.normalize(X.T, norm='l1')
    X=X.T
    df = pd.DataFrame(X)
    '''

    #print (y)
    #print (X)
    #print (df.head(10))

    
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    df = pd.DataFrame(X_r)
    df.columns=['X','Y']
    print (df.head(10))
    #sys.exit()
    print(pca.explained_variance_ratio_)  
    #print (X_r)
    lst=[1,104,116,180,208,311,384,388,394,424,572,586,632,637,768,775,903,910,1115,1244,1411,1516,1588,1931,1937, 3134,11016, 11902, 15087, 17848, 31379, 33607, 35634, 41241, 47986, 53021, 73701, 74277, 74318, 83560, 96769, 96799, 97922, 103218, 104658, 105145, 105235, 108839, 113216, 116522, 117653, 119414, 124362, 125545,1, 5, 13, 14, 16, 31, 53, 73, 95, 104, 116, 120, 122, 125, 148, 161, 180, 182, 194, 197, 198, 229, 255, 261, 265, 276,  87, 301, 310, 338]
    fig, ax = plt.subplots()
    for i in range(len(X_r)):
        ax.scatter(X_r[i][0], X_r[i][1], alpha=.8, s=5)
        if (y[i] in lst):
            ax.annotate(str(y[i]), (X_r[i][0], X_r[i][1]), size=6)
    
    plt.title('PCA')
    fig1 = plt.gcf()
    fig1.savefig(str('PCA4++.jpg'), format='jpg', dpi=800)
    plt.clf()

    
    colmap = {1: '#074358', 2: '#458985', 3: '#FFC897', 4:'#734E67', 5:'#A55C55'}

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df)

    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_
    fig = plt.figure()

    colors = list(map(lambda x: colmap[x+1], labels))

    plt.scatter(df['X'], df['Y'], color=colors, alpha=0.7, s=6)
    for idx, centroid in enumerate(centroids):
        plt.scatter(*centroid, color=colmap[idx+1], s=10)

    plt.title('K-Means')
    fig1 = plt.gcf()
    fig1.savefig(str('K-means++.jpg'), format='jpg', dpi=800)
    plt.clf()
    
    
    Nc = range(1, 20)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    
    score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
    
    pl.plot(Nc,score)
    pl.xlabel('Number of Clusters')
    pl.ylabel('Score')
    pl.title('Elbow Curve')
    fig1 = plt.gcf()
    fig1.savefig(str('Elbow_Curve++.jpg'), format='jpg', dpi=800)
    plt.clf()
    