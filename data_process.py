import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from scipy.interpolate import spline

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 2000)


def readfile(rows):
    global movies, ratings, tags, g_scores, g_tags, links, data_list, data_list_name
    data_list_name = ["movies", "ratings", "tags", "g_scores", "g_tags", "links"]
    movies = pd.read_csv('input/movies.csv', nrows=rows)
    ratings = pd.read_csv('input/ratings.csv', nrows=rows)
    tags = pd.read_csv('input/tags.csv', nrows=rows)
    g_scores = pd.read_csv('input/genome-scores.csv', nrows=rows)
    g_tags = pd.read_csv('input/genome-tags.csv', nrows=rows)
    links = pd.read_csv('input/links.csv', nrows=rows)
    data_list = [movies, ratings, tags, g_scores, g_tags, links]

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

def plot(genre):        

    sns.set_style("white")

    xnew_t1 = np.linspace(t1.year.min(),t1.year.max(),500) #300 represents number of points to make between T.min and T.max
    xnew_t2 = np.linspace(t2.year.min(),t2.year.max(),500) #300 represents number of points to make between T.min and T.max
    xnew_t3 = np.linspace(t3.year.min(),t3.year.max(),500) #300 represents number of points to make between T.min and T.max
    xnew_t4 = np.linspace(t4.year.min(),t4.year.max(),500) #300 represents number of points to make between T.min and T.max
    power_smooth_t1 = spline(t1.year,t1[genre],xnew_t1)
    power_smooth_t2 = spline(t2.year,t2[genre],xnew_t2)
    power_smooth_t3 = spline(t3.year,t3[genre],xnew_t3)
    power_smooth_t4 = spline(t4.year,t4[genre],xnew_t4)

    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()

    offset = 60
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par2,
                                        offset=(offset, 0))

    par2.axis["right"].toggle(all=True)

    offset2 = 120
    new_fixed_axis = par3.get_grid_helper().new_fixed_axis
    par3.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par3,
                                        offset=(offset2, 0))

    par3.axis["right"].toggle(all=True)


    host.set_xlabel("Year")
    host.set_ylabel("Production")
    par1.set_ylabel("Score")
    par2.set_ylabel("Views")
    par3.set_ylabel("AvgViews")

    p1, = host.plot(xnew_t1, power_smooth_t1, label="Production")
    p2, = par1.plot(xnew_t2, power_smooth_t2, label="Score")
    p3, = par2.plot(xnew_t3, power_smooth_t3, label="Views")
    p4, = par3.plot(xnew_t4, power_smooth_t4, label="AvgViews")

    host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.get_color())
    par3.axis["right"].label.set_color(p4.get_color())

    fig1 = plt.gcf()

    fig1.savefig(str(genre+'.jpg'), format='jpg', dpi=500)
    plt.clf()


def plot2(genre):        

    sns.set_style("white")
    
    xnew_t1 = np.linspace(t1.year.min(),t1.year.max(),300) #300 represents number of points to make between T.min and T.max
    xnew_t2 = np.linspace(t2.year.min(),t2.year.max(),300) #300 represents number of points to make between T.min and T.max
    xnew_t3 = np.linspace(t3.year.min(),t3.year.max(),300) #300 represents number of points to make between T.min and T.max
    xnew_t4 = np.linspace(t4.year.min(),t4.year.max(),300) #300 represents number of points to make between T.min and T.max
    power_smooth_t1 = spline(t1.year,t1[genre],xnew_t1)
    power_smooth_t2 = spline(t2.year,t2[genre],xnew_t2)
    power_smooth_t3 = spline(t3.year,t3[genre],xnew_t3)
    power_smooth_t4 = spline(t4.year,t4[genre],xnew_t4)


    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.7)

    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()

    offset = 60
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par2,
                                        offset=(offset, 0))

    par2.axis["right"].toggle(all=True)

    offset2 = 120
    new_fixed_axis = par3.get_grid_helper().new_fixed_axis
    par3.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par3,
                                        offset=(offset2, 0))

    par3.axis["right"].toggle(all=True)


    host.set_xlabel("Year")
    host.set_ylabel("Production")
    par1.set_ylabel("Score")
    par2.set_ylabel("Views")
    par3.set_ylabel("AvgViews")

    p1, = host.plot(xnew_t1, power_smooth_t1, label="Production")
    p2, = par1.plot(xnew_t2, power_smooth_t2, label="Score")
    p3, = par2.plot(xnew_t3, power_smooth_t3, label="Views")
    p4, = par3.plot(xnew_t4, power_smooth_t4, label="AvgViews")

    plt.ylim(ymin = 0)
    par1.set_ylim(ymin = 0, ymax=5)

    host.legend()

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.get_color())
    par3.axis["right"].label.set_color(p4.get_color())

    fig1 = plt.gcf()
    #plt.show()
    #plt.draw()
    fig1.savefig(str(genre+'2.jpg'), format='jpg', dpi=500)
    plt.clf()


if __name__ == "__main__":
    testing = True
    if testing:
        readfile(100)
    else:
        readfile(30000000)

    for i in range(len(data_list)):
        #print ("-----------")
        #print (data_list_name[i])
        #print (data_list[i].isnull().any())
        #print (data_list[i].shape)
        data_list[i] = data_list[i].dropna(axis=0, how='any')
        #print ("Clean...")
        #print (data_list[i].shape)
        #print (data_list[i].isnull().any())
        #print ("\n\n")
    
    
    movies['year'] = movies.apply(lambda row: get_year(row.title), axis=1)
    movies = movies.set_index(movies['movieId'])

    #table1
    t1 = movies.genres.dropna().str.split('|').apply(f)
    t1 = t1.set_index(movies.year)
    print(t1.head(5))
    t1 = t1.groupby('year').sum().reset_index().fillna(0)
    col_list= list(t1)
    col_list.remove('year')
    t1['total'] = t1[col_list].sum(axis=1)
    t1 = t1[t1.year < 2010]
    

    #print (ratings)
    t2 = ratings.groupby('movieId').agg({'userId':'count', 'rating':'mean'})
    t2.columns = ['ViewCount','avgScore']

    t3 = movies.genres.dropna().str.split('|').apply(f)
    t3.set_index(movies.movieId)
    t3.insert(0, 'year', movies.year)

    t2 = t2.join(t3, lsuffix='movieId', rsuffix='movieId')
    #t2 = t2.drop('title', 1)
    #print (t2)
    t3 = t2.copy()
    t4 = t2.copy()

    for i in t2.columns:
        if i in ['ViewCount', 'avgScore', 'year']:
            continue
        t2[i] = t2[i]*t2.avgScore
    t2 = t2.drop(['ViewCount', 'avgScore'], axis=1)
    t2 = t2.groupby('year').mean().reset_index()
    t2['total'] = t2[col_list].mean(axis=1)
    t2 = t2[t2.year < 2010]
    
    for i in t3.columns:
        if i in ['ViewCount', 'avgScore', 'year']:
            continue
        t3[i] = t3[i]*t3.ViewCount
    t3 = t3.drop(['ViewCount', 'avgScore'], axis=1)  
    t3 = t3.groupby('year').sum().reset_index().fillna(0)
    t3['total'] = t3[col_list].sum(axis=1)
    t3 = t3[t3.year < 2010]
    
    for i in t4.columns:
        if i in ['ViewCount', 'avgScore', 'year']:
            continue
        t4[i] = t4[i]*t4.ViewCount
    t4 = t4.drop(['ViewCount', 'avgScore'], axis=1)  
    t4 = t4.groupby('year').sum().reset_index().fillna(0)
    t4['total'] = t4[col_list].sum(axis=1)
    for i in t4.columns:
        if i in ['year']:
            continue
        t4[i] = t4[i]/t1[i]

    t4 = t4[t4.year < 2010]

    t1 = t1[t1.year > 1990]
    t2 = t2[t2.year > 1990]
    t3 = t3[t3.year > 1990]
    t4 = t4[t4.year > 1990]

    print (">>>>>>>>Number of Movies<<<<<<<<<")
    print (t1.head(10))

    print ("\n\n>>>>>>>>Score<<<<<<<<<")
    print (t2.head(10))
    
    print ("\n\n>>>>>>>>Count<<<<<<<<<")
    print (t3.head(10))

    print ("\n\n>>>>>>>>AvgView<<<<<<<<<")
    print (t4.head(10))
    
    
    for i in t4.columns:
        if i == 'year':
            continue
        plot2(i)
    
    '''
    sns.set_style("white")
    
    plt.figure(1)
    plt.subplot(111)
    for i in t4.columns:
        if i in ['year', 'total']:
            continue   
        xnew_t1 = np.linspace(t1.year.min(),t1.year.max(),2)
        power_smooth_t1 = spline(t1.year,t1[i],xnew_t1)
        plt.plot(xnew_t1, power_smooth_t1, label=str(i))

    #plt.plot(t1.year, t1.Adventure, color='blue')
    #plt.plot(t1.year, t1.Drama, color='red')
    '''

    '''
    
    '''
    '''
    plt.legend()
    plt.show()
    '''