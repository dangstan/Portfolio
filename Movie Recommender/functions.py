import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import Javascript, display, Image, clear_output, Markdown, Latex, HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyautogui
import webbrowser
import nbformat as nbf
import codecs



def getting_dataframes(path):
    movies = pd.read_csv(path  + 'movies.csv')
    ratings = pd.read_csv(path + 'ratings.csv')
    df_merged = pd.merge(movies, ratings)
    print('How many reviews are we dealing with? ', len(ratings), '\nAnd how many movies? ', len(movies))
    
    ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId' )

    movie_by_user = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
    
    return ratings, df_merged, movie_by_user



def most_common(df, group):
    df[group].to_csv('test.txt', header=False, index=False)

    common = open("test.txt", 'r')
    common = common.read()
    common = common.replace('\n','|').split('|')
    common = pd.DataFrame(common)
    return common


# In[127]:


def getting_and_plotting_genres(dataframe, df_merged, show):

    genres = most_common(df_merged, 'genres')[:-1]

    generos = list(genres[0].values)

    if show:
        genres_to_plot = pd.DataFrame({'title': genres[0].value_counts().keys(), 'count': genres[0].value_counts().values})

        fig = plt.figure(1, figsize=(9,6.5))
        ax2 = fig.add_subplot(2,1,2)
        y_axis = genres_to_plot.iloc[:,1]
        x_axis = genres_to_plot.index
        x_label = genres_to_plot.iloc[:,0]
        plt.xticks(rotation=85, fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.xticks(x_axis, x_label)
        plt.ylabel("Ocorrências", fontsize = 24, labelpad = 0)
        ax2.bar(x_axis, y_axis, align = 'center')
        plt.title("Popularidade dos gêneros",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 25)

        plt.show()
    
    return genres[0].value_counts().keys()[:8]


# In[128]:


def dummizing_dataframe(dataframe, generos):
    for item in generos:
        dataframe.loc[dataframe['genres'].str.contains(item, na = False), item] = 1
        dataframe.loc[item] = dataframe[item].fillna(0)
        
    dataframe.iloc[:,6:] = dataframe.iloc[:,6:].fillna(0).astype('int')
    dataframe = dataframe.dropna()
    return dataframe    


# In[129]:


def averaging_genre_by_user(genres, dataframe):
    
    user_ratings = pd.DataFrame()
    for item in genres:
        temp = dataframe[dataframe['genres'].str.contains(item)]
        temp = temp[['userId', 'rating']].groupby(['userId']).mean().round(2)
        user_ratings = pd.concat([user_ratings, temp], axis = 1)


    columns = []    
    for nome in genres:
        columns.append('media_' + nome)

    user_ratings.columns = columns    
    
    return user_ratings    


# In[130]:


def averaging_dummmized_movies(genres, dataframe):
    
    movie_ratings = pd.DataFrame()
    for item in genres:
        temp = dataframe[dataframe['genres'].str.contains(item)]
        temp = temp[['title', 'rating'] + list(genres)].groupby(['title']).mean().round(2)
        movie_ratings = pd.concat([movie_ratings, temp], axis = 0)

    movie_ratings = movie_ratings.loc[~movie_ratings.index.duplicated(keep='first')]
    
    return movie_ratings


# In[131]:


def evaluating_genres_correlation(dataframe, show):
    
    import statsmodels.formula.api as smf

    dataframe = dataframe.rename(columns={"media_Sci-Fi": "media_SciFi"})

    if show:
        print('\n')
        print("Genre's correlation:\n")
    
        display(dataframe.corr())
        print('\n')
        print('Logistic Regression for media_Drama as target:\n')

        reg_user = smf.ols('media_Drama ~ media_Comedy + media_Thriller + media_Action + media_Romance + media_Adventure + media_Crime + media_SciFi', data = dataframe).fit()
    
        print(reg_user.summary())
        
        print('\n')
    
        print('Logistic Regression for media_SciFi as target:\n')
    
        reg_user = smf.ols('media_SciFi ~ media_Comedy + media_Thriller + media_Action + media_Romance + media_Adventure + media_Crime + media_Drama', data = dataframe).fit()
    
        print(reg_user.summary())
    
        print('\n')
    
        print('Logistic Regression for media_Action as target:\n')
    
        reg_user = smf.ols('media_Action ~ media_Comedy + media_Thriller + media_SciFi + media_Romance + media_Adventure + media_Crime + media_Drama', data = dataframe).fit()
    
        print(reg_user.summary())
    
        print('\n')
    
        print('Logistic Regression for media_Comedy as target:\n')
    
        reg_user = smf.ols('media_Comedy ~ media_Action + media_Thriller + media_SciFi + media_Romance + media_Adventure + media_Crime + media_Drama', data = dataframe).fit()
    
        print(reg_user.summary())
    
        print('\n')
    
        print('Logistic Regression for media_Romance as target:\n')
    
        reg_user = smf.ols('media_Romance ~ media_Comedy + media_Thriller + media_SciFi + media_Action + media_Adventure + media_Crime + media_Drama', data = dataframe).fit()
    
        print(reg_user.summary())
    
        print('\n')
    
        print('Logistic Regression for media_Thriller as target:\n')
    
        reg_user = smf.ols('media_Thriller ~ media_Comedy + media_Action + media_SciFi + media_Romance + media_Adventure + media_Crime + media_Drama', data = dataframe).fit()
    
        print(reg_user.summary())
    
        print('\n')
    
        print('Logistic Regression for media_Adventure as target:\n')
    
        reg_user = smf.ols('media_Adventure ~ media_Comedy + media_Thriller + media_SciFi + media_Romance + media_Action + media_Crime + media_Drama', data = dataframe).fit()
    
        print(reg_user.summary())
        
    
    


# ### Primeira função geral

# In[132]:


def analysing_dataframes(path, show = False):

    ratings,df_merged, movie_by_user = getting_dataframes(path) # load dataframes and do transformations to allow for analysis 
    
    common_users = most_common(ratings, 'userId')
    
    generos = getting_and_plotting_genres(common_users, df_merged, show) # plot distribution of genres in dataframe

    df_merged = dummizing_dataframe(df_merged, generos) # get dummies for each genre in each movie
    
    user_ratings = averaging_genre_by_user(generos, df_merged) # get average rating for genre by user

    movie_ratings = averaging_dummmized_movies(generos, df_merged) # get average rating for each movie

    evaluating_genres_correlation(user_ratings, show) # logistic regression to analyse correlations between genres
    
    return common_users, movie_by_user, movie_ratings, user_ratings


# In[133]:


def votes_evaluation(movie_by_user, movie_ratings, user_ratings):
    
    counter = user_ratings.index.astype(int)

    not_seen_dict = {}

    seen_dict = {}

    for n in range(len(counter)):

        id_n = int(counter[int(n)])

        nan_movies = movie_by_user.loc[int(id_n),:].isna()

        nan_movies = nan_movies[nan_movies == True]

        movies_seen = [x for x in movie_by_user.columns if x not in nan_movies]

        seen_dict[id_n] =  list(movies_seen)

        not_seen_dict[id_n] =  list(nan_movies.index)

        lista_cont = movie_ratings[movie_ratings.index.isin(movies_seen)].iloc[:,1:].sum() 

        user_ratings.iloc[int(id_n)-1,:] = np.array((user_ratings.iloc[int(id_n)-1,:].values.T*lista_cont)/max(lista_cont))

    return seen_dict, not_seen_dict,user_ratings


# In[134]:


def similarity_eval(user_ratings, id_n):
    
    user_ratings = user_ratings.apply(lambda row: row.fillna(row.mean()), axis=1)

    cosine = cosine_similarity(user_ratings)
    
    np.fill_diagonal(cosine, 0 )
    similarity_with_user =pd.DataFrame(cosine,index=user_ratings.index)
    similarity_with_user.columns=user_ratings.index
    
    most_related_users = similarity_with_user.iloc[id_n-1,:]
    most_related_users = most_related_users.sort_values(ascending = False)[most_related_users>0.8]
    
    return most_related_users


# In[135]:


def recommending(most_related_users, movie_by_user, not_seen_dict, id_n, boolean = False):    

    movies_to_watch = []
    movies_to_def_watch = []

    movie_by_related_user = movie_by_user[movie_by_user.index.isin(list(most_related_users.index))]
    movie_means_for_related_user = movie_by_related_user.iloc[:,:].mean()[movie_by_related_user.iloc[:,:].mean().notnull()]
    movies_to_watch = movie_means_for_related_user[(movie_means_for_related_user>4) & (movie_means_for_related_user<4.81)]
    movies_to_def_watch = movie_means_for_related_user[movie_means_for_related_user>4.8]
    print('\n')
    print("Out of " + str(len(not_seen_dict[id_n])) + " movies, " + str(len(movies_to_watch)) + " fits the user's taste.")
    print("From users with similar taste, " + str(len(movies_to_def_watch)) + " movies were found to be greatly enjoyed by a lot of them.\n")
    
        
    
    movies_list_join = list(zip(list(movies_to_watch.index),list(movies_to_def_watch.index)+['']*(len(movies_to_watch.index)-len(movies_to_def_watch.index))))
    
    file = open("copy.txt", "w") 
    file.write(str(movies_list_join))
    file.close() 
    
    table = '''<table>
  <tr>
    <th>Movies you'll probably enjoy:</th>
    <th>Movies you'll definitely enjoy:</th>
  </tr>'''
  

    for i in movies_list_join:

        table += '\n'

        table += '''<tr>
        <td>'''+i[0]+'''</td>
        <td>'''+i[1]+'''</td>
      </tr>'''

    table += '''
    </table>'''
    
    display(HTML(table))
    
    if boolean is False:
        return movies_to_watch, movies_to_def_watch


# ### Segunda função geral

# In[136]:


def recommendation(n, movie_by_user, movie_ratings, user_ratings):
    
    seen_dict, not_seen_dict,user_ratings = votes_evaluation(movie_by_user, movie_ratings, user_ratings) # get seen and unseen movies and each genre's rating average for every user
    
    id_n = int(user_ratings.reset_index().loc[n,'userId']) # get user n's id
    
    most_related_users = similarity_eval(user_ratings, id_n) # calculate similarity correlation between users
    
    return recommending(most_related_users, movie_by_user, not_seen_dict, id_n, True)


# In[137]:


def get_most_watched(movie_by_user):
    movie_count = movie_by_user.copy()
    movie_count.fillna(0, inplace = True)
    movie_count = pd.DataFrame(np.where(movie_count!=0,1,0), index = movie_by_user.index, columns = movie_by_user.columns)
    most_watched = movie_count.sum(axis=0).sort_values()
    return most_watched.reset_index()


# In[138]:


def profile(movie_name, movie_by_user, rate, most_watched):

    rate_list = [rate-1,rate,rate+1]
    
    movie_same_rate = movie_by_user[movie_by_user[movie_name].isin(rate_list)].index
    
    movie_means_for_related_user = movie_by_user.loc[movie_same_rate,:].mean()[movie_by_user.loc[movie_same_rate,:].mean().notnull()]

    
    movie_means_for_related_user = movie_means_for_related_user[((movie_means_for_related_user>rate-1.2) & (movie_means_for_related_user<rate+1.2)) | (movie_means_for_related_user<1.5) | (movie_means_for_related_user>4.5)]
        
    
    most_similar = movie_means_for_related_user.reset_index()
    most_similar.rename(columns = {0:'rates'}, inplace = True)
    watches = get_most_watched(movie_by_user.loc[:,movie_by_user.columns.isin(most_similar['title'])])

    
    watches.rename(columns = {0:'count'}, inplace = True)
    most_similar = most_similar.merge(watches, on = 'title')

    
    most_similar = most_similar.sort_values(['count','rates'], ascending = False)
    
    most_similar = most_similar[~most_similar['title'].isin(most_watched)][::5]
    
    if len(most_similar)>5:
        most_similar = most_similar.iloc[:25,:]['title']
    else:
        most_similar = most_similar['title']
        
    return list(most_similar)


# In[148]:


def run_checkbox_1(ev):
    display(Javascript('IPython.notebook.execute_cells([2])'))

def run_checkbox_4(ev):
    display(Javascript('IPython.notebook.execute_cells([5])'))
    clear_output(wait=True)
    image = Image(filename='arrow_up.png', width = 40, height = 40)
    display(image, image, image, image, image, image)
    time.sleep(1.5)
    clear_output(wait=False)

def run_checkbox_5(ev):
    display(Javascript('IPython.notebook.execute_cells([6])'))
    clear_output(wait=True)
    image = Image(filename='arrow_down.png', width = 40, height = 40)
    display(image, image, image, image, image, image)

def run_checkbox_6_7(ev):
    display(Javascript('IPython.notebook.execute_cells([7,8])'))
    clear_output(wait=True)
    image = Image(filename='arrow_down.png', width = 40, height = 40)
    display(image, image, image, image, image, image)


# In[141]:


def request(most_rated, ratings):

    names = []
    checkbox_objects = []
    for i in most_rated:
        checkbox_objects.append(widgets.Checkbox(value=False, description=i))
        names.append(i)  

    arg_dict = {names[i]: checkbox for i, checkbox in enumerate(checkbox_objects)}

    widget = widgets.VBox(children=checkbox_objects)
    
    button = widgets.Button(description="Go")
    button.on_click(run_checkbox_5)

    if len(ratings) > 30:
        button2 = widgets.Button(description="That's enough")
        button2.on_click(run_checkbox_6_7)

        display(widget, button, button2)

    else:
        display(widget, button)
    
    return widget


# In[142]:


def rating(lista):
    temp_dict = {}
    i = 0
    while i < len(lista):
        rate = input('From 1 to 5, what would be your rate for ' + lista[i] + ':')
        print('\n')
        time.sleep(.6)
        clear_output(wait=True)
        try:
            aux = float(rate)
        except:
            print('Not a valid rate. Try again.')
            continue
        temp_dict[lista[i]] = aux
        i +=1
    return temp_dict

def getting_selected(idn,widget, movie_by_user, temp, ratings, most_watched):
    
    selected_options = [w.description for w in widget.children if w.value]

    rates = rating(selected_options)

    ratings = {**rates, **ratings}

    temp = temp + most_watched


    similar = []
    for k,v in rates.items():
        movie_by_user.loc[idn,k] = v
        similar += profile(k,movie_by_user,v, temp)

    most_watched = list(set(similar)-set(temp))


    button = widgets.Button(description="Go")
    button.on_click(run_checkbox_4)
    display(button)
    
    return most_watched, temp, ratings

def profiling_new_user(ratings,movie_ratings, user_ratings):
    
    the_seen = pd.DataFrame.from_dict(ratings, orient = 'index', columns = ['rates']).reset_index()
    the_seen.rename(columns = {'index':'title'}, inplace = True)
    the_seen = the_seen.merge(movie_ratings.drop('rating', axis = 1).reset_index(), on = 'title')  
    
    seen_genres = pd.DataFrame()
    for item in the_seen.columns[2:]:
        temp = the_seen[the_seen[item] == 1.0]
        temp = temp[[ 'rates']].mean().round(2)
        seen_genres = pd.concat([seen_genres, temp], axis = 1)

    columns = []    
    for nome in the_seen.columns[2:]:
        columns.append('media_' + nome)

    seen_genres.columns = columns   

    user_ratings.loc[len(user_ratings),:]=seen_genres.values
    
    return user_ratings

