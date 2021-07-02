import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import spotifylib as sp

artists_tuple = sp.connection("SELECT * FROM artistes","projetspotify.db")
artists_columns = ['id_x','followers','genres','name','popularity']
artists_ = sp.tuple_to_list(artists_tuple)
artists = pd.DataFrame(artists_, columns = artists_columns)

chansons_tuple = sp.connection("SELECT * FROM chansons", "projetspotify.db")

chansons_columns = ['id_y','name','popularity','duration_ms','explicit','artists'
                    ,'id_artists','release_date','danceability','energy','key'
                    ,'loudness','mode','speechiness','acousticness'
                    ,'instrumentalness','liveness','valence','tempo','time_signature']
chansons_ = sp.tuple_to_list(chansons_tuple)

chansons_ = sp.str_to_list(chansons_, 6, len(chansons_))
chansons_, index_ = sp.spitting_id_artists(chansons_)
artists_ = sp.str_to_list(artists_, 2, len(artists_))

chansons_list_bis = sp.tuple_to_list(chansons_tuple)
chansons_bis = pd.DataFrame(chansons_list_bis, columns = chansons_columns)
chansons_bis['release_date'] = pd.to_datetime(chansons_bis['release_date'])
chansons_bis['year'] = chansons_bis['release_date'].dt.year

chansons = pd.DataFrame(chansons_, columns = chansons_columns)
chansons['release_date'] = pd.to_datetime(chansons['release_date'])
chansons['year'] = chansons['release_date'].dt.year
rows = chansons.index[index_]
chansons.drop(rows, inplace=True)

fusion = artists.merge(chansons, how="left", left_on=("id_x"), right_on=("id_artists"))
fusion_cleaned = fusion.dropna(axis = 0, how = 'any')

top_songs = fusion_cleaned.sort_values('popularity_y', ascending=False).reset_index(drop=True).head(50)

def danceability_by_tempo(chansons):
    danceability_by_tempo = chansons.groupby('tempo')['danceability'].sum()
    print(danceability_by_tempo)
    danceability_by_tempo.plot()

def plot_danceability_over_year(chansons):
    danceability_by_time = chansons.groupby('year')['danceability'].mean()
    print (danceability_by_time)
    danceability_by_time.plot()

def artists_by_year(chansons_bis):
    artists_by_year = chansons_bis.groupby('year')['artists'].nunique()
    artists_by_year.plot(title='Artists over time')
    
def plots(chansons_bis):
    year_avg = chansons_bis[['danceability'
                     ,'energy'
                     ,'liveness'
                     ,'acousticness'
                     ,'valence'
                     ,'year'
                     ,'loudness']].groupby('year').mean().sort_values(by='year').reset_index()
    plt.figure()
    plt.title('The trends of songs over time', fontsize = 15)
    lines = ['danceability', 'energy', 'liveness', 'acousticness', 'valence','loudness']
    for line in lines:
        sb.lineplot(x='year',y=line, data=year_avg)
    plt.legend(lines)
    
    plt.figure()
    sb.countplot(x="mode", data=chansons_bis)
    
    plt.figure()
    sb.boxplot(y=chansons_bis['danceability'])
    
    plt.figure()
    
    corr = chansons_bis[['acousticness','danceability','energy','instrumentalness','liveness','tempo']].corr()
    sb.set()
    plt.figure()
    sb.heatmap(corr, cmap='YlGnBu', annot=True)

def exploration (chansons):
    print(chansons['duration_ms'].mean())
    time_by_year = chansons.groupby('popularity')['duration_ms'].mean()
    time_by_year.plot()
    plt.figure()
    time_by_year = chansons.groupby('popularity')['energy'].mean()
    time_by_year.plot()

def songs_by_artists(fusion_cleaned):
    songs_by_artists = fusion_cleaned.groupby('id_x')['id_y'] \
                                     .count() \
                                     .reset_index() \
                                     .sort_values('id_y', ascending=False)
    return songs_by_artists

def description(df):
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.max_columns', 17)
    fusion.describe()
    return df.describe()

def exploration_more(fusion_cleaned):
    danceability_by_tempo = fusion_cleaned.groupby('tempo')['danceability'].mean()
    danceability_by_tempo.plot(title='Danceability by tempo')
    plt.figure()
    result = songs_by_artists(fusion_cleaned)
    
    followers_by_popularity = fusion_cleaned.groupby('popularity_y')['popularity_x'].mean()
    followers_by_popularity.plot()
    
    plt.figure()
    
    followers_by_popularity = fusion_cleaned.groupby('energy')['loudness'].mean()
    followers_by_popularity.plot()
    
    plt.figure()
    plots(fusion_cleaned)
    
    fusion_corr = fusion_cleaned.select_dtypes(exclude=[object])
    fusion_corr = fusion_corr.corr()
    fusion_corr = round(fusion_corr, 3)
    fusion_corr.corr().loc[:,'popularity_y'].abs().sort_values(ascending=False)[1:]
    
    fusion_corr = fusion_cleaned[['explicit','mode','time_signature','valence','liveness','tempo','speechiness','duration_ms','key']].corr()
    sb.set()
    plt.figure()
    sb.heatmap(fusion_corr, cmap='YlGnBu', annot=True)

def modelisation__before(top_songs):
    top_songs.describe()
    columns_names = ['acousticness','danceability','popularity_y','instrumentalness','energy','valence']
    for i in columns_names:
        column_count = top_songs[i].value_counts().sort_index()
        sb.distplot(top_songs[i])
        top_songs[i].describe()
        plt.figure()
        sb.boxplot(y=top_songs[i])
        if i != 'popularity_y':
            plt.figure()
            sb.regplot(x=i, y='popularity_y', scatter=True, fit_reg=False, data=top_songs)
        plt.figure()
    select_corr = top_songs[columns_names].select_dtypes(exclude=[object])
    corr_ = select_corr.corr()
    round(corr_, 3)
    sb.set()
    plt.figure()
    sb.heatmap(corr_, cmap='YlGnBu', annot=True)

def modelisation_lin_reg(top_songs):
    top_songs_cleaned = top_songs.dropna(axis=0)
    
    x1 = top_songs_cleaned.drop(['popularity_y'], axis='columns', inplace=False)
    y1 = top_songs_cleaned['popularity_y']
    
    popularity = pd.DataFrame({"Before":y1, "After":np.log(y1)})
    
    popularity.hist()
    
    x_columns = ['acousticness','danceability','instrumentalness','energy','valence']
    
    y1 = np.log(y1)
    x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state = 3)
    x_train = x_train[x_columns]
    x_test = x_test[x_columns]
    
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    
    
    base_pred = np.mean(y_test)
    base_pred = np.repeat(base_pred, len(y_test))
    base_rmse = np.sqrt(mean_squared_error(y_test, base_pred))
    lgr = LinearRegression(fit_intercept=True)
    model_lin = lgr.fit(x_train, y_train)
    
    top_songs_cleaned_pred = lgr.predict(x_test)
    
    lin_mse = mean_squared_error(y_test, top_songs_cleaned_pred)
    lin_msre = np.sqrt(lin_mse)
    
    r2_lin_test = model_lin.score(x_test, y_test)
    r2_lin_train = model_lin.score(x_train, y_train)
    print("r² of test: ", r2_lin_test, "\nr² of train: ",r2_lin_train)

    residuals = y_test - top_songs_cleaned_pred
    sb.regplot(x=top_songs_cleaned_pred, y=residuals, scatter=True, fit_reg=False)