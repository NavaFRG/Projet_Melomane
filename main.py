import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
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
                     ,'year']].groupby('year').mean().sort_values(by='year').reset_index()
    plt.figure()
    plt.title('The trends of songs over time', fontsize = 15)
    lines = ['danceability', 'energy', 'liveness', 'acousticness', 'valence']
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

