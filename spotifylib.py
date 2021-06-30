"""
Python library used for a Spotify AI prediction project

"""

import sqlite3
import re

def connection (query, db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(query)
    ans = cursor.fetchall()
    conn.commit()
    conn.close()
    return ans

def tuple_to_list(tuple_):
    list_ = []
    for line in tuple_:
        line = list(line)
        if len(line) >= 7:
            line[7] = str(line[7])
        list_.append(line)
    return list_

def str_to_list(str_, index, len_):
    for i in range(len_):
        str_[i][index] = str_[i][index].strip("['"+"']")
        str_[i][index] = re.split("', '", str_[i][index])
    return str_

def spitting_id_artists(chansons_):
    add = []
    index_=[]
    for i in range(len(chansons_)):
        line = chansons_[i].copy()
        if len(line[6]) > 1:
            index_.append(i)
            for j in line[6]:
                add_line = line.copy()
                add_line[6] = j
                add.append(add_line)
        else:
            chansons_[i][6] = line[6][0]
    chansons_.extend(add)
    return chansons_, index_
