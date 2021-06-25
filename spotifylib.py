import sqlite3

def connection (query, db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute(query)
    ans = cursor.fetchall()
    
    conn.commit()
    conn.close()
    return ans

def tuple_to_list(tpl):
      length = len(tpl)
      str = []
      
      for i in range (length):
          line = list(tpl[i])
          str.append(line)
      return str
