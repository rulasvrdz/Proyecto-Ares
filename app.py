import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from fuzzywuzzy import process

# Instances for FE
labelE = preprocessing.LabelEncoder()
scaler = MinMaxScaler()

# Important functions
def tagToIndex(i):
    r = df[df['tag'] == i].index
    return r[0]

def listToTag(a,b):
    data = []
    
    for i in range(len(a[0])-1):
        r = b[0][i+1]
        data.append({
            'name': df.iloc[[r]].name.values[0],
            'tag': df.iloc[[r]].tag.values[0],
            'simil': 1-a[0][i+1] })
    return data

# Data Collection
df = pd.read_csv('players.csv')

# Feature Engineering
df_r = df.drop(['name','name_clan','name_arena','tag'], axis = 1)

df_r['tag_clan'] = df_r['tag_clan'].fillna('#NoClan')
df_r['name_pais'] = labelE.fit_transform(df_r['name_pais'])
df_r['tag_clan'] = labelE.fit_transform(df_r['tag_clan'])

scaler.fit(df_r)
values = scaler.transform(df_r)

df_F = pd.DataFrame(values)
df_F.columns = ['name_pais', 'rank', 'expLevel', 'trophies', 'tag_clan', 'id_arena']

 # Model
neigh = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
neigh.fit(df_F)

# Use Flask
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

@app.route('/recommend',methods=['POST'])
def recommend():
    input = list(request.form.values())

    name = input[0]
    n = int(input[1])

    true_name = process.extractOne(name, df['tag'].values)
    id = tagToIndex(true_name[0])

    a, b = neigh.kneighbors(df_F.iloc[[id]], n_neighbors=n+1)

    res = true_name

    return render_template('index.html', res = res)
