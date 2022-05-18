import pandas as pd
from fuzzywuzzy import process
import pickle

# Instances for FE

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
            'simil': "{:.2f}".format((1-a[0][i+1])*100) })
    return data

df_F=pd.read_pickle('finalDf.pkl')
df=pd.read_pickle('originalDf.pkl')

 # Model
neigh = pickle.load(open('KNN.sav', 'rb'))

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

    res = listToTag(a,b)

    return render_template('index.html', res = res)
