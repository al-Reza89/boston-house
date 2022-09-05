from crypt import methods
import imp
import pickle
import re
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app=Flask(__name__)
#load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    # if we get the data first we will do standarization scaling.pkl eita transform korbe
    print(np.array(list(data.values())).reshape(1,-1))
    new_data= scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.prdict(new_data)
    #  2D tai first value return korbo
    print(output[0])
    return jsonify(output[0])




if __name__=="__main__":
    app.run(debug=True)