# -*- coding: utf-8 -*-
import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from sklearn import *
from flask import Flask,request,jsonify,render_template,redirect,url_for



app = Flask(__name__, static_folder='static')
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "La3IiYTExqaKp-e-AtUKXO6B8yHx03IS8fHXJxZAnkbW"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/pred',methods=["POST","GET"])
def pred():
   inp_feature = [x for x in request.form.values()]
   inp_feature=inp_feature[:16]
   print(inp_feature)
   scale= pickle.load(open("./scale.pkl","rb"))

   feature_values = [np.array(inp_feature)]


   names = [['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3am','Humidity9am','Humidity3am','Pressure9am','Pressure3am','Cloud9am','Cloud3am','Temp9am','Temp3am']]

   data = pandas.DataFrame(feature_values,columns=names)
   data = scale.fit_transform(data)
   print(data)
   data = pandas.DataFrame(data,columns=names)
   print(data)
   header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
   payload_scoring = {"input_data": [{"fields": [['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3am','Humidity9am','Humidity3am','Pressure9am','Pressure3am','Cloud9am','Cloud3am','Temp9am','Temp3am']], "values": [inp_feature]}]}

   response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/2a5b30f2-eb37-4c7a-83ed-2bdb4fa5986b/predictions?version=2022-11-17', json=payload_scoring,
   headers={'Authorization': 'Bearer ' + mltoken})
   print("Scoring response")
   print(response_scoring.json())
   pred=response_scoring.json()
   output=pred['predictions'][0]['values'][0][0]
   print(output)
   if output == "Yes":
      return render_template("predict1.html")
   else:
      return render_template("predict2.html")





if __name__ == '__main__':
   app.run(debug= True)