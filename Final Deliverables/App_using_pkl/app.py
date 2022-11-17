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
model = pickle.load(open("./weather1.pkl","rb"))

scale= pickle.load(open("./weather.pkl","rb"))
encoder = pickle.load(open("weather2.pkl","rb"))

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/pred',methods=["POST","GET"])
def pred():
   inp_feature = [x for x in request.form.values()]
   inp_feature=inp_feature[:18]
   print(inp_feature)

   feature_values = [np.array(inp_feature)]


   names = [['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
             'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
             'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'risk', 'RainToday',
             'WindGustDir', 'WindDir9am', 'WindDir3pm']]

   data = pandas.DataFrame(feature_values,columns=names)
   data = scale.fit_transform(data)
   print(data)
   data = pandas.DataFrame(data,columns=names)
   print(data)
   prediction = model.predict(data)
   print(prediction)
   if prediction == "Yes":
      return render_template("predict1.html")
   else:
      return render_template("predict1.html")





if __name__ == '__main__':
   app.run(debug= True)