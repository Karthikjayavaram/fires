from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
application = Flask(__name__)
app=application

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictData",methods=['Get','POST'])
def predict_datapoint():
        if request.method=="POST":
             Tempertaure=float(request.form.get('Temperature'))
             Rh=float(request.form.get('Rh'))
             Ws=float(request.form.get('Ws'))
             Rain=float(request.form.get('Rain'))
             FFMC=float(request.form.get('FFMC'))
             DMC=float(request.form.get('DMC'))
             ISI=float(request.form.get('ISI'))
             Region=float(request.form.get('Region'))

             new_scaled_data=standard_scaler.transform([[Tempertaure,Rh,Ws,Rain,FFMC,DMC,ISI,Region]])
             result=ridge_model.predict(new_scaled_data)
             
             return render_template('home.html',results=result[0])

        else:
            return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")