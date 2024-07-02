from flask import Flask,render_template,request

import pickle
import numpy as np
import pandas as pd
app=Flask(__name__)
model=pickle.load(open("Randf.pkl","rb"))
@app.route('/')
def home():
    return render_template("home.html")
@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict',methods=["POST","GET"])
def predict():
    return render_template("predict.html")

@app.route('/submit',methods=["POST","GET"])
def submit():
    # reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values()]
    input_feature=[np.array(input_feature)]
    names=['is_ecommerce','is_otherstate','has_VC','has_angel','has_roundA','has_roundB' , 'has_roundC', 'has_roundD', 'is_top500', 'relationships', 'funding_rounds', 'milestones']
    
    print("Number of columns in names:", len(names))
    print("Number of columns in input_feature:", len(input_feature))
    print("Column names:", names)
    data=pd.DataFrame([input_feature],columns=names)
    pred=model.predict(data)
    if pred==0:
        print(pred)
        return render_template("submit.html", pred="success")
    else:
        return render_template("submit.html", pred="closed")
if __name__ == "__main__":
        app.run(port=5000)

    