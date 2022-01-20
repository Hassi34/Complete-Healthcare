#!/usr/bin/env python

from wsgiref import simple_server
from flask_cors import CORS, cross_origin
import os
import flask_monitoringdashboard as dashboard
from flask import Flask, render_template, url_for, redirect, request
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(r"classes/app_helper.py"))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from classes.app_helper import (ValuePredictorCancer, ValuePredictorDiabetes, ValuePredictorKidney,CustomLabelEncode,
                                ValuePredictorHeart, ValuePredictorLiver)
import  joblib
import  numpy as np 

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)

@app.route("/")
@cross_origin()
def index():
    return render_template('index.html')
@app.route("/about")
@cross_origin()
def about_app():
    return render_template('about_app.html')

@app.route("/cancer")
@cross_origin()
def cancer():
    #return render_template(r"C:\Users\Mahesh Sharma\Desktop\HealthApp\Indivisual_Deployment\Breast_Cancer API\cancer_model.pkl")
    return render_template('cancer.html')
@app.route('/predict_cancer', methods = ["POST"])
@cross_origin()
def predict_cancer():
    if request.method == "POST":
        if request.json is not None:
            to_predict_list_cancer = request.json["values"] #{'values':[25.2,16.4,0.3211,....]}
            to_predict_list_cancer = list(map(float, to_predict_list_cancer)) 
            if(len(to_predict_list_cancer)==7):
                result_cancer_rest_api, probabilities_cancer = ValuePredictorCancer(to_predict_list_cancer,7)
            if(int(result_cancer_rest_api)==1):
                prediction = f"Oh, Dear, you have {round(probabilities_cancer[1],2)}% chance of getting the disease. Please consult the doctor immediately"
            else:
                prediction = f"No need to worry, as your probability of getting the disease is {round(probabilities_cancer[1],2)}% which is less than 50%"
            return(prediction)
        elif request.form is not None:
            to_predict_list_cancer = request.form.to_dict()
            to_predict_list_cancer = list(to_predict_list_cancer.values())
            to_predict_list_cancer = list(map(float, to_predict_list_cancer))

            if(len(to_predict_list_cancer)==7):
                result_cancer, probabilities_cancer = ValuePredictorCancer(to_predict_list_cancer,7)
            
            if(int(result_cancer)==1):
                 prediction = f"Oh, Dear, you have {round(probabilities_cancer[1],2)}% chance of getting the disease. Please consult the doctor immediately"
            else:
                prediction = f"No need to worry, as your probability of getting the disease is {round(probabilities_cancer[1],2)}% which is less than 50%"

            return(render_template("predict_cancer.html", prediction_text=prediction)) 
#-----------------------------------------------------------------------------------------------------------------------------   
@app.route("/diabetes")
@cross_origin()
def diabetes():
    return render_template("diabetes.html")

@app.route('/predict_diabetes', methods = ["POST"])
@cross_origin()
def predict_diabetes():
    if request.method == "POST":
        if request.json is not None:
            to_predict_list_diabetes = request.json["values"] 
            to_predict_list_diabetes = list(map(float, to_predict_list_diabetes)) 
            if(len(to_predict_list_diabetes)==8):
                result_diabetes_rest_api, probabilities_diabetes = ValuePredictorDiabetes(to_predict_list_diabetes,8)
            if(int(result_diabetes_rest_api)==1):
                prediction = f"Oh, Dear, you have {round(probabilities_diabetes[1],2)}% chance of getting the disease. Please consult the doctor immediately"
            else:
                prediction = f"No need to worry, as your probability of getting the disease is {round(probabilities_diabetes[1],2)}% which is less than 50%"
            return(prediction)
        elif request.form is not None:
            to_predict_list_diabetes = request.form.to_dict()
            to_predict_list_diabetes = list(to_predict_list_diabetes.values())
            to_predict_list_diabetes = list(map(float, to_predict_list_diabetes))
            if(len(to_predict_list_diabetes)==8):
                result_diabetes, probabilities_diabetes = ValuePredictorDiabetes(to_predict_list_diabetes,8)
            
            if(int(result_diabetes)==1):
                prediction = f"Oh, Dear, you have {round(probabilities_diabetes[1],2)}% chance of getting the disease. Please consult the doctor immediately"
            else:
                prediction = f"No need to worry, as your probability of getting the disease is {round(probabilities_diabetes[1],2)}% which is less than 50%"
            return(render_template("predict_diabetes.html", prediction_text=prediction))
#-----------------------------------------------------------------------------------------------------------------------------
@app.route("/heart")
@cross_origin()
def heart():
    return render_template("heart.html")
@app.route('/predict_heart', methods = ["POST"])
@cross_origin()
def predict_heart():
    if request.method == "POST":
        if request.json is not None:
            to_predict_list_heart = request.json["values"] 
            to_predict_list_heart = list(map(float, to_predict_list_heart)) 
            if(len(to_predict_list_heart)==11):
                result_heart_rest_api, probabilities = ValuePredictorHeart(to_predict_list_heart,11)
            if(int(result_heart_rest_api)==1):
                prediction = f"Oh, Dear, you have {round(probabilities[1],2)}% chance of getting the disease. Please consult the doctor immediately"
            else:
                prediction = f"No need to worry, as your probability of getting the disease is {round(probabilities[1],2)}% which is less than 50%"
            return(prediction)
        elif request.form is not None:
            to_predict_list_heart = request.form.to_dict()
            to_predict_list_heart = list(to_predict_list_heart.values())
            to_predict_list_heart = list(map(float, to_predict_list_heart))
            if(len(to_predict_list_heart)==11):
                result_heart, probabilities = ValuePredictorHeart(to_predict_list_heart,11)
            
            if(int(result_heart)==1):
                prediction = f"Oh, Dear, you have {round(probabilities[1],2)}% chance of getting the disease. Please consult the doctor immediately"
            else:
                prediction = f"No need to worry, as your probability of getting the disease is {round(probabilities[1],2)}% which is less than 50%"
            return(render_template("predict_heart.html", prediction_text=prediction))
#-----------------------------------------------------------------------------------------------------------------------------  
@app.route("/kidney")
@cross_origin()
def Kindeny_fun():
    return render_template("kidney.html")

@app.route('/predict_kidney', methods = ["POST"])
@cross_origin()
def predict_kidney():
    if request.method == "POST":
        if request.json is not None:
            to_predict_list_kidney = request.json["values"] 
            to_predict_list_kidney = list(to_predict_list_kidney)
            if(len(to_predict_list_kidney)==6):
                result_kidney_rest_api, probabilities_kidney = ValuePredictorKidney(to_predict_list_kidney,6)
            if(int(result_kidney_rest_api)==1):
                prediction = f"Oh, Dear, you have {round(probabilities_kidney[1],2)}% chance of getting the disease. Please consult the doctor immediately"
            else:
                prediction = f"No need to worry, as your probability of getting the disease is {round(probabilities_kidney[1],2)}% which is less than 50%"
            return(prediction)
        elif request.form is not None:
            to_predict_list_kidney = request.form.to_dict()
            to_predict_list_kidney = list(to_predict_list_kidney.values())
            to_predict_list_kidney = list(to_predict_list_kidney)
            if(len(to_predict_list_kidney)==6):
                result_kidney, probabilities_kidney = ValuePredictorKidney(to_predict_list_kidney,6)
            if(int(result_kidney)==1):
                prediction = f"Oh, Dear, you have {round(probabilities_kidney[1],2)}% chance of getting the disease. Please consult the doctor immediately"
            else:
                prediction = f"No need to worry, as your probability of getting the disease is {round(probabilities_kidney[1],2)}% which is less than 50%"
            return(render_template("predict_kideny.html", prediction_text=prediction)) 
#-----------------------------------------------------------------------------------------------------------------------------  
@app.route("/liver")
@cross_origin()
def liver():
    return render_template("liver.html")

@app.route('/predict_liver', methods = ["POST"])
@cross_origin()
def predict_liver():
    if request.method == "POST":
        if request.json is not None:
            to_predict_list_liver = request.json["values"] 
            to_predict_list_liver = list(to_predict_list_liver)
            if(len(to_predict_list_liver)==7):
                result_liver_rest_api, probabilities_liver = ValuePredictorLiver(to_predict_list_liver,7)
            if(int(result_liver_rest_api)==1):
                prediction = f"Oh, Dear, you have {round(probabilities_liver[1],2)}% chance of getting the disease. Please consult the doctor immediately"
            else:
                prediction = f"No need to worry, as your probability of getting the disease is {round(probabilities_liver[1],2)}% which is less than 50%"
            return(prediction)
        elif request.form is not None:
            to_predict_list_liver = request.form.to_dict()
            to_predict_list_liver = list(to_predict_list_liver.values())
            to_predict_list_liver = list(to_predict_list_liver)
            if(len(to_predict_list_liver)==7):
                result_liver, probabilities_liver = ValuePredictorLiver(to_predict_list_liver,7)
            if(int(result_liver)==1):
                prediction = f"Oh, Dear, you have {round(probabilities_liver[1],2)}% chance of getting the disease. Please consult the doctor immediately"
            else:
                prediction = f"No need to worry, as your probability of getting the disease is {round(probabilities_liver[1],2)}% which is less than 50%"
    return(render_template("predict_liver.html", prediction_text=prediction))  

port = int(os.getenv("PORT", 5000))
if __name__ == "__main__":
    #app.run(debug=False)
    host = '0.0.0.0'
    #port = 5000
    httpd = simple_server.make_server(host, port, app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
