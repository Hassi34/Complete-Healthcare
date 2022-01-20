import numpy as np 
import pickle , gzip
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder

def ValuePredictorCancer(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        cols = ['area_se', 'texture_mean', 'concavity_mean', 'symmetry_worst','smoothness_worst', 'symmetry_mean', 'fractal_dimension_worst']
        to_predict = pd.DataFrame(to_predict, columns=cols)
        with gzip.open(r'./models/breast_cancer_model.pickle.gz', 'rb') as f :
            loaded_model = pickle.load(f)
        result = loaded_model.predict(to_predict)
        probabilities_cancer = (loaded_model.predict_proba(to_predict)[0])*100
    return result[0], probabilities_cancer

def ValuePredictorDiabetes(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==8):
        cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
        to_predict = pd.DataFrame(to_predict, columns=cols)
        with gzip.open('./models/diabetes_model.pickle.gz', 'rb') as f:
            pipeline_diabetes = pickle.load(f)
        result_diabetes = pipeline_diabetes.predict(to_predict)
        probabilities_diabetes = (pipeline_diabetes.predict_proba(to_predict)[0])*100
    return result_diabetes[0], probabilities_diabetes

def ValuePredictorKidney(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==6):
        cols = ['sg','al','sc','hemo','pcv','htn']
        to_predict = pd.DataFrame(to_predict, columns=cols)
        with gzip.open('./models/kidney_model.pickle.gz', 'rb') as f:
            pipeline_kidney = pickle.load(f)
        result_kidney = pipeline_kidney.predict(to_predict)
        probabilities_kidney = (pipeline_kidney.predict_proba(to_predict)[0])*100
    return result_kidney[0], probabilities_kidney
cat_cols = ['htn']
class CustomLabelEncode(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X ,y=None):
        le=LabelEncoder()
        for i in cat_cols:
            X[i]=le.fit_transform(X[i])
        return X

def ValuePredictorHeart(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==11):
        cols = ['thalach', 'oldpeak', 'ca', 'cp', 'exang', 'chol', 'age', 'trestbps','slope', 'sex', 'thal']
        to_predict = pd.DataFrame(to_predict, columns=cols)
        with gzip.open('./models/heart_model.pickle.gz', 'rb') as f:
            pipeline_heart = pickle.load(f)
        result_heart = pipeline_heart.predict(to_predict)
        probabilities = (pipeline_heart.predict_proba(to_predict)[0])*100
    return result_heart[0], probabilities

def ValuePredictorLiver(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        cols = ['Alamine_Aminotransferase', 'Alkaline_Phosphotase', 'Total_Bilirubin', 'Age', 'Albumin_and_Globulin_Ratio', 'Gender', 'Total_Protiens']
        to_predict = pd.DataFrame(to_predict, columns=cols)
        with gzip.open('./models/liver_model.pickle.gz', 'rb') as f:
            pipeline_liver = pickle.load(f)
        result_liver = pipeline_liver.predict(to_predict)
        probabilities = (pipeline_liver.predict_proba(to_predict)[0])*100
    return result_liver[0], probabilities