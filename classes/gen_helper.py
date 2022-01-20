import numpy as np 
import pandas as pd 
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, recall_score, f1_score,confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
from sklearn.pipeline import Pipeline

class EDA:
    def __init__(self, df):
        self.df = df

    def plot_pie(self, target):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(14,6))

        ax1 = self.df[target].value_counts().plot.pie( x="Did not pass 90 days", 
                        autopct = "%1.0f%%",labels=[f"{self.df[target].value_counts().index[0]}",f"{self.df[target].value_counts().index[1]}"], startangle = 60,
                                ax=ax1,wedgeprops={"linewidth":1,"edgecolor":"k"},explode=[.05,.1],shadow =True)
        ax1.set(title = f'Percentage in {target}')

        ax2 = self.df[target].value_counts().plot(kind="barh" ,ax =ax2)
        for i,j in enumerate(self.df[target].value_counts().values):
            ax2.text(0.5,i,j,fontsize=20, color='red')
        ax2.set(title = f'Count in {target}')
        plt.show()

    def corr_map(self, df, fig_width=25, fig_height=20 ):
        corr = df.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(fig_width, fig_width))
            ax = sns.heatmap(corr, mask=mask, vmax=1.0, linewidths=.5, fmt= '.1f', annot=True)

    def voilin_plot(self, X, y, total_features_to_include = 10):
        """
        X = Total Features
        y = hue
        
        """
        data_n_2 = (X - X.mean()) / (X.std())
        data = pd.concat([y,data_n_2.iloc[:,0:total_features_to_include]],axis=1)
        data = pd.melt(data,id_vars=y.name,
                            var_name="features",
                            value_name='value')
        #plt.figure(figsize=(10,10))
        sns.violinplot(x="features", y="value", hue= y.name, data=data,split=True, inner="quart")
        plt.xticks(rotation=90)

    def box_plot(Self, X, fig_len = 16, fig_height = 25):
        plt.figure(figsize=(fig_len,fig_height))
        for i in range((len(X.columns))):
            ax = plt.subplot(int(len(X.columns)/3), 3, i+1 )
            sns.boxplot(X[X.columns[i]], color='blue')
        plt.tight_layout()
        plt.show()

    def plot_histogram(self, X, columns=1, fig_width = 18, fig_height = 20):
        plt.figure(figsize=(fig_width,fig_height))
        for i in range((len(X.columns))):
            ax = plt.subplot(int(len(X.columns)/columns), columns, i+1 )
            sns.distplot(X[X.columns[i]], hist=True, kde=True, 
                    bins=int(180/5), color = 'green', 
                    hist_kws={'edgecolor':'black'},
                    kde_kws={'linewidth': 1})

        plt.tight_layout()
        plt.show()

    def joint_plot(self, X, var1 , var2):
        return sns.jointplot(X.loc[:, var1], X.loc[:,var2], kind="reg", color="#ce1414")

class FeatureEngineering:
    def __init__(self):
        pass
    def calculate_vif(self, df_features):
        vif = pd.DataFrame()
        vif['Features'] = df_features.columns
        vif['VIF_Values'] = [variance_inflation_factor(df_features.values, i) for i in range(df_features.shape[1])]
        return vif.sort_values(by= 'VIF_Values', ascending=False)
    def select_features(self, X, y, top_n_features=3):
        bestfeatures = SelectKBest(score_func=chi2, k = top_n_features)
        fit = bestfeatures.fit(X,y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        #concat two dataframes for better visualization 
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['Columns','Score']  #naming the dataframe columns
        return (featureScores.nlargest(top_n_features,'Score').reset_index(drop=True))  #print n best features

class ModelTraining:
    def __init__(self):
        pass
    def compare_base_classifiers(self, models, X_train, y_train, X_val , y_val,
     sort_by = 'acc_val',scaler = StandardScaler(), imputer= KNNImputer(n_neighbors=3, missing_values= np.nan),
     cols_to_impute = slice(1,100,1), poly = PolynomialFeatures(degree=2, include_bias = True) ):
        '''
        This method takes the dictionary of pre-defined models along 
        with the respective dataframes to evaluate the model
        sort_by options are [acc_val,f1_val,recall_val]
        '''
        metrics_dict = {'acc_train' : [],
                        'acc_val' : [],
                        'f1_train' : [],
                        'f1_val' : [],
                        'recall_train':[],
                        'recall_val':[]}
        ct = ColumnTransformer(
        remainder='passthrough',
        transformers=[
        ('imputer' , imputer, cols_to_impute)
        ])
        for i in models:
            pipe = Pipeline([
            ('col_transformer', ct),
            ('scaler', scaler),
            ('poly', poly),
            ('classifier', models[i])
            ])
            pipe.fit(X_train, y_train)
            y_predicted_train =  pipe.predict(X_train)
            y_predicted = pipe.predict(X_val)
            
            metrics_dict['acc_train'].append(round(accuracy_score(y_train, y_predicted_train), 4))
            metrics_dict['acc_val'].append(round(accuracy_score(y_val,y_predicted),4))

            metrics_dict['f1_train'].append(round(f1_score(y_train, y_predicted_train),4))
            metrics_dict['f1_val'].append(round(f1_score(y_val,y_predicted),4))
            
            metrics_dict['recall_train'].append(round(recall_score(y_train, y_predicted_train),4))
            metrics_dict['recall_val'].append(round(recall_score(y_val,y_predicted),4))

        return (pd.DataFrame(metrics_dict, index= models.keys()).sort_values(by=sort_by, ascending=False))
    def plot_cm(self,y_val, y_predicted):
        '''
        This method takes true y values and predicted y values to draw a Confusion Matrix

        '''
        conf_matrix = confusion_matrix(y_val,y_predicted)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()



class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: Hasnain
                Version: 1.0
                Revisions: None

                """

    def __init__(self):
        self.rfc = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')
        self.nb  = GaussianNB()
    def get_best_params_for_random_forest(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Hasnain
                                Version: 1.0
                                Revisions: None

                        """
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.rfc, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)

            return self.clf

        except Exception as e:
        	raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: Hasnain
                                        Version: 1.0
                                        Revisions: None

                                """
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            return self.xgb
        except Exception as e:
            raise Exception()
    def get_best_params_for_gnb(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_gnb
                                Description: get the parameters for Gaussian Naive Bayes Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Hasnain
                                Version: 1.0
                                Revisions: None

                        """
        try:
            # initializing with different combination of parameters
            self.param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.nb, param_grid=self.param_grid, cv=5,  verbose=3)
            self.pipe_nb = Pipeline([
            		('scaler', StandardScaler()),
            		('grid', grid)
            	])
            #finding the best parameters
            self.pipe_nb.fit(train_x, train_y)

            #extracting the best parameters
            self.var_smoothing = self.pipe.named_steps['grid'].best_params_['var_smoothing']

            #creating a new model with the best parameters
            self.clf = GaussianNB(var_smoothing = self.var_smoothing)
            # training the mew model
            self.clf.fit(train_x, train_y)

            return self.clf

        except Exception as e:
        	raise Exception()



    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: Hasnain
                                                Version: 1.0
                                                Revisions: None

                                        """
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost

            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            self.prediction_random_forest=self.random_forest.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y,self.prediction_random_forest)
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest) # AUC for Random Forest

            # create best model for Naive Bayes
            self.nb=self.get_best_params_for_gnb(train_x,train_y)
            self.prediction_nb=self.nb.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.nb_score = accuracy_score(test_y,self.prediction_nb)
            else:
                self.nb_score = roc_auc_score(test_y, self.prediction_nb)

            #comparing the two models
            if(self.random_forest_score <  self.xgboost_score) & (self.nb_score <  self.xgboost_score):
                return 'XGBoost',self.xgboost
            elif (self.random_forest_score <  self.nb_score) & (self.xgboost_score <  self.nb_score):
            	 return 'GaussianNB',self.nb
            else:
                return 'RandomForest',self.random_forest

        except Exception as e:
            raise Exception()
    def hyper_pipeline_individual(scaler = StandardScaler(), classifier = RandomForestClassifier(),
                                 imputer = KNNImputer( n_neighbors=3, missing_values=np.nan) , cols_to_impute = None):
        ct = ColumnTransformer(
        remainder='passthrough',
        transformers=[
        ('imputer' , imputer, cols_to_impute)
        ])
        pipe = Pipeline([
        ('col_transformer', ct),
        ('scaler', scaler),
        ('classifier', classifier)
        ])
        return pipe 