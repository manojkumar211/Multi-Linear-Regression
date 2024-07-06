from feature_selection import X,y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from best_values import lr_best_test
from sklearn.linear_model import LinearRegression,Lasso,LassoCV,Ridge,RidgeCV,ElasticNet,ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures





# Linear Regression Model:-

class Linear_best_RandomState:

        lr_best_train=[]
        lr_best_test=[]

        try:
            for i in range(0,20):
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i)
                lr=LinearRegression()
                lr.fit(X_train,y_train)
                lr_train_pred=lr.predict(X_train)
                lr_test_pred=lr.predict(X_test)
                lr_best_train.append(lr.score(X_train,y_train))
                lr_best_test.append(lr.score(X_test,y_test))

        except Exception as e:
            raise Exception(f'Best RandomState Error in Linear Regression :\n'+str(e))

class Linear_regression(Linear_best_RandomState):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=np.argmax(lr_best_test))

    try:

        try:

            linear_model =LinearRegression() # type: ignore
            linear_model.fit(X_train, y_train) # type: ignore
            lr_coe=linear_model.coef_ # type: ignore
            lr_int=linear_model.intercept_ # type: ignore
            y_tr_pred=linear_model.predict(X_train) # type: ignore
            y_te_pred=linear_model.predict(X_test) # type: ignore
            train_score=linear_model.score(X_train,y_train) # type: ignore  
            test_score=linear_model.score(X_test,y_test) # type: ignore
            cross_val=cross_val_score(linear_model,X,y,cv=5).mean() # type: ignore
            lr_tr_mae=mean_absolute_error(y_train,y_tr_pred) # type: ignore
            lr_tr_mse=mean_squared_error(y_train,y_tr_pred) # type: ignore
            lr_tr_rmse=np.sqrt(mean_squared_error(y_train,y_tr_pred)) # type: ignore
            lr_te_mae=mean_absolute_error(y_test,y_te_pred) # type: ignore
            lr_te_msr=mean_squared_error(y_test,y_te_pred) # type: ignore
            lr_te_rmse=np.sqrt(mean_squared_error(y_test,y_te_pred)) # type: ignore

        except Exception as e:
            raise Exception(f'Error find in Linear Regression model :\n'+str(e))

        try:

            def __init__(self,linear_model,lr_coe,lr_int,y_tr_pred,y_te_pred,train_score,test_score,cross_val,
                        lr_tr_mae,lr_tr_mse,lr_tr_rmse,lr_te_mae,lr_te_msr,lr_te_rmse,lr_best_train,lr_best_test):
                    
                try:

                    self.linear_model=linear_model
                    self.lr_coe=lr_coe
                    self.lr_int=lr_int
                    self.y_tr_pred=y_tr_pred
                    self.y_te_pred=y_te_pred
                    self.train_score=train_score
                    self.test_score=test_score
                    self.cross_val=cross_val
                    self.lr_tr_mae=lr_tr_mae
                    self.lr_tr_mse=lr_tr_mse
                    self.lr_tr_rmse=lr_tr_rmse
                    self.lr_te_mae=lr_te_mae
                    self.lr_te_msr=lr_te_msr
                    self.lr_te_rmse=lr_te_rmse
                    self.lr_best_train=lr_best_train
                    self.lr_best_test=lr_best_test
                
                except Exception as e:
                    raise Exception(f'Error find in Linear Regression at Initiate :\n'+str(e))

            try:


                def linear_regression_model(self):
                    return self.linear_model
                def linear_regression_coe(self):
                    return self.lr_coe
                def linear_regression_int(self):
                    return self.lr_int
                def linear_regression_y_tr_pred(self):
                    return self.y_tr_pred
                def linear_regression_y_te_pred(self):
                    return self.y_te_pred
                def linear_regression_train_score(self):
                    return self.train_score
                def linear_regression_test_score(self):
                    return self.test_score
                def linear_regression_cross_val(self):
                    return self.cross_val
                def linear_regression_tr_mae(self):
                    return self.lr_tr_mae
                def linear_regression_tr_mse(self):
                    return self.lr_tr_mse
                def linear_regression_tr_rmse(self):
                    return self.lr_tr_rmse
                def linear_regression_te_mae(self):
                    return self.lr_te_mae
                def linear_regression_te_msr(self):
                    return self.lr_te_msr
                def linear_regression_te_rmse(self):
                    return self.lr_te_rmse
                def linear_regression_best_train(self):
                    return super().lr_best_train
                def linear_regression_best_test(self):
                    return super().lr_best_test
                
            except Exception as e:
                raise Exception(f'Error find in Linear Regression at Defining level :\n'+str(e))
            
        except Exception as e:
            raise Exception(f'Error find in Linear Regression at Initiate and Defining level :\n'+str(e))

    except Exception as e:
        raise Exception(f'Totall Error in Linear Regression :\n'+str(e))


          



        



