from joblib import load
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression



df_new=pd.DataFrame({'tv_power':[45689.02313546,89765.56967864],'radio_root':[89657.0098456,99978.442699]})



model_file='file_linear.pickle'

final_model=pickle.load(open(model_file,'rb'))
print(final_model.predict(df_new))



