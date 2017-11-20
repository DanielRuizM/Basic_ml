from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.decomposition import PCA


 
data_tia = pd.read_csv('/Users/danielruizmayo/datos_cartel_confidencial.csv', delimiter=";")
clase=data_tia['target']
data=data_tia.loc[:, data_tia.columns != 'target']
X_train, X_test, y_train, y_test = train_test_split(data, clase, test_size=0.2)

print("RANDOM FOREST")
rf = RandomForestRegressor()

rf.fit(X_train, y_train)
predicted = rf.predict(X_test)

accuracy_rf_pre_pca=rf.score(X_test, y_test)
mae_rf_pre_pca=mean_absolute_error(predicted,y_test)
mse_rf_pre_pca=mean_squared_error(predicted,y_test)
r2_rf_pre_pca=r2_score(predicted,y_test)



#accuracy = accuracy_score(y_test, predicted)
#print('Out-of-bag score estimate: {}'.format(rf.oob_score))
print('Mean accuracy score pre_pca: {}'.format(accuracy_rf_pre_pca))
print('MAE pre_pca: {}'.format(mae_rf_pre_pca))
print('MSE pre_pca: {}'.format(mse_rf_pre_pca))
print('R2 pre_pca: {}'.format(r2_rf_pre_pca))

print("PASAMOS AL PCA")

pca = PCA(n_components=170)

#print(pca.explained_variance_ )
X_train, X_test, y_train, y_test = train_test_split(data, clase, test_size=0.2)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)

#print(X_train_pca)
rf.fit(X_train_pca, y_train)
predicted = rf.predict(X_test_pca)

accuracy_rf_post_pca=rf.score(X_test_pca, y_test)
mae_rf_post_pca=mean_absolute_error(predicted,y_test)
mse_rf_post_pca=mean_squared_error(predicted,y_test)
r2_rf_post_pca=r2_score(predicted,y_test)


print('Mean accuracy score _post_pca: {}'.format(accuracy_rf_post_pca))
print('MAE _post_pca: {}'.format(mae_rf_post_pca))
print('MSE _post_pca: {}'.format(mse_rf_post_pca))
print('R2 _post_pca: {}'.format(r2_rf_post_pca))


print("Quito las variables con mas de 0.9 de correlacion entre ellas sin contar la clase")

corr_value = data_tia.corr().values

list_to_delete=[]
for x in range(0,len(corr_value[0])):
    for y in range(0,len(corr_value[0])):
        if x<y and corr_value[x][y] > 0.90:
            list_to_delete.append(y)
myset=set(list_to_delete)
#print(myset)
list_to_delete = list(myset)

data_wo_vars = data_tia.drop(data_tia.columns[list_to_delete], 1)
#si quiero ver las variables con las que me quedo
#print(data_wo_vars.dtypes)    

X_train, X_test, y_train, y_test = train_test_split(data_wo_vars, clase, test_size=0.2)
rf.fit(X_train, y_train)
predicted = rf.predict(X_test)

accuracy_rf_post_delete=rf.score(X_test, y_test)
mae_rf_post_delete=mean_absolute_error(predicted,y_test)
mse_rf_post_delete=mean_squared_error(predicted,y_test)
r2_rf_post_delete=r2_score(predicted,y_test)


print('Mean accuracy score _post_delete: {}'.format(accuracy_rf_post_delete))
print('MAE _post_delete: {}'.format(mae_rf_post_delete))
print('MSE _post_delete: {}'.format(mse_rf_post_delete))
print('R2 _post_delete: {}'.format(r2_rf_post_delete))


print("Ademas quito las que tienen menos de 0.01 de correlacion con target")

corr_value = data.corr().values

list_to_delete=[]
for x in range(1,len(corr_value[0])):
    for y in range(1,len(corr_value[0])):
        if x<y and (abs(corr_value[x][y]) > 0.90 or abs(corr_value[0][y])<0.01 ):
            list_to_delete.append(y)
myset=set(list_to_delete)
#print(myset)
list_to_delete = list(myset)

data_wo_vars = data_tia.drop(data_tia.columns[list_to_delete], 1)
#si quiero ver las variables con las que me quedo
#print(data_wo_vars.dtypes)    

X_train, X_test, y_train, y_test = train_test_split(data_wo_vars, clase, test_size=0.2)
rf.fit(X_train, y_train)
predicted = rf.predict(X_test)

accuracy_rf_post_delete_target=rf.score(X_test, y_test)
mae_rf_post_delete_target=mean_absolute_error(predicted,y_test)
mse_rf_post_delete_target=mean_squared_error(predicted,y_test)
r2_rf_post_delete_target=r2_score(predicted,y_test)


print('Mean accuracy score _post_delete_target: {}'.format(accuracy_rf_post_delete_target))
print('MAE _post_delete_target_target: {}'.format(mae_rf_post_delete_target))
print('MSE _post_delete_target: {}'.format(mse_rf_post_delete_target))
print('R2 _post_delete_target: {}'.format(r2_rf_post_delete_target))
