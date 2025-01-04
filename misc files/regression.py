import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor




print(X.shape, Y.shape)
x = X.T ; y = Y.reshape(-1, 1)
print(x.shape, y.shape)


df = pd.DataFrame(x, columns = ['Distance', 'Speed', 'Mask area', 'Acceleration_x', 'Acceleration_y', 'Relative_Position'])
df['Time duration'] = Y.tolist()


corr_matrix = df.corr()
print(corr_matrix)

plt.matshow(corr_matrix)
plt.show()

df_train = df.sample(frac = 0.6, random_state=1)
df_test = df.drop(index=df_train.index)

Y_train = df_train[['Time duration']]
X_train = df_train.drop(['Time duration'], axis=1)

Y_test = df_test[['Time duration']]
X_test = df_test.drop(['Time duration'], axis=1)



#normalized_df=(df-df.mean())/df.std()

X_train=(X_train-X_train.mean())/X_train.std()
X_test=(X_test-X_test.mean())/X_test.std()


Y_train=Y_train-Y_train.mean()
Y_test = Y_test - Y_test.mean()

linear_rg = LinearRegression().fit(X_train, Y_train)
training_error = 1 - linear_rg.score(X_train, Y_train)
#print(training_error)
y_pred = linear_rg.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

# Print the results
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R-Squared: {r2:.2f}")



pca = PCA()
pca.fit(X_train)
X_pca = pca.transform(X_train)
scores = cross_val_score(LinearRegression(), X_pca, Y_train, cv=5)
print("PCR R-squared scores: ", scores.mean())

# Partial Least Squares Regression (PLSR)
pls = PLSRegression(n_components=3)
pls.fit(X_train, Y_train)
scores = cross_val_score(pls, X_train, Y_train, cv=5)
print("PLSR R-squared scores: ", scores.mean())


rf_regr = RandomForestRegressor(max_depth=5, random_state=0)
rf_y = (Y_train.to_numpy()).T[0]
#print(rf_y[0])
rf_regr.fit(X_train, rf_y)
print(rf_regr.score(X_train, rf_y))
rf_y_test = (Y_test.to_numpy()).T[0]

from sklearn.tree import DecisionTreeRegressor

dt_regr = DecisionTreeRegressor(random_state=0)
dt_y = (Y_train.to_numpy()).T[0]
#print(rf_y[0])
dt_regr.fit(X_train, dt_y)
print(dt_regr.score(X_train, dt_y))
dt_y_test = (Y_test.to_numpy()).T[0]
# Calculate MSE, RMSE, MAE, and R-Squared
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

# Print the results
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R-Squared: {r2:.2f}")