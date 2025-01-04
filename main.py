import os
import numpy as np
from numpy import asarray, save, load
import matplotlib.pyplot as plt

from ObjectTracking import track_objects
from feature_extraction import feature_extractor
from files_selection import frame_selection
from instance_segmentation import instance_segmentation

from regression import preprocessing, linear_regression, RandomForestRegression
from regression import DecisionTreeRegression, GradientBoostingRegression, BaggingRegression
from regression import PolynomialRegression

working_dir = os.getcwd()
dataset_dir = os.path.join(working_dir, "Dataset")

"""
print('Testing 20 images')
selected_frames = frame_selection(dataset_dir, 1, 20)

finres = instance_segmentation(selected_frames)

object_deletion_dict, active_objects = track_objects(finres)

X, Y = feature_extractor(finres, object_deletion_dict, active_objects,dataset_dir)
data_X=asarray(X)
data_Y = asarray(Y)
save('data_X_20.npy', data_X)
save('data_Y_20.npy', data_Y)
print(X.shape, Y.shape)

print('Testing 50 images')

selected_frames = frame_selection(dataset_dir, 1, 50)
finres = instance_segmentation(selected_frames)
object_deletion_dict, active_objects = track_objects(finres)
X, Y = feature_extractor(finres, object_deletion_dict, active_objects,dataset_dir)

data_X=asarray(X)
data_Y = asarray(Y)
save('data_X_50.npy', data_X)
save('data_Y_50.npy', data_Y)
print(X.shape, Y.shape)


print('Testing 100 images')

selected_frames = frame_selection(dataset_dir, 1, 100)

finres = instance_segmentation(selected_frames)

object_deletion_dict, active_objects = track_objects(finres)

X, Y = feature_extractor(finres, object_deletion_dict, active_objects,dataset_dir)
data_X=asarray(X)
data_Y = asarray(Y)
save('data_X_100.npy', data_X)
save('data_Y_100.npy', data_Y)
print(X.shape, Y.shape)


print('Testing 200 images')
selected_frames = frame_selection(dataset_dir, 1, 200)

finres = instance_segmentation(selected_frames)

object_deletion_dict, active_objects = track_objects(finres)

X, Y = feature_extractor(finres, object_deletion_dict, active_objects,dataset_dir)
data_X=asarray(X)
data_Y = asarray(Y)
save('data_X_200.npy', data_X)
save('data_Y_200.npy', data_Y)
print(X.shape, Y.shape)

print('Testing 500 images')

selected_frames = frame_selection(dataset_dir, 1, 500)

finres = instance_segmentation(selected_frames)

object_deletion_dict, active_objects = track_objects(finres)

X, Y = feature_extractor(finres, object_deletion_dict, active_objects,dataset_dir)
data_X=asarray(X)
data_Y = asarray(Y)
save('data_X_500.npy', data_X)
save('data_Y_500.npy', data_Y)
print(X.shape, Y.shape)

print('Testing 800 images')
selected_frames = frame_selection(dataset_dir, 1, 800)

finres = instance_segmentation(selected_frames)

object_deletion_dict, active_objects = track_objects(finres)

X, Y = feature_extractor(finres, object_deletion_dict, active_objects,dataset_dir)
data_X=asarray(X)
data_Y = asarray(Y)
save('data_X_800.npy', data_X)
save('data_Y_800.npy', data_Y)

print(X.shape, Y.shape)
"""
X = load('data_X_500.npy')
Y = load('data_Y_500.npy')
df, X_train, Y_train, X_test, Y_test = preprocessing(X, Y)

corr_matrix = df.corr()
print(corr_matrix)
fig, ax = plt.subplots()

heatmap = ax.matshow(corr_matrix, cmap='coolwarm')
cbar = plt.colorbar(heatmap)
columns = ['Distance', 'Speed', 'Mask area', 'Acceleration_x', 'Acceleration_y',
           'Relative_Position', 'Time duration']
ax.set_xticks(np.arange(len(columns)))
ax.set_yticks(np.arange(len(columns)))
ax.set_xticklabels([columns[i] for i in range(len(columns))], rotation=45)
ax.set_yticklabels([columns[i] for i in range(len(columns))])

# Set title and show plot


##Linear Regression results
print("Linear regression")
linear_regression(X_train, Y_train, X_test, Y_test)
print()


##Polynomial regression results
print("Polynomial regression")
for degree in [2, 3, 4, 5, 6]:
    print('Degree', degree)
    PolynomialRegression(X_train, Y_train, X_test, Y_test, degree)
    print()
print('###################################################################')

##Random Forest Regression results
print("Random forest regression")
for max_depth in [1, 2, 3, 4, 5]:
    n_estimators = 100
    print("Max depth", max_depth)
    RandomForestRegression(X_train, Y_train, X_test, Y_test, max_depth, n_estimators)
    print()

for n_estimators in [10, 20, 50, 100, 200, 500, 1000]:
    max_depth = 2
    print("No. of estimators", n_estimators)
    RandomForestRegression(X_train, Y_train, X_test, Y_test, max_depth, n_estimators)
    print()

print('###################################################################')

##Decision Tree Regression results
print("Decision Tree regression")
for max_depth in [1, 2, 3, 4, 5]:
    print("Max depth", max_depth)
    DecisionTreeRegression(X_train, Y_train, X_test, Y_test, max_depth)
    print()
print('###################################################################')

##Gradient Boosting Regression results

print("Gradient boosting regression")
for alpha in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]:
    print('Learning rate', alpha)
    estimators = 100
    GradientBoostingRegression(X_train, Y_train, X_test, Y_test, alpha, estimators)
    print()

for estimators in [10, 20, 50, 100, 200, 500, 1000, 2000]:
    alpha = 0.1
    print('No. of estimators', estimators)
    GradientBoostingRegression(X_train, Y_train, X_test, Y_test, alpha, estimators)
    print()

print('###################################################################')

##Bagging Regression results
print("Bagging regression")
for estimators in [5, 10, 20, 50]:
    print('No. of estimators', estimators)
    BaggingRegression(X_train, Y_train, X_test, Y_test, estimators)
    print()

ax.set_title('Correlation Matrix')
plt.show()