import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential

train_data = pd.read_csv("train.csv", sep=",")

# Drop all columns which are not important
train_data = train_data.drop(["Course Number","Launch Date", "Course Title", "Instructors","Course Subject"], axis=1)
train_data.fillna(0, inplace=True)
X = np.array(train_data.ix[:, 1:])
y = np.ravel(train_data.Institution)
print(X[0,:])

#model for a binary classification
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(17,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
			   optimizer='adam',
			   metrics=['accuracy'])
model.fit(X, y, epochs=2, batch_size=1, verbose=1)
model.save("course.model")

# Make predictions on test dataset
from keras.models import load_model, Sequential
from keras.layers import Dense

test_data = pd.read_csv("test.csv", sep=",")
test_data = test_data.drop(["Institution","Course Number","Launch Date", "Course Title", "Instructors","Course Subject"], axis=1)

test_data.fillna(0, inplace=True)

X_test = np.array(test_data)
model = load_model("course.model")
predictions = np.round(model.predict(X_test))
predictions = np.array(predictions, dtype=np.int64)
print(predictions.shape)
