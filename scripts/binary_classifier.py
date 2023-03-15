from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from generate_datasets import generate_datasets

# load dataset
circle_filename = "../data/circle_data.npz"
rectangle_filename = "../data/rectangle_data.npz"

training_dataset, testing_dataset = generate_datasets(circle_filename,
                                                      rectangle_filename)


# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_shape=(60, ), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# evaluate model with standardized dataset
estimator = KerasClassifier(model=create_baseline,
                            epochs=100,
                            batch_size=5,
                            verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)


def get_features(features, labels):
    return features


def get_labels(features, labels):
    return labels


features = training_dataset.map(get_features)
labels = training_dataset.map(get_labels)

results = cross_val_score(estimator, features, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" %
      (results.mean() * 100, results.std() * 100))
