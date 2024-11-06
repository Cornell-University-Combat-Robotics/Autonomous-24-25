from data import *
from model import *

load_data(dataset="Signs")
X_train, y_train = fetch_data("train", size=100)
X_val, y_val = fetch_data("valid", size=383)
X_test, y_test = fetch_data("test", size=150)
show_sample_data("train")

model = make_model()

X, y = fetch_data("test", size=1)
show_predict(X, y, model, threshold=0.1,
             img_title="Model Prediction Before Training w/ threshold=0.1")

model.fit(X_train, y_train, batch_size=32, epochs=10,
          shuffle=True, validation_data=(X_val, y_val))

show_predict(X, y, model, threshold=0.5,
             img_title="Model Prediction After Training w/ threshold=0.5")

model.evaluate(X_test, y_test, batch_size=8)
