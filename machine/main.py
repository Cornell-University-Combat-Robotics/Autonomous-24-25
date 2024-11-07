from data import *
from model import *

train_new_model = False
load_new_data = False

if load_new_data:
    load_data(dataset="Signs")
else:
    select_data(dataset="Fall")

show_sample_data("train")

X, y = fetch_data("test", size=1)

if train_new_model:
    model = make_model()

    show_predict(X, y, model, threshold=0.1,
                 img_title="Model Prediction Before Training w/ threshold=0.1")

    X_train, y_train = fetch_data("train", size=100)
    X_val, y_val = fetch_data("valid", size=10)
    model.fit(X_train, y_train, batch_size=32, epochs=10,
              shuffle=True, validation_data=(X_val, y_val))

    model.save('saved_models/test')

else:
    model = tf.keras.models.load_model(
        'saved_models/test', custom_objects={'loss_func': loss_func})

X_test, y_test = fetch_data("test", size=10)
show_predict(X, y, model, threshold=0.5,
             img_title="Model Prediction After Training w/ threshold=0.5")

model.evaluate(X_test, y_test, batch_size=8)
