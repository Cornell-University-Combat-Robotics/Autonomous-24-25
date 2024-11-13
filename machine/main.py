from data import *
from model import *

train_new_model = False
load_new_data = False

if load_new_data:
    load_data(dataset="NHRL")
else:
    select_data(dataset="NHRL")

show_sample_data("train")

X, y = fetch_data("test", size=1)

if train_new_model:
    model = make_model()

    show_predict(X, y, model, threshold=0.1,
                 img_title="Model Prediction Before Training w/ threshold=0.1")

    X_train, y_train = fetch_data("train", size=408)
    X_val, y_val = fetch_data("valid", size=117)
    model.fit(X_train, y_train, batch_size=32, epochs=10,
              shuffle=True, validation_data=(X_val, y_val))

    model.save('saved_models/NHRL_400_train_size')

else:
    model = tf.keras.models.load_model(
        'saved_models/NHRL_400_train_size', custom_objects={'loss_func': loss_func})

X_test, y_test = fetch_data("test", size=58)
show_predict(X, y, model, threshold=0.05,
             img_title="Model Prediction After Training w/ threshold=0.05")

model.evaluate(X_test, y_test, batch_size=8)
