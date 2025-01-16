import cv2
import numpy as np
import time
from predict import YoloModel

def test_model(model_name,model_type, device = None, img = '12567_png.rf.6bb2ea773419cd7ef9c75502af6fe808.jpg'):
    print('starting testing with PT model')
    predictor = YoloModel(model_name, model_type, device)

    img = cv2.imread(img)

    # cv2.imshow("Original image", img)
    # cv2.waitKey(0)
    total = 0
    for predict in range(100):
        start_time = time.time()
        bots = predictor.predict(img, show=False)
        end_time = time.time()
        elapsed = end_time - start_time
        #print(f'elapsed time: {elapsed:.4f}')
        total += elapsed
        # predictor.show_predictions(img, bots)
        print('-'*10)
    return total / 100

if __name__ == "__main__":
    models = ['ONNX', 'PT']
    with open('test_results.txt', 'w') as file:
        for model in models:
            avg = test_model('100epoch11',model)
            file.write(f'{model} average = {avg}')
            file.write('\n')

        for model in models:
            avg = test_model('100epoch11',model, device = "mps")
            file.write(f'{model} average, mps = {avg}')
            file.write('\n')