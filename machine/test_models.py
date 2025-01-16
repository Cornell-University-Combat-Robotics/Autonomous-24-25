import cv2
import numpy as np
import time
from predict import YoloModel 
from predict import RoboflowModel

def test_model(method,model_name,model_type, device = None, img = '12567_png.rf.6bb2ea773419cd7ef9c75502af6fe808.jpg'):
    if method == 'YoloModel':
        print('starting testing with PT model')
        predictor = YoloModel(model_name, model_type, device)
    elif method == 'RoboflowModel':
        print('starting testing with Roboflow model')
        predictor = RoboflowModel()

    img = cv2.imread(img)

    # cv2.imshow("Original image", img)
    # cv2.waitKey(0)
    data = np.zeros(100)
    for i in range(100):
        start_time = time.time()
        predictor.predict(img, show=False)
        end_time = time.time()
        elapsed = end_time - start_time
        #print(f'elapsed time: {elapsed:.4f}')
        data[i] = elapsed
        # predictor.show_predictions(img, bots)
    print([data])
    return data

if __name__ == "__main__":
    method = input("Method: y or r")
    if method == 'YoloModel' or method == 'y':
        models = ['ONNX', 'PT']
        with open('test_results.txt', 'w') as file:
            for model in models:
                data = test_model('YoloModel','100epoch11',model)
                avg = np.mean(data)
                avgn1 = np.mean(data[1:])
                med = np.median(data)
                file.write(f'{model} = {data}')
                file.write('\n')
                file.write(f'Average: {avg}')
                file.write('\n')
                file.write(f'Average no first: {avgn1}')
                file.write('\n')
                file.write(f'Median: {med}')
                file.write('\n')

            for model in ['PT']:
                data = test_model('100epoch11',model, device = "mps")
                avg = np.mean(data)
                avgn1 = np.mean(data[1:])
                med = np.median(data)
                file.write(f'{model}, mps = {data}')
                file.write('\n')
                file.write(f'Average: {avg}')
                file.write('\n')
                file.write(f'Average no first: {avgn1}')
                file.write('\n')
                file.write(f'Median: {med}')
                file.write('\n')
    elif method == 'RoboflowModel' or 'r':
        with open('test_results.txt', 'w') as file:
            data = test_model(method = 'RoboflowModel', model_name = None, model_type = None)
            avg = np.mean(data)
            avgn1 = np.mean(data[1:])
            med = np.median(data)
            file.write(f'Roboflow = {data}')
            file.write('\n')
            file.write(f'Average: {avg}')
            file.write('\n')
            file.write(f'Average no first: {avgn1}')
            file.write('\n')
            file.write(f'Median: {med}')
            file.write('\n')

    else:
        print("you suck and messed up")