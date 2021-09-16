import sys
import pickle
import flask
from flask import Flask, request, jsonify
# from flask_restful import Resource, Api
# from pure_sklearn.map import convert_estimator
import numpy as np
import time
import pandas as pd
# from sklearn.utils import column_or_1d
# from sklearn.decomposition import PCA

app = flask.Flask(__name__)
app.config["DEBUG"] = True


# api=Api(app)


def convertRawInput(rawInput):
    cardNumberString=rawInput[0]
    transactionType=rawInput[1]
    amount=rawInput[2]
    currencyName=rawInput[3]
    responseCode=rawInput[4]
    countryName=rawInput[5]
    vendorCodeString=rawInput[6]
    year=rawInput[7]
    print(f"year: {year}")
    month=rawInput[8]
    print(f"month: {month}")
    day=rawInput[9]
    print(f"day: {day}")
    hour=rawInput[10]
    print(f"hour: {hour}")
    min=rawInput[11]
    print(f"min: {min}")
    sec=rawInput[12]
    print(f"sec: {sec}")
    millisec=rawInput[13]
    print(f"millisec: {millisec}")
    cardNumber=int(cardNumberString)
    vendorCode=int(vendorCodeString)
    ts = pd.Timestamp(year,month,day,hour,min,sec,millisec)
    julianDate = ts.to_julian_date()
    print(f"currencyName: {currencyName}")
    encodedCurrency=currencyEncoder.transform([currencyName])
    print(f"encodedCurrency: {encodedCurrency}")
    encodedCountry=countryEncoder.transform([countryName])
    requestParams=[[cardNumber,transactionType,julianDate,amount,encodedCurrency[0],responseCode,encodedCountry[0],vendorCode]]
    print(f"requestParams: {requestParams}")
    return requestParams


@app.route('/api/v1/resources/predict_and_proba', methods=['POST'])
def api_predict_and_proba_sample():
    start_time = time.process_time()
    content = request.get_json()
    # content = json.loads(content_json)
    # books.append(content)
    rawInput = content.get("values")
    requestParams=convertRawInput(rawInput)



    # values_for_prediction=np.reshape(values,(1,-1))
    print(rawInput)
    print(estimatorContainer)
    values_for_prediction = [rawInput]
    values_for_prediction_array = np.array(values_for_prediction)
    currencyEncoder = estimatorContainer["currencyEncoder"]
    countryEncoder = estimatorContainer["countryEncoder"]
    fittedPipeline = estimatorContainer["pipeline"]
    # X_for_pure_predict = X_New.tolist()
    # predicted_value = pipelineNew.predict(X_New)
    # pure_predict_value = pure_predict.predict(X_for_pure_predict)
    # probability = pure_predict.predict_proba(X_for_pure_predict)[0]
    # prediction = pure_predict_value[0]
    # reshapedRequestParams=requestParams.reshape(-1,1)

    print(f"requestParams: {requestParams}")
    prediction=fittedPipeline.predict(requestParams)[0]
    probability=fittedPipeline.predict_proba(requestParams)[0]
    response = {
        'prediction': int(prediction),
        # 'prediction': 1,
        'negativeProbability': float(probability[0]),
        # 'negativeProbability': 0.5,
        'positiveProbability': float(probability[1])
        # 'positiveProbability': 0.5
    }
    end_time = time.process_time()
    elapsed_time = (end_time - start_time)
    print(elapsed_time)
    return jsonify(response)


if __name__ == '__main__':
    print("START")
    # estimatorName=sys.argv[1]
    estimatorName="estimator_1"
    # portNumber=sys.argv[2]
    portNumber=8084
    # estimatorContainerFile = open('C:\\Users\\machine\\Documents\\MKI\\Estimators\\Python\\estimator_1.pickle', 'rb')
    estimatorContainerFile = open(f'C:\\Users\\machine\\Documents\\MKI\\Estimators\\Python\\{estimatorName}.pickle', 'rb')
    estimatorContainer = pickle.load(estimatorContainerFile)
    currencyEncoder=estimatorContainer.get("currencyEncoder")
    countryEncoder=estimatorContainer.get("countryEncoder")
    fittedPipeline=estimatorContainer.get("pipeline")
    estimatorContainerFile.close()
    # pure_predict = convert_estimator(pipelineNew)
    app.run(port=portNumber)
