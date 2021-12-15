from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from random import randint
from keras.models import load_model
import numpy as np
# initialize our Flask application

model = load_model('models/model.h5')
app= Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods=["POST", "GET"])
@cross_origin()
def evaluateData():
    # Receive Data
    eegdata = request.args.get('eegdata')
    eegdata = eegdata.replace("[", "")
    eegdata = eegdata.replace("]", "")
    eegdata = list(eegdata.split(","))
    eegdata = list(map(float, eegdata))


    # Process Data
    X_test = np.array([eegdata])
    # Make Prediction
    result = model.predict((X_test[:,::4]-X_test.mean())/X_test.std())
    yp=np.zeros((result.shape[0]))
    for i in range(result.shape[0]):
        yp[i]=np.argmax(result[i])+1

    #conversion of classes
    for i in range(result.shape[0]):
        if yp[i]!=1:
            yp[i]=0
    return jsonify(yp.tolist())


#  main thread of execution to start the server
if __name__=='__main__':
    app.run(debug=True, port=3254)