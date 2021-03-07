
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__) # initializing a flask app
CORS(app)

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            area=request.form['area']
            sqft = float(request.form['sqft'])
            bath = float(request.form['bath'])
            bhk = float(request.form['bhk'])

            filename = 'BHP_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file

            def predict_price_pickle(location, sqft, bath, bhk):
                '''
                Function which helps to actually predict the prices.
                '''
                X = pd.read_csv("X_dataset.csv", index_col=0)
                loc_index = np.where(X.columns == location)[0][
                    0]  # gives column num    # np.where() function returns the indices of elements in an input array where the given condition is satisfied.

                x = np.zeros(
                    len(X.columns))  # np.zeros() function returns a new array of given shape and type, with zeros.
                x[0] = sqft
                x[1] = bath
                x[2] = bhk
                if loc_index >= 0:
                    x[loc_index] = 1

                return loaded_model.predict([x])[0]


            prediction=predict_price_pickle(area,sqft,bath,bhk).round(3)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=prediction)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app