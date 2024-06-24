from flask import Flask , render_template,request
import pickle
import  numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

model = pickle.load(open("Logestic_model.pkl",'rb'))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))


app = Flask(__name__)

@app.route('/')
def main():
    return render_template("home.html")

@app.route("/predict",methods = ['POST'])
def  predict():
    try:
        
        data = request.form['content']
        input_data = np.array([[data]])
        print("Raw input data :",input_data)
        input_data = vectorizer.transform([data])
        print("Vectorized input data:", input_data)
        
        prediction = model.predict(input_data)
        
        print("The Predicted value of the model is :",prediction[0])
        
        return render_template("result.html",data = prediction[0])
    
    except Exception as e:
        print(f"Error occured  : {e}")
        return render_template("home.html",error_message = "An error occured during the prediction. Please check your input values")
    
    
if __name__ == "__main__":
    app.run(debug=True)
        