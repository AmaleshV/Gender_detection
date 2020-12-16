from flask import Flask,request, url_for, redirect, render_template
import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import threading
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#run_with_ngrok(app)
# logging.basicConfig(filename="flask.log", level=logging.DEBUG,
#                     format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

#health = HealthCheck(app, "/hcheck")
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-90M")
 
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-90M")
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
# with open('tokenizer_Bert_adamw_0.81.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
def name_country(text):
    inputs = tokenizer(text, return_tensors='pt',return_token_type_ids=False)
    reply_ids = model.generate(**inputs)
    outcome= [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in reply_ids]
    return outcome 
    
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])



def predict():
    if request.method == 'POST':
        message = str(request.form['message'])
        print(message)
        #data = [message]
        my_prediction = name_country(message)
        print(my_prediction)
    return render_template('result.html',prediction = my_prediction)
if __name__ == '__main__':
	app.run(debug=True)