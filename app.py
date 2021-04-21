#### import packages
from flask import Flask,request, url_for, redirect, render_template
import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import TFBertModel,  BertConfig, BertTokenizerFast,BatchEncoding
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
import transformers
import pickle
from transformers import ElectraTokenizer, TFElectraModel,ElectraConfig

#from flask_ngrok import run_with_ngrok


#import google.cloud.logging







#### logging
#client =  google.cloud.logging.Client()
#client.setup_logging()



app = Flask(__name__)
#run_with_ngrok(app)
# logging.basicConfig(filename="flask.log", level=logging.DEBUG,
#                     format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')





#### Load tokenizer

model_name = 'google/electra-small-discriminator'
# Max length of tokens
max_length = 6
# Load transformers config and set output_hidden_states to False
configuration = ElectraConfig.from_pretrained(model_name)
configuration.output_hidden_states = False
# Load BERT tokenizer
tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', config = configuration)

#### Load electra model 
model = tf.keras.models.load_model('electra_small_model')



def gender_detection(text):
    a = tokenizer.encode(
        text= text,
        add_special_tokens=True,
        max_length=6,
        truncation=True,
        padding=True, 
        return_tensors='tf',
        return_token_type_ids = False,
        return_attention_mask = False,
        verbose = True)
    a_as_vector = tf.reshape(a, [-1])
    zero_padding = tf.zeros([1 * 6] - tf.shape(a_as_vector), dtype=a.dtype)
    a_reshape=tf.concat([a_as_vector, zero_padding],0)
    a_reshape=tf.reshape(a_reshape,[1,6])
    d=model.predict(a_reshape)
    confidence_female= d['GENDER'][0][1]-d['GENDER'][0][0]
    confidence_female= (confidence_female+6.5)/(13)
    
    c=np.argmax(d['GENDER'])
    gender_predicted= np.where(c==1, 'Female','Male')
    if gender_predicted=='Male':
        confidence_female = 1- confidence_female
    gender_reveal= str('{} is {}. The probability they are {} is {:.0%}'.format(text,gender_predicted,gender_predicted,confidence_female))
    prob_female= str('{} is ')
    return gender_reveal



    
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
	if request.method == 'POST':
		message = str(request.form['message'])
		print(message)
        #data = [message]
		my_prediction = gender_detection(message)
		print(my_prediction)
	return render_template('result.html',prediction = my_prediction)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = gender_detection(str(data['text']))

    return jsonify(prediction)
if __name__ == '__main__':
    app.run(debug=True)
