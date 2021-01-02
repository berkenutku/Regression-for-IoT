import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

tf.autograph.set_verbosity(0, alsologtostdout=False)

def Call_CNN_Latency(x):
    new_model = tf.keras.models.load_model('saved_model_1/CNN_Latency')
    y = new_model.predict([x])
    return y


def Call_CNN_Power(x):
    new_model = tf.keras.models.load_model('saved_model_1/CNN_Power')
    y = new_model.predict([x])
    return y

# Example call for Norm layer
# y = dnn_model.predict([[1000]])
# The value is the 'Total Input Features'
def Call_Norm_Power(x):
    new_model = tf.keras.models.load_model('saved_model_1/Norm_Power')
    y = new_model.predict([x])
    return y

def Call_Norm_Latency(x):
    new_model = tf.keras.models.load_model('saved_model_1/Norm_Latency')
    y = new_model.predict([x])
    return y

# Example call for Relu layer
# y = dnn_model.predict([[1000]])
# The value is the 'Total Input Features'
def Call_Relu_Latency(x):
    new_model = tf.keras.models.load_model('saved_model_1/Relu_Latency')
    y = new_model.predict([x])
    return y

def Call_Relu_Power(x):
    new_model = tf.keras.models.load_model('saved_model_1/Relu_Power')
    y = new_model.predict([x])
    return y

# Example call for Pool layer
# y = dnn_model.predict([[75264,1000]])
# The first value is the 'Total Input Features'
# The second one is the 'Total Output Features'
def Call_Pool_Latency(x):
    new_model = tf.keras.models.load_model('saved_model_1/Pool_Latency')
    y = new_model.predict([x])
    return y

def Call_Pool_Power(x):
    new_model = tf.keras.models.load_model('saved_model_1/Pool_Power')
    y = new_model.predict([x])
    return y

# Example call for Fc layer
# y = dnn_model.predict([[75264,1000]])
# The first value is the 'Total Input Features'
# The second one is the 'Total Output Features'
def Call_Fc_Latency(x):
    new_model = tf.keras.models.load_model('saved_model_1/Fc_Latency')
    y = new_model.predict([x])
    return y

def Call_Fc_Power(x):
    new_model = tf.keras.models.load_model('saved_model_1/Fc_Power')
    y = new_model.predict([x])
    return y