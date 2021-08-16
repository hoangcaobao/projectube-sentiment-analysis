from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf


class my_model(tf.keras.models.Model):
    def __init__(self, phobert, num_classes):
        super(my_model,self).__init__()
        self.phobert=phobert
        self.classify=Dense(num_classes, activation="softmax")
        self.fully_connected_layer1=Dense(512, activation="relu")
        self.dropout1=Dropout(0.5)
        self.fully_connected_layer2=Dense(512, activation="relu")
        self.dropout2=Dropout(0.5)

    def call(self, data):
        x=self.phobert(data)
        x=x["pooler_output"]  
        x=self.fully_connected_layer1(x)
        x=self.dropout1(x)
        x=self.fully_connected_layer2(x)
        x=self.dropout2(x)
        x=self.classify(x)
        return x
    
    