import numpy as np
import pandas as pd
#import preprocessor as p
import counselor
import tensorflow as tf
load_model = tf.keras.models.load_model
import joblib
from pathlib import Path
import nltk

nltk.download('wordnet')
