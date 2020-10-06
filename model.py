import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd 
from cv2 import imread
import numpy as np
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras import losses
from sklearn.metrics import roc_auc_score
import tensorflow as tf 


def image_generator(files, batch_size=5): 
    while True:     
        batch_paths = np.random.choice(a=files.index, size=batch_size) 
                
        batch_input = [] 
        batch_output = [] 
        
        for index in batch_paths: 
            f = files.copy()
            f = f.loc[f.index==index]     
            for path, clf in zip(f['Path'], f['Pleural Effusion']): 
                img = imread(path)
                img = preprocess_input(img)
                
                batch_input.append(img)
                batch_output.append(int(clf))
                    
        batch_x = np.array(batch_input)
        
        batch_y = np.array(batch_output)
        
        yield( batch_x, batch_y )

#Since most images in the validation set belong to the AP view, 
#we are only considering AP view images for training
#Moreover, we are not considering uncertain labels
df = pd.read_pickle('pleural_all_AP.pickle')

dense_model = DenseNet121(include_top=True, weights='imagenet', \
                          input_shape=(224,224,3))

output = Dense(1, activation='sigmoid', name='clf')\
(dense_model.layers[-2].output)

model = Model(inputs=dense_model.input, outputs=output)

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

model.compile(optimizer='adadelta', loss=losses.binary_crossentropy, \
              metrics=['accuracy', auroc])


model.fit_generator(image_generator(files=df, batch_size=32), \
                    steps_per_epoch=len(df.index)/32, epochs=10, \
                    verbose=1)

model.save('pleural_dense_AP.hdf5')

#For the fine-tuned model (with saliency maps) the same structure was used 
    
    
    
    
    




