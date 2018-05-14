import os

import numpy as np
from keras import Input, Model
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.layers import Conv1D, MaxPool1D, Dropout, Dense, Flatten
from keras.utils import np_utils
from sklearn.externals import joblib
from sklearn.metrics import f1_score, precision_score, recall_score

batch_size = 50
num_epochs = 30
kernel_size1 = 50
kernel_size2 = 20
pool_size1 = 10
pool_size2 = 10
conv_depth1 = 32
conv_depth2 = 64
drop_prob1 = 0.25
drop_prob2 = 0.4
hidden_size = 512


class Metrics(Callback):
    """Callback to print validation f1, accuracy.  Adapted from
    https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2"""
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        avg = 'weighted'
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average=avg)
        _val_recall = recall_score(val_targ, val_predict, average=avg)
        _val_precision = precision_score(val_targ, val_predict, average=avg)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("— val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return

# Load, prep data
Xtr, Xte, Ytr, Yte = joblib.load('tfidf.pkl')
num_train, length = Xtr.shape
num_test = Xte.shape[0]
Yte = np_utils.to_categorical(Yte, 3)
Ytr = np_utils.to_categorical(Ytr, 3)

Xtr = np.expand_dims(Xtr.toarray(), axis=2)
Xte = np.expand_dims(Xte.toarray(), axis=2)


inp = Input(shape=(length, 1))
conv1 = Conv1D(filters=conv_depth1, kernel_size=kernel_size1,
               strides=kernel_size1, padding='same', activation='relu')(inp)
pool1 = MaxPool1D(pool_size=pool_size1, strides=pool_size1, padding='same')(conv1)
drop1 = Dropout(rate=drop_prob1)(pool1)
conv2 = Conv1D(filters=conv_depth2, kernel_size=kernel_size2,
               strides=kernel_size2, padding='same', activation='relu')(drop1)
pool2 = MaxPool1D(pool_size=pool_size2, strides=pool_size2, padding='same')(conv2)
drop2 = Dropout(rate=drop_prob1)(pool2)

flat = Flatten()(drop2)
hidden = Dense(units=hidden_size, activation='relu')(flat)
drop3 = Dropout(rate=drop_prob2)(hidden)
out = Dense(units=3, activation='softmax')(drop3)

metrics = Metrics()

model = Model(inputs=inp, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(Xtr, Ytr, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1,
          callbacks=[TensorBoard(log_dir=os.path.expanduser('~/TensorBoard/')), metrics])
scores = model.evaluate(Xte, Yte, verbose=1)
print('Test Loss after Epoch 12 :', scores[0])
print('Test Accuracy after Epoch 12: ', scores[1])
model.save(os.path.expanduser('~/TensorBoard/kerasmodel.h5py'), 'w+')

