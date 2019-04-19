### Code for the GUI for Emotion Recognition
### Author: Anurag Das
### Date: April 18th, 2019

import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
from playsound import playsound
import numpy as np
from keras.models import Sequential
from keras.layers import *
import keras 
import librosa

top = Tk()
# topFrame = Frame(top)
top.title = 'Emotion recognition from audio'
top.geometry('950x350')
canvas = Canvas(top, width=40,height=40, bd=0,bg='white')
# canvas.grid(row=1, column=0)

def openAudio():
    ''' Opens the audio file'''
    File = askopenfilename(title='Open an Audio file') 
    e.set(File)

def playAudio():
    ''' Play the audio file '''
    playsound(e.get())

e = StringVar()
submit_button = Button(top, text ='Open an Audio file', command = openAudio)
submit_button.grid(row=1, column=0)

submit_button = Button(top, text ='Play an Audio File', command = playAudio)
submit_button.grid(row=3, column=0)

emotions_used = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def model_A():
    ''' LSTM model definition and architecture
    The model returned here is referred to as model A
    '''
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
    model.add(Dense(64))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

def model_B():
    ''' CNN model definition and architecture
    The model returned here is referred to as model B
    '''
    model = Sequential()
    model.add(Conv1D(8, kernel_size = 3, input_shape=(40, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(16,kernel_size = 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size = 3))
    model.add(Activation('relu'))
    model.add(Conv1D(16, kernel_size = 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

def model_C():
    ''' CNN model definition and architecture
    The model returned here is referred to as model C
    '''
    model = Sequential()
    model.add(Conv1D(8, 5,padding='same', input_shape=(40, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(16, 5,padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(32, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(16, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])
    return model

def model_D():
    ''' CNN model definition and architecture
    The model returned here is referred to as model D
    '''
    model = Sequential()
    model.add(Conv1D(128, 5,padding='same',
                 input_shape=(40,1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
    

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    return model

def Predict_A():
    ''' Prediction using model A'''
    mfcc = extract_mfcc(e.get())
    model = model_A()
    model.load_weights("Model_A.h5")
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)
    print(mfcc.shape)
    cls_wav = model.predict_classes(mfcc)
    textvar = "The object is : %s" %(emotions_used[int(cls_wav)])
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', textvar+'\n')
    t1.update()

def Predict_B():
    ''' Prediction using model B'''
    mfcc = extract_mfcc(e.get())
    model = model_B()
    model.load_weights("Model_B.h5")
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)
    print(mfcc.shape)
    cls_wav = model.predict_classes(mfcc)
    textvar = "The object is : %s" %(emotions_used[int(cls_wav)])
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', textvar+'\n')
    t1.update()

def Predict_C():
    ''' Prediction using model C'''
    mfcc = extract_mfcc(e.get())
    model = model_C()
    model.load_weights("Model_C.h5")
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)
    print(mfcc.shape)
    cls_wav = model.predict_classes(mfcc)
    textvar = "The object is : %s" %(emotions_used[int(cls_wav)])
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', textvar+'\n')
    t1.update()

def Predict_D():
    ''' Prediction using model D'''
    mfcc = extract_mfcc(e.get())
    model = model_D()
    model.load_weights("Model_D.h5")
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)
    print(mfcc.shape)
    cls_wav = model.predict_classes(mfcc)
    textvar = "The object is : %s" %(emotions_used[int(cls_wav)])
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', textvar+'\n')
    t1.update()

def extract_mfcc(wav_file_name):
    ''' Extracts mfcc features and outputs the average of each dimension'''
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
    return mfccs

submit_button = Button(top, text ='Predict using Model A', command = Predict_A)
submit_button.grid(row=1, column=1)

submit_button = Button(top, text ='Predict using Model B', command = Predict_B)
submit_button.grid(row=2, column=1)

submit_button = Button(top, text ='Predict using Model C', command = Predict_C)
submit_button.grid(row=3, column=1)

submit_button = Button(top, text ='Predict using Model D', command = Predict_D)
submit_button.grid(row=4, column=1)

l1=Label(top,text='Press <Open> to open, <Play> to play an audio file, then press <Predict> ')
l1.grid(row=5)

l1=Label(top, text='')
l1.grid(row=6)

l1=Label(top,text='7-8th letters in the RAVDESS data file names represent the labels, eg 03-01-02-01-01-01-01.wav has label 02, where')
l1.grid(row=7)

l1 = Label(top, text='01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised')
l1.grid(row=8)

l1=Label(top, text='')
l1.grid(row=9)

l1=Label(top,text='First character in the SAVEE data files represent the labels, eg n01.wav has label neutral, where')
l1.grid(row=10)

l1=Label(top,text='d=disgust, f=fearful, sa=sadness, su=surprised')
l1.grid(row=11)

t1=Text(top,bd=0, width=20,height=2,font='Fixdsys -14')
t1.grid(row=0, column=1)

top.mainloop()