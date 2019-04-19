# Audio_emotion_recognition
Emotion recognition using speech data, complete with a GUI demo

# Contributors
Anurag Das

Dependencies
============
- numpy
- playsound
- tensorflow
- keras
- matplotlib
- scipy
- seaborn
- librosa
- tkinter

Usage
============
To install the necessary packages, run the following command:-<br/>
`pip install -r requirements.txt`

Run,
`python gui.py`

Trained models
==============
- Model A - LSTM
- Model B - 5 layer CNN
- Model C - 5 layer CNN without dropout
- Model D - 3 layer CNN

Among the trained models Model A and Model B perform the best achieving almost 98% accuarcy on the test set

**Accuarcy per epoch**
![Link to acc](https://github.com/dasanurag/Audio_emotion_recognition/tree/master/media/acc_epoch.png)

**Accuracy and loss plots**
![Acc](https://github.com/dasanurag/Audio_emotion_recognition/tree/master/media/modelB1_accuracy.png)
![Loss](https://github.com/dasanurag/Audio_emotion_recognition/tree/master/media/modelB1_loss.png)

Dataset
=======
- RAVDESS - [`Audio visual dataset`](https://zenodo.org/record/1188976#.XLlgGENOnq8)
Filenames follow a 7 part numerical identifier. (e.g., 02-01-06-01-02-01-12.mp4). These identifiers define the stimulus characteristics:-
1. Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
2. Vocal channel (01 = speech, 02 = song).
3. Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
4. Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
5. Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
6. Repetition (01 = 1st repetition, 02 = 2nd repetition).
7. Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Some samples from the original dataset are provided in the Audio_Speech_Actors_01-05.zip file for testing. Some other samples are present in the RAVDESS_files folder. Also included are samples from the [`SAVEE`](http://kahlan.eps.surrey.ac.uk/savee/Introduction.html) dataset for testing. 

Demo link
===========
[`https://www.youtube.com/watch?v=ItnfknVA1dA`](https://www.youtube.com/watch?v=ItnfknVA1dA)

Kaggle kernel link
==================
[`https://www.kaggle.com/dasanurag38/audio-emotion-recognition?scriptVersionId=13099888`](https://www.kaggle.com/dasanurag38/audio-emotion-recognition?scriptVersionId=13099888)
