from tkinter.tix import X_REGION
import wget
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import os, glob, pickle
import re
import zipfile

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report

from matplotlib import pyplot as plt

import librosa.display
import soundfile
import IPython.display as ipd

import warnings
warnings.filterwarnings("ignore")

def download_data():
  url = 'https://cdn.iisc.talentsprint.com/CDS/MiniProjects/Ravdess_Tess.zip'
  file_name = url.split("/")[-1]
  if(not os.path.exists(file_name)):
    wget.download(url, file_name)
  return file_name

def all_emotions_in_dataset(wav_files, mapping, emt_logic):
  emotions_ = []
  emotions_count = {}
  for file_name in wav_files:
      emt = mapping[emt_logic(file_name)]
      emotions_.append(emt)
      if(emt in emotions_count.keys()):
         emotions_count[emt] += 1
      else:
         emotions_count[emt] = 1
  print(set(emotions_))
  return emotions_count

def display_sample_audio_waveform(ravdess_wav_files):
    sample_audio_path = ravdess_wav_files[4]
  # librosa is used for analyzing and extracting features of an audio signal
    data, sampling_rate = librosa.load(sample_audio_path)
    fig = plt.figure(figsize=(15, 5))
    # plt.title('Sample Audio Waveform')
  # librosa.display.waveshow is used to plot waveform of amplitude vs time
    _ = librosa.display.waveshow(data, sr=sampling_rate)
    plt.savefig('plots/sample_audio_waveform.png')
    st.pyplot(fig)

def extract_feature(file_name):
    # load audio
    X, sample_rate = librosa.load(file_name)
    # apply stft()
    stft=np.abs(librosa.stft(X))
    result = np.array([])
    # compute mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    # combine the features
    result = np.hstack((result, mfccs))
    # compute chroma features and combine
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    result = np.hstack((result, chroma))
    # compute melspectrogram and combine
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
    result = np.hstack((result, mel))
    return result

def extract_features_and_lables(ravdess_wav_files, ravdess_mapping, ravdess_emt_logic, tess_wav_files, tess_mapping, tess_emt_logic):
    features, labels = [], []

    if(not os.path.exists('training_data/df_features.csv')):
      for wv_file in ravdess_wav_files:
        features.append(extract_feature(wv_file))
      # extracting label
        emt = ravdess_mapping[ravdess_emt_logic(wv_file)]
        labels.append(emt)
  
      for wv_file in tess_wav_files:
        features.append(extract_feature(wv_file))
      # extracting label
        emt = tess_mapping[tess_emt_logic(wv_file)]
        labels.append(emt)

      df_features = pd.DataFrame(features)
      df_labels = pd.DataFrame(labels)
      df_features.to_csv('training_data/df_features.csv', header=False, index=False)
      df_labels.to_csv('training_data/df_labels.csv', header=False, index=False)
    else:
      df_features = pd.read_csv('training_data/df_features.csv', header=0)
      df_labels = pd.read_csv('training_data/df_labels.csv',header=0)
    return df_features, df_labels

def main():
  #download and extract zip file
  file_name = download_data()
  extracted_folder = 'Ravdess_Tess'
  if(not os.path.exists(extracted_folder)):
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
      zip_ref.extractall(extracted_folder)

  # read ravdess audio files
  ravdess_wav_files = glob.glob(f"{extracted_folder}/ravdess/*/*.wav")
  ravdess_mapping = {"01" : "neutral", "02" : "calm", "03" : "happy", "04" : "sad",\
               "05" : "angry", "06" : "fear", "07" : "disgust", "08" : "surprised"}
  ravdess_emt_logic = lambda x: x.split("-")[2]
  # calm is not available in training data set so can be removed while training
  ravdess_emotions_count = all_emotions_in_dataset(ravdess_wav_files, ravdess_mapping, ravdess_emt_logic)
  print(ravdess_emotions_count)
  fig = plt.figure(figsize=(10,10))
  plt.title('Ravdess countplot')
  plt.plot(ravdess_emotions_count.keys(), ravdess_emotions_count.values())
  plt.savefig('plots/ravdess_class_counts.png')
  st.pyplot(fig)

  display_sample_audio_waveform(ravdess_wav_files)
  # ipd.Audio( ravdess_wav_files[4])

  tess_wav_files = glob.glob(f"{extracted_folder}/Tess/*/*.wav")
  tess_mapping = {"neutral": "neutral", "calm": "calm", "happy": "happy", "sad": "sad",\
               "angry": "angry", "fear": "fear", "disgust": "disgust", "surprised": "surprised"}
  tess_emt_logic = lambda x: x.split("/")[-1].split(".")[0].split("_")[-1].lower()
  tess_emotions_count = all_emotions_in_dataset(tess_wav_files, tess_mapping, tess_emt_logic)
  print(tess_emotions_count)
  fig = plt.figure(figsize=(10,10))
  plt.title('Tess countplot')
  plt.plot(tess_emotions_count.keys(), tess_emotions_count.values())
  plt.savefig('plots/tess_class_counts.png')
  st.pyplot(fig)
  
  df_features, df_labels = extract_features_and_lables(ravdess_wav_files, ravdess_mapping,\
                                  ravdess_emt_logic, tess_wav_files, tess_mapping, tess_emt_logic)
  
  X = df_features.copy()
  y = df_labels.copy()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 14)


  # Random Forest classifier
  print("---------------------------------")
  st.write("# Random Forest Classifier")
  print("-------------------")
  rf_clf = RandomForestClassifier(n_estimators = 500, random_state = 123)
  rf_clf = rf_clf.fit(X_train, y_train)
  y_pred = rf_clf.predict(X_test)
  print(rf_clf.__class__.__name__, accuracy_score(y_test, y_pred))
  y_pred_train = rf_clf.predict(X_train)
  print(rf_clf.__class__.__name__, accuracy_score(y_train, y_pred_train))
  st.write("## Important features")
  imp_features = np.array([[idx, score] for (idx ,score) in enumerate(rf_clf.feature_importances_) if score > 0.009])
  st.write(imp_features)

  # st.write("# Random Forest Classifier with important features only")
  # X_train_imp_features = X_train[:, imp_features[:, 1]]

  # Voting classifiers
  # https://stackoverflow.com/questions/58580273/why-does-logisticregression-take-target-variable-of-type-object-without-any
  log_clf = LogisticRegression(random_state=42)
  svm_clf = SVC(probability= True, random_state = 42)
  dt_clf = DecisionTreeClassifier(max_depth = 10, random_state = 42)
  rf_clf = RandomForestClassifier(n_estimators = 500, random_state = 42)
  classifiers = [('lr', log_clf), ('svc', svm_clf), ('dec_tree', dt_clf), ('rf', rf_clf)]
  voting_clf = VotingClassifier(estimators = classifiers, voting = 'soft')


  for clf in (log_clf, svm_clf, dt_clf, rf_clf, voting_clf):
    print('------------------------------')
    st.write("# " + clf.__class__.__name__)
    print('----------------')
    clf.fit(X_train, y_train)
    st.write('## Test data Accuracy')
    print('----------------')
    y_pred = clf.predict(X_test)
    st.write(accuracy_score(y_test, y_pred))
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

    print('----------------')
    st.write('## Training data Accuracy')
    print('----------------')
    y_pred_train = clf.predict(X_train)
    st.write(accuracy_score(y_train, y_pred_train))
    st.dataframe(pd.DataFrame(classification_report(y_train, y_pred_train, output_dict=True)))
    
main()