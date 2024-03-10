from email import header
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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score


from matplotlib import pyplot as plt

import librosa.display
import soundfile
import IPython.display as ipd

from hyperopt import hp,fmin,tpe,STATUS_OK,Trials

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

def plot_bar_of_each_class_counts(ravdess_wav_files, ravdess_mapping, ravdess_emt_logic, tess_wav_files, tess_mapping, tess_emt_logic):
    ravdess_emotions_count = all_emotions_in_dataset(ravdess_wav_files, ravdess_mapping, ravdess_emt_logic)
    print(ravdess_emotions_count)
    fig = plt.figure(figsize=(10,10))
    plt.title('Ravdess countplot')
    plt.plot(ravdess_emotions_count.keys(), ravdess_emotions_count.values())
    plt.savefig('plots/ravdess_class_counts.png')
    st.pyplot(fig)

    tess_emotions_count = all_emotions_in_dataset(tess_wav_files, tess_mapping, tess_emt_logic)
    print(tess_emotions_count)
    fig = plt.figure(figsize=(10,10))
    plt.title('Tess countplot')
    plt.plot(tess_emotions_count.keys(), tess_emotions_count.values())
    plt.savefig('plots/tess_class_counts.png')
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
      df_features = pd.read_csv('training_data/df_features.csv', header=None)
      df_labels = pd.read_csv('training_data/df_labels.csv',header=None)
    return df_features, df_labels

def extract_kaggle_data_features():
    kaggle_data_features = []
    kaggle_test_files = glob.glob("Kaggle_Testset/*/*.wav")
    kaggle_test_files = sorted(kaggle_test_files, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
    if(not os.path.exists('Kaggle_Testset/kaggle_data_df.csv')):
      for wv_file in kaggle_test_files:
          kaggle_data_features.append(extract_feature(wv_file))
      kaggle_data_df = pd.DataFrame(kaggle_data_features)
      kaggle_data_df.to_csv('Kaggle_Testset/kaggle_data_df.csv', header=False, index = False)
    else:
      kaggle_data_df = pd.read_csv('Kaggle_Testset/kaggle_data_df.csv', header=None)
    
    return kaggle_data_df

def main():
  #download and extract zip file
  X_train, X_test, y_train, y_test = load_data_and_split()

  # Random Forest classifier
  print("---------------------------------")
  st.write("# Random Forest Classifier")
  print("-------------------")

  maxDepth = 8
  estimators = 200
  rf_clf = RandomForestClassifier(n_estimators = estimators, max_depth=maxDepth, random_state = 123)
  rf_clf = rf_clf.fit(X_train, y_train)

  write_train_and_test_reports(X_train, X_test, y_train, y_test, rf_clf)

  # st.write("## Important features")
  # imp_features = np.array([[idx, score] for (idx ,score) in enumerate(rf_clf.feature_importances_) if score > 0.009])
  # st.write(imp_features)

  st.write("# Random Forest Classifier using Random Search cv")
  rf_randomcv = randomized_search_rf(X_train, y_train)
  st.write('## Best params')
  st.write(rf_randomcv.best_params_)
  best_clf = rf_randomcv.best_estimator_
  write_train_and_test_reports(X_train, X_test, y_train, y_test, best_clf)

  kaggle_data_df = extract_kaggle_data_features()
  predict_kaggle_for_classifier(rf_clf, kaggle_data_df, f'rf_max_depth_{maxDepth}_est_{estimators}')
  predict_kaggle_for_classifier(rf_clf, kaggle_data_df, f'rf_random_search')

  # Voting classifiers
  # https://stackoverflow.com/questions/58580273/why-does-logisticregression-take-target-variable-of-type-object-without-any
  # try_different_classifiers_and_save_models(X_train, X_test, y_train, y_test)


def load_data_and_split():
    file_name = download_data()
    extracted_folder = 'Ravdess_Tess'
    if(not os.path.exists(extracted_folder)):
      with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)

    ravdess_wav_files = glob.glob(f"{extracted_folder}/ravdess/*/*.wav")
    ravdess_mapping = {"01" : "neutral", "02" : "calm", "03" : "happy", "04" : "sad",\
               "05" : "angry", "06" : "fear", "07" : "disgust", "08" : "surprised"}
    ravdess_emt_logic = lambda x: x.split("-")[2]
  # calm is not available in training data set so can be removed while training

    tess_wav_files = glob.glob(f"{extracted_folder}/Tess/*/*.wav")
    tess_mapping = {"neutral": "neutral", "calm": "calm", "happy": "happy", "sad": "sad",\
               "angry": "angry", "fear": "fear", "disgust": "disgust", "surprised": "surprised"}
    tess_emt_logic = lambda x: x.split("/")[-1].split(".")[0].split("_")[-1].lower()

  # plot_bar_of_each_class_counts(ravdess_wav_files, ravdess_mapping, ravdess_emt_logic, tess_wav_files, tess_mapping, tess_emt_logic)
  
    df_features, df_labels = extract_features_and_lables(ravdess_wav_files, ravdess_mapping,\
                                  ravdess_emt_logic, tess_wav_files, tess_mapping, tess_emt_logic)
    X = df_features.copy()
    y = df_labels.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 14)
    return X_train,X_test,y_train,y_test

def write_train_and_test_reports(X_train, X_test, y_train, y_test, rf_clf):
    y_pred = rf_clf.predict(X_test)
    st.write("## Testing Report")
    st.write( accuracy_score(y_test, y_pred))
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

    y_pred_train = rf_clf.predict(X_train)
    st.write("## Training Report")
    st.write( accuracy_score(y_train, y_pred_train))
    st.dataframe(pd.DataFrame(classification_report(y_train, y_pred_train, output_dict=True)))

def try_different_classifiers_and_save_models(X_train, X_test, y_train, y_test):
    log_clf = LogisticRegression(random_state=42)
    svm_clf = SVC(probability= True, random_state = 42)
    dt_clf = DecisionTreeClassifier(max_depth = 5, random_state = 42)
    rf_clf = RandomForestClassifier(n_estimators = 150, max_depth = 10, random_state = 42)
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
      pickle.dump(clf, open(f'models/{clf.__class__.__name__}.sav', 'wb'))
      print('----------------')
      st.write('## Training data Accuracy')
      print('----------------')
      y_pred_train = clf.predict(X_train)
      st.write(accuracy_score(y_train, y_pred_train))
      st.dataframe(pd.DataFrame(classification_report(y_train, y_pred_train, output_dict=True)))

def kaggle_predictions_of_all_used_in_voting_clf():
  models_files = glob.glob(f"models/*.sav")
  kaggle_data_df = extract_kaggle_data_features()

  for clf_file in models_files:
    clf = pickle.load(open(clf_file, 'rb'))
    clf_file_name = clf_file.split('\\')[-1].split(".")[0]
    predict_kaggle_for_classifier(clf, kaggle_data_df, clf_file_name)

def predict_kaggle_for_classifier(clf, kaggle_data_df, file_suffix):
    st.write("# Kaggle Predictions")
    kaggle_pred = clf.predict(kaggle_data_df.to_numpy())

    kaggle_pred_df = pd.DataFrame(kaggle_pred, columns = ['Label'])
    kaggle_pred_df['Id'] = list(np.arange(1, 202))
    kaggle_pred_df = kaggle_pred_df[['Id', 'Label']]
    kaggle_pred_df.to_csv(f'submissions/kaggle_submission_{file_suffix}.csv', index=False)
    print(kaggle_pred)

def hyper_opt_best_model(X_train, y_train):
    space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
          'max_depth': hp.quniform('max_depth', 10, 1200, 10),
          'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
          'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
          'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
          'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200,1300,1500])
      }
    def objective(space):
      model = RandomForestClassifier(criterion = space['criterion'], max_depth = space['max_depth'],
                                 max_features = space['max_features'],
                                 min_samples_leaf = space['min_samples_leaf'],
                                 min_samples_split = space['min_samples_split'],
                                 n_estimators = space['n_estimators'], 
                                 )
    
      accuracy = cross_val_score(model, X_train, y_train, cv = 5).mean()

    # We aim to maximize accuracy, therefore we return it as a negative value
      return {'loss': -accuracy, 'status': STATUS_OK }
  
    trials = Trials()
    best = fmin(fn= objective,
              space= space,
              algo= tpe.suggest,
              max_evals = 80,
              trials= trials)
                
    return best

def randomized_search_rf(X_train, y_train):
  # Number of trees in random forest
  n_estimators = [int(x) for x in np.linspace(start = 100, stop = 800, num = 10)]
  # Number of features to consider at every split
  max_features = ['auto', 'sqrt','log2']
  # Maximum number of levels in tree
  max_depth = [int(x) for x in np.linspace(5, 30, 5)]
  # Minimum number of samples required to split a node
  min_samples_split = [2, 5, 10,14]
  # Minimum number of samples required at each leaf node
  min_samples_leaf = [1, 2, 4,6,8]
  # Create the random grid
  random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'criterion':['entropy','gini']}
  print(random_grid)

  rf=RandomForestClassifier()
  rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,
                                random_state=42,n_jobs=5)
  # fit to randomized rf
  rf_randomcv.fit(X_train, y_train.to_numpy())
  rf_randomcv.best_params_
  print(rf_randomcv.best_params_)

# to make new classifier with the best estimator
  return rf_randomcv

def grid_search_from_input_from_random_search(rf_randomcv, X_train, y_train):
  param_grid = {
    'criterion': [rf_randomcv.best_params_['criterion']],
    'max_depth': [rf_randomcv.best_params_['max_depth']],
    'max_features': [rf_randomcv.best_params_['max_features']],
    'min_samples_leaf': [rf_randomcv.best_params_['min_samples_leaf'], 
                         rf_randomcv.best_params_['min_samples_leaf']+2, 
                         rf_randomcv.best_params_['min_samples_leaf'] + 4],
    'min_samples_split': [rf_randomcv.best_params_['min_samples_split'] - 2,
                          rf_randomcv.best_params_['min_samples_split'] - 1,
                          rf_randomcv.best_params_['min_samples_split'], 
                          rf_randomcv.best_params_['min_samples_split'] +1,
                          rf_randomcv.best_params_['min_samples_split'] + 2],
    'n_estimators': [rf_randomcv.best_params_['n_estimators'] + x for x in range(-300, 400, 100)]
  }
  rf=RandomForestClassifier()
  grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=10,n_jobs=-1,verbose=2)
  grid_search.fit(X_train,y_train)
  return grid_search
# main()

from sklearn.ensemble import HistGradientBoostingClassifier

def xgBoost():
  st.write('# Xg booost')
  X_train, X_test, y_train, y_test = load_data_and_split()
  xg_clf = HistGradientBoostingClassifier(random_state = 42)
  xg_clf.fit(X_train, y_train)

  write_train_and_test_reports(X_train, X_test, y_train, y_test, xg_clf)
  kaggle_data_df = extract_kaggle_data_features()
  predict_kaggle_for_classifier(xg_clf, kaggle_data_df, f'xgboost_plain')

xgBoost()
