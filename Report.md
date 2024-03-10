# Ravdess
{'fearful', 'surprised', 'happy', 'disgust', 'angry', 'neutral', 'sad'}
{'neutral': 88, 'happy': 174, 'sad': 183, 'angry': 179, 'fearful': 182, 'disgust': 180, 'surprised': 182}

# Tess
{'fear', 'surprised', 'happy', 'disgust', 'angry', 'neutral', 'sad'}
{'angry': 382, 'disgust': 391, 'fear': 379, 'happy': 383, 'neutral': 378, 'sad': 379, 'surprised': 387}  

# Important Features index and score
[(2, 0.00987215694993619), (17, 0.021663274416528645), (19, 0.01145941274989755), (20, 0.013220141594834292), (21, 0.0163063098191837), (22, 0.009534516538040699), (26, 0.012384309649848197), (27, 0.011081363943689004), (28, 0.009882731031784891), (30, 0.01591902328687038), (31, 0.011690551295034221), (32, 0.010922746334594533), (34, 0.011806004519302235), (35, 0.015946808273729733), (36, 0.016390006198941418), (37, 0.01019725289294567), (57, 0.00981818615446035), (58, 0.016416185056253602), (59, 0.011562207968461496), (61, 0.014397922019446553), (62, 0.021909304681627315), (63, 0.019213067368304686), (88, 0.01122179163352452), (89, 0.010517809892587868)]

# random search
{
  "n_estimators": 177,
  "min_samples_split": 5,
  "min_samples_leaf": 1,
  "max_features": "sqrt",
  "max_depth": 30,
  "criterion": "entropy"
}

# grid search using random search
{
  "criterion": "entropy",
  "max_depth": 30,
  "max_features": "sqrt",
  "min_samples_leaf": 1,
  "min_samples_split": 5,
  "n_estimators": 377
}