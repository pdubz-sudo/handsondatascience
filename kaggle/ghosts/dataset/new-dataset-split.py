import pandas as pd

"""
this code creates new train and test dataset from Kaggles dataset so the participants can 
get an idea of if their ML code is generalized without having to submit it on Kaggle.
However, the newly made test dataset is pretty small (20%)
"""

df = pd.read_csv('train.csv')
shuffled = df.sample(frac = 1)  # shuffle the whole dataset
shuffled.iloc[0:295,:].to_csv("train-new.csv", index=False)  # create new train csv file
shuffled.iloc[295::, :].drop(shuffled.iloc[:,1:6], axis =1).to_csv('test-labels-new.csv', index=False)  # create new test answer csv file
shuffled.iloc[295::, :].drop(['type'], axis=1).to_csv('test-new.csv', index=False)  # create new test set