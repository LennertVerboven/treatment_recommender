import argparse
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser(description='Train the machine learning model on data')
parser.add_argument('training_data', type=str, help='path to the training dataset')
parser.add_argument('out_file', type=str, help='path where the resulting model will be stored')
args = parser.parse_args()

categories = ['cost_category', 'toxicity_category', 'bactericidal_activity_category', 'bactericidal_activity_early_category', 'sterilizing_activity_category', 'resistance_prevention_category', 'synergism_category', 'antagonism_category', 'contraindications_category']
normalised = ['cost_normalised_inverted', 'toxicity_normalised_inverted', 'bactericidal_activity_normalised', 'bactericidal_activity_early_normalised', 'sterilizing_activity_normalised', 'resistance_prevention_normalised', 'synergism_normalised', 'antagonism_normalised_inverted', 'contraindications_normalised_inverted']
binary = ['qt_prolongation', 'high_bactericidal_activity_early', 'high_bactericidal_activity', 'high_sterilizing_activity', 'efficacy', 'mechanism_of_action', 'route_of_administration', 'route_of_administration_hospitalized']

features = categories + normalised + binary
truth = 'accept'

full_training_set = pd.read_csv(args.training_data, index_col=[0,1])
feedback_round = 5

training_data = full_training_set[full_training_set.apply(lambda regimen: True if regimen.accept != -1 and regimen.feedback_round <= feedback_round else False, axis=1)]
training_data = full_training_set[full_training_set.accept != -1]

X_train, y_train = training_data[features], training_data[truth]
model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=0)
#model = GradientBoostingClassifier(n_estimators=100, max_depth=10, random_state=0)
model.fit(X_train, y_train)
pickle.dump(model, open(args.out_file, 'wb'))
