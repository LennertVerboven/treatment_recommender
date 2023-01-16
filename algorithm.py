import logging
import pickle
import os
import argparse

import pandas as pd

from drugdatabase import DrugDatabase

DB = os.path.join('./pickles/')
MODEL = os.path.join(DB, './model.pkl')

categories = ['cost_category', 'toxicity_category', 'bactericidal_activity_category', 'bactericidal_activity_early_category', 'sterilizing_activity_category', 'resistance_prevention_category', 'synergism_category', 'antagonism_category', 'contraindications_category']
normalised = ['cost_normalised_inverted', 'toxicity_normalised_inverted', 'bactericidal_activity_normalised', 'bactericidal_activity_early_normalised', 'sterilizing_activity_normalised', 'resistance_prevention_normalised', 'synergism_normalised', 'antagonism_normalised_inverted', 'contraindications_normalised_inverted']
binary = ['qt_prolongation', 'high_bactericidal_activity_early', 'high_bactericidal_activity', 'high_sterilizing_activity', 'efficacy', 'mechanism_of_action', 'route_of_administration', 'route_of_administration_hospitalized']

features = categories + normalised + binary
truth = 'accept'



def compute(profile):
    production_db = DrugDatabase(drug_db=os.path.join(DB, 'drugdatabase_v9.csv'), ddi_db=os.path.join(DB, 'drug-drug_interactions_v2.xls'))
    production_db.load_rules_matrix(os.path.join(DB, 'all_regimens.csv'))

    production_model = pickle.load(open(MODEL, 'rb'))

    regimens = production_db.get_top_regimens(profile.reset_index(drop=True), -1)
    regimens['regimen_id'] = regimens.regimen.apply(lambda regimen: str(sorted(regimen)))
    regimens['score'] = pd.DataFrame(production_model.predict_proba(regimens[features]), columns=['p_reject', 'p_accept'])['p_accept'].values
    regimens = regimens.sort_values('score', ascending=False)

    return regimens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the machine learning model on data')
    parser.add_argument('resistance_profile', type=str, help='path to the csv file containing the resistance profile')
    parser.add_argument('out_file', type=str, help='path where the resulting recommendations will be placed')
    args = parser.parse_args()

    profile = pd.read_csv(args.resistance_profile, index_col=0).transpose().iloc[0]
    print(profile)
    recommendations = compute(profile)
    recommendations.to_csv(args.out_file)
    pass
