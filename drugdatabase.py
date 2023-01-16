import re
import ast

import pandas as pd

score_categories = ['cost_category', 'toxicity_category', 'bactericidal_activity_category', 'bactericidal_activity_early_category', 'sterilizing_activity_category', 'resistance_prevention_category', 'synergism_category', 'antagonism_category', 'contraindications_category']
score_normalised = ['cost_normalised_inverted', 'toxicity_normalised_inverted', 'bactericidal_activity_normalised', 'bactericidal_activity_early_normalised', 'sterilizing_activity_normalised', 'resistance_prevention_normalised', 'synergism_normalised', 'antagonism_normalised_inverted', 'contraindications_normalised_inverted']


class DrugDatabase:
    ''' The class representing the database of all of the drugs '''

    def __init__(self):
        pass

    def __init__(self, drug_db, ddi_db):
        self.drugs = pd.read_csv(drug_db, index_col=0)
        caseinsensitive_high_dose = re.compile(re.escape(' high dose'), re.IGNORECASE)
        self.high_dose = [self.get_drug_id_by_name(caseinsensitive_high_dose.sub('', drug)) for drug in self.drugs.name if ' high dose' in drug.lower()]

        ''' Read the "antagonism" and "synergism" from file '''
        drug_interaction_matrix = pd.read_excel(ddi_db, index_col=0).fillna('')#.replace('contraindication', 3).replace('Antagonism', 2).replace('Synergism', 1)
        self.drug_interaction_matrix = (drug_interaction_matrix + drug_interaction_matrix.transpose())

    def get_drug_id_by_abbrev(self, abbrev):
        drug_id = self.drugs[self.drugs.abbreviation.apply(lambda cell: True if cell.lower() == abbrev.lower() else False)].index
        if len(drug_id) != 1:
            warnings.warn("Drug abbreviation '{}' occurs {} times".format(abbrev, len(drug_id)))
            return None
        else:
            return drug_id[0]

    def get_drug_id_by_name(self, drug_name):
        drug_name = drug_name.replace('_', ' ').lower()
        if drug_name == 'para-aminosalisylic acid':
            drug_name = 'para-aminosalicylic acid'
        drug_id = self.drugs[self.drugs.name.apply(lambda cell: True if cell.lower() == drug_name.lower() else False)].index
        if len(drug_id) != 1:
            warnings.warn("Drug name '{}' occurs {} times".format(drug_name, len(drug_id)))
            return -1
        else:
            return drug_id[0]

    def get_drug_names(self, upper=False, lower=False):
        drug_names = [i for i in self.drugs.name]
        if upper == True:
            drug_names = [drug.upper() for drug in drug_names]
        if lower == True:
            drug_names = [drug.lower() for drug in drug_names]
        return drug_names

    def __satisfies_regimen_contraints(self, regimen):
        for drug in regimen:
            constraint = self.drugs.loc[drug].regimen_constraints
            if pd.isnull(constraint):
                continue
            constraint = ast.literal_eval(constraint)
            for condition in constraint:
                if not condition in regimen:
                    return False
        return True

    def __van_deun_companion_drugs(self, regimen):
        # At least 1 companion (2 in total when also counting the core drug) drugs with a moderate (3) bactericidal_activity
        if len([i for i in [self.drugs.loc[drug].bactericidal_activity for drug in regimen] if i >= 3]) < 2:
            return False
        # At least 1 companion drug with a high (4) bactericidal_activity_early
        if len([i for i in [self.drugs.loc[drug].bactericidal_activity_early for drug in regimen] if i >= 4]) < 1:
            return False
        # At least 2 companion (3 in total when also counting the core drug) drugs with a moderate (3) sterilizing_activity
        if len([i for i in [self.drugs.loc[drug].sterilizing_activity for drug in regimen] if i >= 3]) < 3:
            return False
        return True

    def __van_deun_core_drug(self, regimen):
        for drug in regimen:
            if self.drugs.loc[drug].bactericidal_activity >= 4 and self.drugs.loc[drug].sterilizing_activity >= 4:
                return True
        return False

    def __compute_cost_of_regimen(self, regimen):
        if set(regimen) == set([22, 20, 16, 12]):
            return 8.71
        else:
            return sum([self.drugs.loc[drug].cost for drug in regimen])

    def __get_category(self, value, categories):
        for idx, category in enumerate(categories):
            if value > category:
                if idx == 0:
                    return 0
                return idx / (len(categories))
        return (idx + 1) / (len(categories))

    def __get_synergism_antagonism_contraindications_samedrug(self, regimen):
        cntr = Counter()
        for idx1, drug1 in enumerate(regimen):
            for drug2 in regimen[idx1 + 1:]:
                cntr[self.drug_interaction_matrix.loc[self.drugs.loc[drug1]['name'], self.drugs.loc[drug2]['name']]] += 1
        #return pd.Series([cntr[1], cntr[2], cntr[3], cntr[4]])
        return pd.Series([cntr['synergism'], cntr['antagonism'], cntr['contraindication'], cntr['same_drug']])

    def __categorize_parameter(self, regimen_db, parameter, num_categories, inverted=False):
        sorted_param = sorted(regimen_db[parameter])
        if len(set(sorted_param)) == 1:
            return 1
        num_categories = min(len(set(sorted_param)), num_categories)
        categories = sorted([sorted_param[j] for j in [int(regimen_db.shape[0]/num_categories)*i for i in range(1, num_categories)]], reverse=inverted)
        # Compute the cost category for the regimen
        if inverted:
            return regimen_db[parameter].apply(lambda parameter: self.__get_category(parameter, categories))
        return regimen_db[parameter].apply(lambda parameter: 1 - self.__get_category(parameter, categories))

    def __normalise_parameter(self, regimen_db, parameter, inverted=False):
        minimum = regimen_db[parameter].min()
        range = regimen_db[parameter].max() - minimum
        # If alls regimens have the same value for 'parameter' return 1 for all regimens
        if range == 0:
            return 1
        if inverted:
            return regimen_db[parameter].apply(lambda parameter: 1-((parameter - minimum)/range))
        return regimen_db[parameter].apply(lambda parameter: ((parameter - minimum)/range))

    def __normal_vs_high_dose(self, regimen, resistance):
        for drug in regimen:
            if pd.isnull(self.drugs.loc[drug, 'high_dose']):
                continue
            if self.drugs.loc[drug, 'high_dose'] not in resistance:
                return False
        return True

    def __compute_score(self, row, categorised=False):
        if row['same_drug_class']:
            return 0
        columns = ['high_bactericidal_activity_early', 'high_bactericidal_activity', 'high_sterilizing_activity', 'efficacy', 'mechanism_of_action', 'route_of_administration', 'route_of_administration_hospitalized']
        if categorised:
            columns += score_categories
        else:
            columns += score_normalised
        return sum(row[columns])

    def save_rules_matrix(self, csv_file_name):
        self.regimens.to_csv(csv_file_name)

    def load_rules_matrix(self, csv_file_name):
        self.regimens = pd.read_csv(csv_file_name, index_col=0)
        self.regimens.regimen = self.regimens.regimen.apply(lambda regimen: ast.literal_eval(regimen))
        self.regimens.regimen_name = self.regimens.regimen_name.apply(lambda regimen: ast.literal_eval(regimen))
        self.regimens.regimen_name_formatted = self.regimens.regimen_name_formatted.apply(lambda regimen: ast.literal_eval(regimen))

    def generate_rules_matrix(self, num_effective_drugs=4, hospitalized=False, num_cost_categories=4, num_efficacy_categories=4, num_toxicity_categories=4):
        self.regimens = pd.DataFrame([[i] for i in itertools.combinations(self.drugs.index, num_effective_drugs)], columns = ['regimen'])
        self.regimens = self.regimens[self.regimens.regimen.apply(lambda regimen: self.__satisfies_regimen_contraints(regimen))]
        self.regimens['regimen_abbrev'] = self.regimens.regimen.apply(lambda regimen: ', '.join([self.drugs.loc[drug].abbreviation for drug in regimen]))
        self.regimens['regimen_name'] = self.regimens.regimen.apply(lambda regimen: [self.drugs.loc[drug]['name'] for drug in regimen])
        self.regimens['regimen_name_formatted'] = self.regimens.regimen.apply(lambda regimen: [self.drugs.loc[drug]['name_formatted'] for drug in regimen])


        ''' Van Deun's rules regarding bactericidal and sterilizing activity '''
        # The regimen should contain at least 1 core drug with moderate to high bactericidal and sterilizing activity
        self.regimens['van_deun_core_drug'] = self.regimens.regimen.apply(lambda regimen: self.__van_deun_core_drug(regimen))
        # The regimen should contain at least 1 high bactericidal, 1 moderate bactericidal, and 2 moderate sterlizing companion drugs
        self.regimens['van_deun_companion_drugs'] = self.regimens.regimen.apply(lambda regimen: self.__van_deun_companion_drugs(regimen))
        self.regimens['van_deun'] = self.regimens.apply(lambda row: True if row.van_deun_core_drug and row.van_deun_companion_drugs else False, axis=1)

        ''' Rules regarding resistance prevention '''
        # The regimen should consist of two drugs with a high (2) resistance prevention
        self.regimens['resistance_prevention'] = self.regimens.regimen.apply(lambda regimen: len([1 for drug in regimen if self.drugs.loc[drug].resistance_prevention > 2]) >= 2)

        ''' Rules regarding the mechanisms of action '''
        # The regimen should not contain two or more drugs with the same mechanism of action (i.e. all drugs (4) shuld have different mechanisms of action)
        self.regimens['mechanism_of_action'] = self.regimens.regimen.apply(lambda regimen: len(set([self.drugs.loc[drug].mechanism_of_action for drug in regimen])) >= 4)

        ''' Rules regarding the route of administration '''
        # Only drugs that can be given orally should be included in the regimen (unless the patient is hospitalized), the model will learn which one is more relevant
        self.regimens['route_of_administration'] = self.regimens.regimen.apply(lambda regimen: len([1 for drug in regimen if self.drugs.loc[drug].route_of_administration != 1]) == 0)
        self.regimens['route_of_administration_hospitalized'] = True

        ''' Rules regarding the cost of the regimen '''
        self.regimens['cost'] = self.regimens.regimen.apply(lambda regimen: self.__compute_cost_of_regimen(regimen))

        ''' Rules regarding the toxicity of the regimen '''
        self.regimens['toxicity'] = self.regimens.regimen.apply(lambda regimen: sum([self.drugs.loc[drug].toxicity for drug in regimen]))
        self.regimens['qt_prolongation'] = self.regimens.regimen.apply(lambda regimen: sum([self.drugs.loc[drug].qt_prolongation for drug in regimen]) > 3) # dont give two drugs with a moderate (2) qt prolongation

        ''' Rules regarding (early) bactericidal activity and sterilizing activity '''
        # The regimen should contain at least one drug with high early bactericidal activity
        self.regimens['high_bactericidal_activity_early'] = self.regimens.regimen.apply(lambda regimen: max([self.drugs.loc[drug].bactericidal_activity_early for drug in regimen]) >= 4)
        # The regimen should contain at least one drug with high bactericidal activity
        self.regimens['high_bactericidal_activity'] = self.regimens.regimen.apply(lambda regimen: max([self.drugs.loc[drug].bactericidal_activity for drug in regimen]) >= 4)
        # The regimen should contain at least one drug with high sterilizing activity
        self.regimens['high_sterilizing_activity'] = self.regimens.regimen.apply(lambda regimen: max([self.drugs.loc[drug].sterilizing_activity for drug in regimen]) >= 4)
        # The regimen contains at least one highly bactericidal_activity_early, bactericidal_activity, and sterilizing_activity
        self.regimens['efficacy'] = self.regimens.apply(lambda row: True if row.high_bactericidal_activity_early and row.high_bactericidal_activity and row.high_sterilizing_activity else False, axis=1)
        # The cumulative (early) bactericidal activity and sterilizing activity should be as high as possible
        self.regimens['bactericidal_activity_early'] = self.regimens.regimen.apply(lambda regimen: sum([self.drugs.loc[drug].bactericidal_activity_early for drug in regimen]))
        self.regimens['bactericidal_activity'] = self.regimens.regimen.apply(lambda regimen: sum([self.drugs.loc[drug].bactericidal_activity for drug in regimen]))
        self.regimens['sterilizing_activity'] = self.regimens.regimen.apply(lambda regimen: sum([self.drugs.loc[drug].sterilizing_activity for drug in regimen]))

        ''' Rules regarding the resistance prevention of the regimen '''
        self.regimens['resistance_prevention'] = self.regimens.regimen.apply(lambda regimen: sum([self.drugs.loc[drug].resistance_prevention for drug in regimen]))

        ''' Rules regarding the synergism and antagonism of the drugs in the regimen '''
        self.regimens[['synergism', 'antagonism', 'contraindications', 'same_drug_class']] = self.regimens.regimen.apply(lambda regimen: self.__get_synergism_antagonism_contraindications_samedrug(regimen))

        ''' Rules regarding the number of first line drugs in the regimen '''
        self.regimens['first_line'] = self.regimens.regimen.apply(lambda regimen: sum([self.drugs.loc[drug].first_line for drug in regimen]))

        ''' Rules regarding the clinical evidence of the drugs '''
        self.regimens['clinical_evidence'] = self.regimens.regimen.apply(lambda regimen: sum([db.drugs.loc[drug].clinical_experience for drug in regimen])) # dont give two drugs with a moderate (2) qt prolongation


    def get_top_regimens(self, resistance_profile, num_regimens=10, random_samples=0, hospitalized=False,
                num_cost_categories=4,
                num_efficacy_categories=4,
                num_toxicity_categories=4,
                num_bactericidal_activity_categories=4,
                num_bactericidal_activity_early_categories=4,
                num_sterilizing_activity_categories=4,
                num_resistance_prevention_categories=4,
                num_synergism_categories=4,
                num_antagonism_categories=4,
                num_contraindications_categories=4,
                num_same_drug_class_categories=4,
                num_first_line_categories=4,
                num_clinical_evidence_categories=4):
        resistance = [drug_id for drug_id, resistance in resistance_profile.items() if resistance > 0]
        regimens = self.regimens[self.regimens.regimen.apply(lambda regimen: len(set(regimen) - set(resistance)) == len(regimen))]
        regimens = regimens[regimens.regimen.apply(lambda regimen: self.__normal_vs_high_dose(regimen, resistance))].copy()

        regimens['cost_category'] = self.__categorize_parameter(regimens, 'cost', num_cost_categories, inverted=True)
        regimens['cost_normalised_inverted'] = self.__normalise_parameter(regimens, 'cost', inverted=True)

        ''' Rules regarding the toxicity of the regimen '''
        regimens['toxicity_category'] = self.__categorize_parameter(regimens, 'toxicity', num_toxicity_categories, inverted=True)
        regimens['toxicity_normalised_inverted'] = self.__normalise_parameter(regimens, 'toxicity', inverted=True)

        ''' Rules regarding the bactericidal activity of the regimen '''
        regimens['bactericidal_activity_category'] = self.__categorize_parameter(regimens, 'bactericidal_activity', num_bactericidal_activity_categories, inverted=False)
        regimens['bactericidal_activity_normalised'] = self.__normalise_parameter(regimens, 'bactericidal_activity', inverted=False)

        ''' Rules regarding the early bactericidal of the regimen '''
        regimens['bactericidal_activity_early_category'] = self.__categorize_parameter(regimens, 'bactericidal_activity_early', num_bactericidal_activity_early_categories, inverted=False)
        regimens['bactericidal_activity_early_normalised'] = self.__normalise_parameter(regimens, 'bactericidal_activity_early', inverted=False)

        ''' Rules regarding the sterilizing activity of the regimen '''
        regimens['sterilizing_activity_category'] = self.__categorize_parameter(regimens, 'sterilizing_activity', num_sterilizing_activity_categories, inverted=False)
        regimens['sterilizing_activity_normalised'] = self.__normalise_parameter(regimens, 'sterilizing_activity', inverted=False)

        ''' Rules regarding the resistance prevention of the regimen '''
        regimens['resistance_prevention_category'] = self.__categorize_parameter(regimens, 'resistance_prevention', num_resistance_prevention_categories, inverted=False)
        regimens['resistance_prevention_normalised'] = self.__normalise_parameter(regimens, 'resistance_prevention', inverted=False)

        ''' Rules regarding the synergism and antagonism of the drugs in the regimen '''
        regimens['synergism_category'] = self.__categorize_parameter(regimens, 'synergism', num_synergism_categories, inverted=False)
        regimens['synergism_normalised'] = self.__normalise_parameter(regimens, 'synergism', inverted=False)

        regimens['antagonism_category'] = self.__categorize_parameter(regimens, 'antagonism', num_antagonism_categories, inverted=True)
        regimens['antagonism_normalised_inverted'] = self.__normalise_parameter(regimens, 'antagonism', inverted=True)

        regimens['contraindications_category'] = self.__categorize_parameter(regimens, 'contraindications', num_contraindications_categories, inverted=True)
        regimens['contraindications_normalised_inverted'] = self.__normalise_parameter(regimens, 'contraindications', inverted=True)

        regimens['same_drug_class_category'] = self.__categorize_parameter(regimens, 'same_drug_class', num_same_drug_class_categories, inverted=True)
        regimens['same_drug_class_normalized_inverted'] = self.__normalise_parameter(regimens, 'same_drug_class', inverted=True)

        ''' Rules regarding the number of first line drugs in the regimen '''
        regimens['first_line_category'] = self.__categorize_parameter(regimens, 'first_line', num_first_line_categories, inverted=False)
        regimens['first_line_normalised'] = self.__normalise_parameter(regimens, 'first_line', inverted=False)

        ''' Rules regarding the clinical evidence of the drugs '''
        regimens['clinical_evidence_category'] = self.__categorize_parameter(regimens, 'clinical_evidence', num_clinical_evidence_categories, inverted=False)
        regimens['clinical_evidence_normalised'] = self.__normalise_parameter(regimens, 'clinical_evidence', inverted=False)

        regimens['score'] = regimens.apply(lambda row: self.__compute_score(row), axis=1)
        if num_regimens == -1:
            return regimens.sort_values(by='score', ascending=False)
        return regimens.sort_values(by='score', ascending=False).head(num_regimens).append(regimens.sample(random_samples))
