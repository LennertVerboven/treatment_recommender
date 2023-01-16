# treatment_recommender

required libraries: pandas and scikit learn

Download the repository and cd into it
a dummy ressitance profile is provided

```
# train the model using the training data and store it in the pickles directory
python train_model.py pickles/full_training_set_v2_includes_all_rounds_supplemented_with_modified_regimens_from_harvesting_rounds_to_exclude_three_drugs.csv pickles/model.pkl

# run the treatment recommender for the resistance profile given in ./profile.csv and write the recommendations to ./recommendations.csv
python3 algorithm.py ./profile.csv ./recommendations.csv

```
