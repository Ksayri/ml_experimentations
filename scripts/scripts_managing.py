from scripts.data import HousingData
from scripts.model import HousingModel

from sklearn.metrics import mean_squared_error
import numpy as np


#  --------------
# Step 1
#  getting data

data_obj = HousingData()
# data_obj.get_dataset_from_archive()
housing_data = data_obj.load_data()
# data_obj.familiarity_with_data(housing_data)
train_set, test_set = data_obj.get_stratified_train_and_test_sets(housing_data)


# --------------
# Step 2
# train model

# train_set_prepared, train_set_labels = data_obj.prepare_data(train_set, is_production=False)
# housing_model = HousingModel()
# housing_model.send_train_data(train_set_prepared, train_set_labels)
# housing_model.compare_models()
# housing_model.configure_hyperparameters()
# final_model = housing_model.train_final_model()


# --------------
# Step 3
# test model with test set

# test_set_prepared, test_set_labels = data_obj.prepare_data(test_set, is_production=False)
# labels_prediction = final_model.predict(test_set_prepared)
#
# test_data_set_mse = mean_squared_error(test_set_labels, labels_prediction)
# test_data_set_rmse = np.sqrt(test_data_set_mse)
# print('Accuracy with test data. RMSE=', test_data_set_rmse)


# ---------------
# Step 4
# if model testing is OK, save final_model for future use

# housing_model.save_final_model(final_model)


# ---------------
# how to use for prediction

# emulating of production data
example_data_set = test_set.iloc[[2]]
example_data_set = example_data_set.drop('median_house_value', axis=1)
example_set_prepared = data_obj.prepare_data(example_data_set, is_production=True)

# prediction
housing_model = HousingModel()
final_model = housing_model.get_final_model()
labels_prediction = final_model.predict(example_set_prepared)
print('Predicted median house value: $', int(labels_prediction[0]))


