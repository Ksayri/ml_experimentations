import settings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import numpy as np


class HousingModel:
	def __init__(self):
		self.final_model_filename = settings.FINAL_MODEL_FILE_NAME
		self.train_set = None
		self.train_labels = None

	def send_train_data(self, train_set, train_labels):
		self.train_set = train_set
		self.train_labels = train_labels

	def compare_models(self):
		self._run_liner_regression()
		self._run_decision_tree_regressor()
		self._run_random_forest_regression()

	def configure_hyperparameters(self):
		forest_reg = RandomForestRegressor()
		param_grid = [
			{'n_estimators': [20, 30, 40, 50], 'max_features': [8, 10, 12]}
		]
		grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
		grid_search.fit(self.train_set, self.train_labels)
		print('The best hyper parameters = ', grid_search.best_params_)

	def train_final_model(self):
		# hyperparameters are choosed by grid search
		final_model = RandomForestRegressor(max_features=12, n_estimators=40)

		# system evaluation with training set
		final_model.fit(self.train_set, self.train_labels)

		scores = cross_val_score(final_model, self.train_set, self.train_labels,
		                         scoring='neg_mean_squared_error', cv=10)
		final_model_rmse_scores = np.sqrt(-scores)
		mean_forest_reg_rmse = final_model_rmse_scores.mean()
		print('\nModel - Final Model (Random Forest Regressor). Train set Mean RMSE=', mean_forest_reg_rmse)
		print('Standard deviation = ', final_model_rmse_scores.std())
		return final_model

	def save_final_model(self, final_model):
		try:
			joblib.dump(final_model, self.final_model_filename)
		except IOError:
			raise ValueError('Something wrong with file save operation.')

	def get_final_model(self):
		try:
			return joblib.load(self.final_model_filename)
		except FileNotFoundError:
			raise ValueError('Model file not found!')

	def _run_liner_regression(self):
		lin_reg = LinearRegression()
		lin_reg.fit(self.train_set, self.train_labels)
		housing_predictions = lin_reg.predict(self.train_set)
		lin_mse = mean_squared_error(self.train_labels, housing_predictions)
		lin_rmse = np.sqrt(lin_mse)
		print('Model - Liner Regression. RMSE=', lin_rmse)
		return lin_reg

	def _run_decision_tree_regressor(self):
		tree_reg = DecisionTreeRegressor()
		tree_reg.fit(self.train_set, self.train_labels)
		scores = cross_val_score(tree_reg, self.train_set, self.train_labels,
		                         scoring='neg_mean_squared_error', cv=10)
		tree_rmse_scores = np.sqrt(-scores)
		mean_tree_rmse = tree_rmse_scores.mean()
		print('\nModel - Decision Tree Regressor. Mean RMSE=', mean_tree_rmse)
		print('Standard deviation = ', tree_rmse_scores.std())

	def _run_random_forest_regression(self):
		forest_reg = RandomForestRegressor()
		forest_reg.fit(self.train_set, self.train_labels)
		scores = cross_val_score(forest_reg, self.train_set, self.train_labels,
		                         scoring='neg_mean_squared_error', cv=10)
		forest_reg_rmse_scores = np.sqrt(-scores)
		mean_forest_reg_rmse = forest_reg_rmse_scores.mean()
		print('\nModel - Random Forest Regressor. Mean RMSE=', mean_forest_reg_rmse)
		print('Standard deviation = ', forest_reg_rmse_scores.std())


