import os
import tarfile
import settings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class HousingData:

	def __init__(self):
		self.data_set = None
		self.housing = pd.DataFrame
		self.housing_labels = []

	@staticmethod
	def get_dataset_from_archive(archive_file_name=settings.HOUSING_ARCHIVE_NAME):
		try:
			tgz_path = os.path.join(settings.DATA_DIR, archive_file_name)
			housing_tgz = tarfile.open(tgz_path)
			housing_tgz.extractall(path=settings.DATA_DIR)
			housing_tgz.close()
		except FileNotFoundError:
			raise ValueError('Archive not found!')

	@staticmethod
	def load_data(dataset_file_name=settings.HOUSING_DATASET_NAME):
		try:
			csv_path = os.path.join(settings.DATA_DIR, dataset_file_name)
			return pd.read_csv(csv_path)
		except FileNotFoundError:
			raise ValueError('CSV file not found!')

	@staticmethod
	def familiarity_with_data(dataset):
		# what's columns and type of data
		# print(dataset.head())

		# what's size and fullness datas
		# print(dataset.info())

		# what's category for ocean_proximing
		print(dataset['ocean_proximity'].value_counts())

		# create plots in order to see data in graphics
		# dataset.hist(bins=50, figsize=(20, 15))
		# plt.show()

		# search corelations
		corr_matrix = dataset.corr()
		print('Correlation to median house value:\n', corr_matrix['median_house_value'].sort_values(ascending=False))

	@staticmethod
	def get_train_and_test_sets(full_dataset):
		# just for familiarity, no need to use
		train_set, test_set = train_test_split(full_dataset, test_size=0.2, random_state=42)
		return train_set, test_set

	@staticmethod
	def get_stratified_train_and_test_sets(full_dataset):
		full_dataset['income_cat'] = np.ceil(full_dataset['median_income'] / 1.5)
		full_dataset['income_cat'].where(full_dataset['income_cat'] < 5, 5.0, inplace=True)
		split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
		train_set = pd.DataFrame
		test_set = pd.DataFrame

		for train_index, test_index in split.split(full_dataset, full_dataset['income_cat']):
			train_set = full_dataset.loc[train_index]
			test_set = full_dataset.loc[test_index]

		return train_set, test_set

	def prepare_data(self, data_set, is_production):
		self.data_set = data_set

		if not is_production:
			self._separate_labels_and_parameters()
		else:
			self.housing = self.data_set
		self._txt_category_values_to_int()
		self._fill_na_values()
		self._add_custom_parameters()

		# self._normalize_data() # after normalization the accuracy is reduced
		# it's need more time to understand why
		# for the prototype the accuracy is quite good

		# self._familiarity_with_data_with_custom_parameters() # it need for debugging and data understanding

		if not is_production:
			return self.housing, self.housing_labels
		else:
			return self.housing

	def _separate_labels_and_parameters(self):
		self.housing = self.data_set.drop('median_house_value', axis=1)
		self.housing_labels = self.data_set['median_house_value'].copy()

	def _fill_na_values(self):
		imputer = SimpleImputer(strategy='median')
		x = self.housing.values
		x_na_filled = imputer.fit_transform(x)
		self.housing = pd.DataFrame(x_na_filled, columns=self.housing.columns, index=self.housing.index)

	def _txt_category_values_to_int(self):

		# approach #3
		# parameters are choosed manualy
		num_categories = {
			'ocean_proximity': {
			'<1H OCEAN': 60,
			'INLAND': 30,
			'ISLAND': 100,
			'NEAR BAY': 70,
			'NEAR OCEAN': 65
		}}
		self.housing.replace(num_categories, inplace=True)

		# чем выше категория, тем больше влияние на размер дохода
		# подобранно очень грубо
		self.housing['income_cat'] = self.housing['income_cat'] * 10

	def _add_custom_parameters(self):
		bedrooms_per_room = self.housing['total_bedrooms'] / self.housing['total_rooms']
		rooms_per_household = self.housing['total_rooms'] / self.housing['households']
		self.housing = self.housing.assign(bedrooms_per_room = bedrooms_per_room,
		                                   rooms_per_household = rooms_per_household)

	def _normalize_data(self):
		x = self.housing.values
		#scaler = MinMaxScaler()
		scaler = StandardScaler()
		x_scaled = scaler.fit_transform(x)
		self.housing = pd.DataFrame(x_scaled, columns=self.housing.columns, index=self.housing.index)

	def _familiarity_with_data_with_custom_parameters(self):
		caller = self.housing
		other = self.housing_labels
		temp_housing = caller.join(other=other, lsuffix='_caller', rsuffix='_other')
		corr_matrix = temp_housing.corr()
		print('Correlation to median house value with custom parameters:\n', corr_matrix['median_house_value'].sort_values(ascending=False))