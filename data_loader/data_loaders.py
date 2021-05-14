import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd


class DogeDataLoader:
    def __init__(self,
                 filename,
                 categorical_cols,
                 target_col,
                 seq_length,
                 batch_size,
                 preprocessor=True,
                 prediction_window=1):
        '''
        :param filename: path to the csv dataset
        :param categorical_cols: name of the categorical columns, if None pass empty list
        :param target_col: name of the targeted column
        :param seq_length: window length to use
        :param prediction_window: window length to predict
        :param preprocessor: if normalize data or not
        :param batch_size: batch size
        '''
        self.data = self.read_and_preprocess(filename)
        self.categorical_cols = categorical_cols
        self.numerical_cols = list(set(self.data.columns) - set(categorical_cols) - set(target_col))
        self.target_col = target_col
        self.seq_length = seq_length
        self.prediction_window = prediction_window
        self.batch_size = batch_size
        self.preprocessor = preprocessor
        self.preprocess = ColumnTransformer(
            [("scaler", StandardScaler(), self.numerical_cols),
             #("encoder", OneHotEncoder(), self.categorical_cols)
             ],
            remainder="passthrough"
        )

    def read_and_preprocess(self, filename):
        # Reading
        df = pd.read_csv(filename)
        # Reorder and resetting index
        df = df[::-1].reset_index(drop=True)
        # Preprocessing 'Change' column
        df['Change %'] = df['Change %'].str.replace("%", "")
        df['Change %'] = pd.to_numeric(df['Change %'].str.replace(",", ""))
        # Preprocessing 'Vol.' column
        vols = [el for el in df['Vol.']]
        for num, el in enumerate(vols):
            # Check if is billion
            isB = el[-1] == 'B'
            try:
                el = float(el[:-1])
            except ValueError:
                print("Value Error at row ", num)
                el = vols[num - 1]
            if isB:
                el = el * 1000
            vols[num] = el
        df['Vol.'] = vols
        # Dropping Date column
        df.pop('Date')
        # Done, returning dataframe
        return df

    def preprocess_data(self):
        '''
        Preprocessing function
        '''
        X = self.data.drop(self.target_col, axis=1)
        y = self.data[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
        if self.preprocessor is not None:
            X_train = self.preprocess.fit_transform(X_train)
            X_test = self.preprocess.fit_transform(X_test)

        if self.target_col:
            return X_train, X_test, y_train.values, y_test.values
        return X_train, X_test

    def frame_series(self, X, y=None):
        '''
        Function used to prepare the data for time series prediction
        :param X: set of features
        :param y: targeted value to predict
        :return: TensorDataset
        '''
        nb_obs, nb_features = X.shape
        features, target, y_hist = [], [], []

        for i in range(1, nb_obs - self.seq_length - self.prediction_window):
            features.append(torch.FloatTensor(X[i:i + self.seq_length, :]).unsqueeze(0))

        features_var = torch.cat(features)

        if y is not None:
            for i in range(1, nb_obs - self.seq_length - self.prediction_window):
                target.append(
                    torch.tensor(y[i + self.seq_length:i + self.seq_length + self.prediction_window]))
            target_var = torch.cat(target)
            return TensorDataset(features_var, target_var)
        return TensorDataset(features_var)

    def get_loaders(self,):
        '''
        Preprocess and frame the dataset
        :return: DataLoaders associated to training and testing data
        '''

        X_train, X_test, y_train, y_test = self.preprocess_data()

        train_dataset = self.frame_series(X_train, y_train)
        test_dataset = self.frame_series(X_test, y_test)

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_iter = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        return train_iter, test_iter
