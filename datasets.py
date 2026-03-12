import numpy as np
from sklearn.model_selection import train_test_split
import os


class IHDP(object):
    def __init__(self, path_data="datasets/IHDP/csv", replications=10):
        self.path_data = path_data
        self.replications = replications
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in range(25) if i not in self.binfeats]

    def __iter__(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            yield (x, t, y), (y_cf, mu_0, mu_1)

    def get_train_valid_test(self):
        for i in range(self.replications):
            data = np.loadtxt(self.path_data + '/ihdp_npci_' + str(i + 1) + '.csv', delimiter=',')
            t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
            mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]
            # this binary feature is in {1, 2}
            x[:, 13] -= 1
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.1, random_state=1)
            itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
            yield train, valid, test, self.contfeats, self.binfeats


class IHDP1000(object):
    """IHDP1000 dataset - larger version with ~7459 samples, pre-split train/test"""
    def __init__(self, path_data="datasets/IHDP1000"):
        self.path_data = path_data
        # IHDP1000 has same feature structure as IHDP
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.contfeats = [i for i in range(25) if i not in self.binfeats]

    def get_train_valid_test(self):
        """Yield train/valid/test splits for IHDP1000"""
        # Load training data
        path_train = os.path.join(self.path_data, "ihdp_npci_1-1000.train")
        t_train = np.load(os.path.join(path_train, "t.npy"))
        yf_train = np.load(os.path.join(path_train, "yf.npy"))
        ycf_train = np.load(os.path.join(path_train, "ycf.npy"))
        mu0_train = np.load(os.path.join(path_train, "mu0.npy"))
        mu1_train = np.load(os.path.join(path_train, "mu1.npy"))
        x_train = np.load(os.path.join(path_train, "x.npy"))

        # Load test data
        path_test = os.path.join(self.path_data, "ihdp_npci_1-1000.test")
        t_test = np.load(os.path.join(path_test, "t.npy"))
        yf_test = np.load(os.path.join(path_test, "yf.npy"))
        ycf_test = np.load(os.path.join(path_test, "ycf.npy"))
        mu0_test = np.load(os.path.join(path_test, "mu0.npy"))
        mu1_test = np.load(os.path.join(path_test, "mu1.npy"))
        x_test = np.load(os.path.join(path_test, "x.npy"))

        # IHDP1000 data structure: (n_samples, n_features, n_replications)
        # Need to transpose and reshape to get individual samples
        # Original: (672, 25, 1000) -> Target: (672000, 25)
        n_train_base, n_features, n_reps = x_train.shape
        n_test_base = x_test.shape[0]

        # Reshape data: (n_samples, n_features, n_replications) -> (n_samples * n_replications, n_features)
        x_train = x_train.transpose(0, 2, 1).reshape(-1, n_features)  # (672000, 25)
        t_train = t_train.T.reshape(-1)  # (672000,)
        yf_train = yf_train.T.reshape(-1)  # (672000,)
        ycf_train = ycf_train.T.reshape(-1)  # (672000,)
        mu0_train = mu0_train.T.reshape(-1)  # (672000,)
        mu1_train = mu1_train.T.reshape(-1)  # (672000,)

        x_test = x_test.transpose(0, 2, 1).reshape(-1, n_features)  # (750000, 25)
        t_test = t_test.T.reshape(-1)  # (750000,)
        yf_test = yf_test.T.reshape(-1)  # (750000,)
        ycf_test = ycf_test.T.reshape(-1)  # (750000,)
        mu0_test = mu0_test.T.reshape(-1)  # (750000,)
        mu1_test = mu1_test.T.reshape(-1)  # (750000,)

        # Split training into train/validation
        # Use smaller validation set due to large dataset
        idxtrain, ival = train_test_split(np.arange(x_train.shape[0]), test_size=0.1, random_state=1)

        # Reshape to match IHDP format
        t_train = t_train.reshape(-1, 1)
        t_test = t_test.reshape(-1, 1)
        yf_train = yf_train.reshape(-1, 1)
        yf_test = yf_test.reshape(-1, 1)
        ycf_train = ycf_train.reshape(-1, 1)
        ycf_test = ycf_test.reshape(-1, 1)
        mu0_train = mu0_train.reshape(-1, 1)
        mu0_test = mu0_test.reshape(-1, 1)
        mu1_train = mu1_train.reshape(-1, 1)
        mu1_test = mu1_test.reshape(-1, 1)

        # Adjust binary feature (same as IHDP)
        x_train[:, 13] -= 1
        x_test[:, 13] -= 1

        # Create splits
        train = (x_train[idxtrain], t_train[idxtrain], yf_train[idxtrain]), \
                (ycf_train[idxtrain], mu0_train[idxtrain], mu1_train[idxtrain])
        valid = (x_train[ival], t_train[ival], yf_train[ival]), \
                (ycf_train[ival], mu0_train[ival], mu1_train[ival])
        test = (x_test, t_test, yf_test), (ycf_test, mu0_test, mu1_test)

        yield train, valid, test, self.contfeats, self.binfeats


class TWINS(object):
    """TWINS dataset - real-world twin birth data, no counterfactuals"""
    def __init__(self, path_data="datasets/TWINS"):
        self.path_data = path_data
        # TWINS has 50 features: need to identify binary vs continuous
        # For now, we'll load and determine from data
        self.binfeats = None
        self.contfeats = None

    def _determine_feature_types(self, x):
        """Determine which features are binary vs continuous from data"""
        self.binfeats = []
        self.contfeats = []
        for i in range(x.shape[1]):
            unique_vals = np.unique(x[:, i])
            if len(unique_vals) <= 10:  # Assume binary/categorical if <=10 unique values
                self.binfeats.append(i)
            else:
                self.contfeats.append(i)

    def get_train_valid_test(self):
        """Yield train/valid/test splits for TWINS dataset"""
        # Load data
        x_data = np.loadtxt(os.path.join(self.path_data, "twin_pairs_X_3years_samesex.csv"),
                           delimiter=",", skiprows=1)
        t_data = np.loadtxt(os.path.join(self.path_data, "twin_pairs_T_3years_samesex.csv"),
                           delimiter=",", skiprows=1)
        y_data = np.loadtxt(os.path.join(self.path_data, "twin_pairs_Y_3years_samesex.csv"),
                           delimiter=",", skiprows=1)

        # Remove first column (index column)
        x = x_data[:, 1:].astype(np.float32)
        t = t_data[:, 1].astype(np.float32)  # Treatment column

        # TWINS data has outcomes for both twins
        y0 = y_data[:, 1].astype(np.float32)  # Twin 0 mortality
        y1 = y_data[:, 2].astype(np.float32)  # Twin 1 mortality

        # Determine feature types
        self._determine_feature_types(x)

        # No counterfactuals available for TWINS (real-world data)
        y_cf = None
        mu_0 = None
        mu_1 = None

        # Handle NaN values
        x = np.nan_to_num(x, nan=0.0)
        t = np.nan_to_num(t, nan=0.0)
        y0 = np.nan_to_num(y0, nan=0.0)
        y1 = np.nan_to_num(y1, nan=0.0)

        # Create factual outcome based on treatment
        # In TWINS: treatment=1 means twin0 got higher mortality risk
        y = np.where(t == 1, y1, y0).reshape(-1, 1)
        t = t.reshape(-1, 1)

        # Split data
        idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.2, random_state=1)
        itr, iva = train_test_split(idxtrain, test_size=0.2, random_state=1)

        train = (x[itr], t[itr], y[itr]), (None, None, None)
        valid = (x[iva], t[iva], y[iva]), (None, None, None)
        test = (x[ite], t[ite], y[ite]), (None, None, None)

        yield train, valid, test, self.contfeats, self.binfeats
