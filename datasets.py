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


class IHDP100(object):
    """IHDP100 dataset - 100 replications, each trained separately"""
    def __init__(self, path_data="datasets/IHDP100", n_replications=None):
        """
        Args:
            path_data: Path to IHDP100 data directory
            n_replications: Number of replications to use (None for all 100)
        """
        self.path_data = path_data
        self.n_replications = n_replications if n_replications else 100
        # IHDP100 has same feature structure as IHDP
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.contfeats = [i for i in range(25) if i not in self.binfeats]

    def get_train_valid_test(self):
        """Yield train/valid/test splits for IHDP100, one replication at a time"""
        # Load training data
        path_train = os.path.join(self.path_data, "ihdp_npci_1-100.train")
        t_train = np.load(os.path.join(path_train, "t.npy"))
        yf_train = np.load(os.path.join(path_train, "yf.npy"))
        ycf_train = np.load(os.path.join(path_train, "ycf.npy"))
        mu0_train = np.load(os.path.join(path_train, "mu0.npy"))
        mu1_train = np.load(os.path.join(path_train, "mu1.npy"))
        x_train = np.load(os.path.join(path_train, "x.npy"))

        # Load test data
        path_test = os.path.join(self.path_data, "ihdp_npci_1-100.test")
        t_test = np.load(os.path.join(path_test, "t.npy"))
        yf_test = np.load(os.path.join(path_test, "yf.npy"))
        ycf_test = np.load(os.path.join(path_test, "ycf.npy"))
        mu0_test = np.load(os.path.join(path_test, "mu0.npy"))
        mu1_test = np.load(os.path.join(path_test, "mu1.npy"))
        x_test = np.load(os.path.join(path_test, "x.npy"))

        for i in range(min(self.n_replications, x_train.shape[2])):
            # Extract single replication
            xtr = x_train[:, :, i]
            ttr = t_train[:, i]
            ytr = yf_train[:, i]
            ycftr = ycf_train[:, i]
            mu0tr = mu0_train[:, i]
            mu1tr = mu1_train[:, i]

            xte = x_test[:, :, i]
            tte = t_test[:, i]
            yte = yf_test[:, i]
            ycfte = ycf_test[:, i]
            mu0te = mu0_test[:, i]
            mu1te = mu1_test[:, i]

            # Adjust binary feature
            xtr[:, 13] -= 1
            xte[:, 13] -= 1

            # Split train into train/validation
            idxtr, iva = train_test_split(np.arange(xtr.shape[0]), test_size=0.3, random_state=1)

            train = (xtr[idxtr], ttr[idxtr].reshape(-1, 1), ytr[idxtr].reshape(-1, 1)), \
                    (ycftr[idxtr].reshape(-1, 1), mu0tr[idxtr].reshape(-1, 1), mu1tr[idxtr].reshape(-1, 1))
            valid = (xtr[iva], ttr[iva].reshape(-1, 1), ytr[iva].reshape(-1, 1)), \
                    (ycftr[iva].reshape(-1, 1), mu0tr[iva].reshape(-1, 1), mu1tr[iva].reshape(-1, 1))
            test = (xte, tte.reshape(-1, 1), yte.reshape(-1, 1)), \
                   (ycfte.reshape(-1, 1), mu0te.reshape(-1, 1), mu1te.reshape(-1, 1))

            yield train, valid, test, self.contfeats, self.binfeats


class IHDP1000(object):
    """IHDP1000 dataset - 1000 replications, each trained separately"""
    def __init__(self, path_data="datasets/IHDP1000", n_replications=None):
        """
        Args:
            path_data: Path to IHDP1000 data directory
            n_replications: Number of replications to use (None for all 1000)
        """
        self.path_data = path_data
        self.n_replications = n_replications if n_replications else 1000
        # IHDP1000 has same feature structure as IHDP
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.contfeats = [i for i in range(25) if i not in self.binfeats]

    def get_train_valid_test(self):
        """Yield train/valid/test splits for IHDP1000, one replication at a time"""
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

        # Yield each replication separately
        for i in range(min(self.n_replications, x_train.shape[2])):
            # Extract single replication
            xtr = x_train[:, :, i]
            ttr = t_train[:, i]
            ytr = yf_train[:, i]
            ycftr = ycf_train[:, i]
            mu0tr = mu0_train[:, i]
            mu1tr = mu1_train[:, i]

            xte = x_test[:, :, i]
            tte = t_test[:, i]
            yte = yf_test[:, i]
            ycfte = ycf_test[:, i]
            mu0te = mu0_test[:, i]
            mu1te = mu1_test[:, i]

            # Adjust binary feature
            xtr[:, 13] -= 1
            xte[:, 13] -= 1

            # Split train into train/validation
            idxtr, iva = train_test_split(np.arange(xtr.shape[0]), test_size=0.3, random_state=1)

            train = (xtr[idxtr], ttr[idxtr].reshape(-1, 1), ytr[idxtr].reshape(-1, 1)), \
                    (ycftr[idxtr].reshape(-1, 1), mu0tr[idxtr].reshape(-1, 1), mu1tr[idxtr].reshape(-1, 1))
            valid = (xtr[iva], ttr[iva].reshape(-1, 1), ytr[iva].reshape(-1, 1)), \
                    (ycftr[iva].reshape(-1, 1), mu0tr[iva].reshape(-1, 1), mu1tr[iva].reshape(-1, 1))
            test = (xte, tte.reshape(-1, 1), yte.reshape(-1, 1)), \
                   (ycfte.reshape(-1, 1), mu0te.reshape(-1, 1), mu1te.reshape(-1, 1))

            yield train, valid, test, self.contfeats, self.binfeats


class JOBS(object):
    """JOBS dataset - 10 replications, has true ATE but no counterfactuals"""
    def __init__(self, path_data="datasets/Jobs", n_replications=None):
        self.path_data = path_data
        self.n_replications = n_replications if n_replications else 10
        # JOBS has 17 features - will determine binary vs continuous from data
        self.binfeats = None
        self.contfeats = None
        # Load true ATE for evaluation
        self.true_ate = float(np.load(os.path.join(path_data, "jobs_DW_bin.new.10.train", "ate.npy")).flatten()[0])

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
        """Yield train/valid/test splits for JOBS dataset, one replication at a time"""
        # Load training data
        path_train = os.path.join(self.path_data, "jobs_DW_bin.new.10.train")
        t_train = np.load(os.path.join(path_train, "t.npy"))
        yf_train = np.load(os.path.join(path_train, "yf.npy"))
        x_train = np.load(os.path.join(path_train, "x.npy"))

        # Load test data
        path_test = os.path.join(self.path_data, "jobs_DW_bin.new.10.test")
        t_test = np.load(os.path.join(path_test, "t.npy"))
        yf_test = np.load(os.path.join(path_test, "yf.npy"))
        x_test = np.load(os.path.join(path_test, "x.npy"))

        for i in range(min(self.n_replications, x_train.shape[2])):
            # Extract single replication
            xtr = x_train[:, :, i]
            ttr = t_train[:, i]
            ytr = yf_train[:, i]

            xte = x_test[:, :, i]
            tte = t_test[:, i]
            yte = yf_test[:, i]

            # Determine feature types from first replication
            if self.binfeats is None:
                self._determine_feature_types(xtr)

            # No counterfactuals for JOBS
            y_cf = None
            mu_0 = None
            mu_1 = None

            # Split train into train/validation
            idxtr, iva = train_test_split(np.arange(xtr.shape[0]), test_size=0.2, random_state=1)

            train = (xtr[idxtr], ttr[idxtr].reshape(-1, 1), ytr[idxtr].reshape(-1, 1)), \
                    (None, None, None)
            valid = (xtr[iva], ttr[iva].reshape(-1, 1), ytr[iva].reshape(-1, 1)), \
                    (None, None, None)
            test = (xte, tte.reshape(-1, 1), yte.reshape(-1, 1)), \
                   (None, None, None)

            yield train, valid, test, self.contfeats, self.binfeats


class TWINS(object):
    """TWINS dataset - real-world twin birth data

    Following the reproduction guide:
    - pair -> individual conversion: each twin pair becomes 2 samples
    - treatment: T=1 if twin is heavier (twin 1), T=0 if lighter (twin 0)
    - potential outcomes: Y0, Y1 are constructed from mortality data
    - evaluation: PEHE and ATE error can be computed since we have both outcomes
    """
    def __init__(self, path_data="datasets/TWINS", n_replications=10):
        self.path_data = path_data
        self.n_replications = n_replications
        # TWINS has 50 features (after removing index and id columns)
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
        """Yield train/valid/test splits for TWINS dataset

        Following reproduction guide:
        1. Convert pair-level data to individual-level
        2. Treatment: T=1 (heavier twin), T=0 (lighter twin)
        3. Construct potential outcomes Y0, Y1
        4. Split: 80% train, 20% test; train -> 20% validation
        """
        # Load data using genfromtxt to handle missing values
        x_data = np.genfromtxt(os.path.join(self.path_data, "twin_pairs_X_3years_samesex.csv"),
                               delimiter=",", skip_header=1, filling_values=0.0)
        y_data = np.genfromtxt(os.path.join(self.path_data, "twin_pairs_Y_3years_samesex.csv"),
                               delimiter=",", skip_header=1, filling_values=0.0)

        # Data structure:
        # - x_data: (N_pairs, 58) - first column is index, rest are features
        # - y_data: (N_pairs, 3) - index, mort_0, mort_1
        n_pairs = x_data.shape[0]

        # Remove index columns
        x_raw = x_data[:, 1:]  # (N_pairs, 57)
        mort_0 = y_data[:, 1].astype(np.float32)  # Twin 0 mortality (lighter)
        mort_1 = y_data[:, 2].astype(np.float32)  # Twin 1 mortality (heavier)

        # === Step 1: Pair -> Individual conversion ===
        # For each twin pair, create 2 individual samples
        # Twin 0 is lighter (T=0), Twin 1 is heavier (T=1)

        # X: (2*N_pairs, n_features) - duplicate features for both twins
        x = np.repeat(x_raw, 2, axis=0).astype(np.float32)

        # T: (2*N_pairs, 1) - treatment assignment
        # [0, 1, 0, 1, ...] - alternating 0 (lighter) and 1 (heavier)
        t = np.zeros(n_pairs * 2, dtype=np.float32)
        t[1::2] = 1  # Every second sample is T=1 (heavier twin)
        t = t.reshape(-1, 1)

        # === Step 2: Construct potential outcomes ===
        # Y0: outcome if untreated (mortality when lighter)
        # Y1: outcome if treated (mortality when heavier)
        # Since we know both outcomes for each pair, we can create true counterfactuals

        # For T=0 samples (lighter twins): Y = mort_0, Y_cf = mort_1
        # For T=1 samples (heavier twins): Y = mort_1, Y_cf = mort_0
        y0_all = np.repeat(mort_0, 2).reshape(-1, 1)  # Y0 for all
        y1_all = np.repeat(mort_1, 2).reshape(-1, 1)  # Y1 for all

        # Factual outcome Y
        y = np.where(t == 1, y1_all, y0_all)

        # Counterfactual Y_cf
        y_cf = np.where(t == 1, y0_all, y1_all)

        # True ITE for PEHE calculation
        mu_0 = y0_all
        mu_1 = y1_all

        # Remove columns with all NaN or too many missing values
        # (infant_id columns, etc.)
        # Keep first 47 features (remove infant_id_0, infant_id_1, dlivord_min, dtotord_min, etc.)
        x = x[:, :47]

        # Handle NaN values
        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        y_cf = np.nan_to_num(y_cf, nan=0.0)
        mu_0 = np.nan_to_num(mu_0, nan=0.0)
        mu_1 = np.nan_to_num(mu_1, nan=0.0)

        # Determine feature types
        self._determine_feature_types(x)

        # Yield multiple replications with different random splits
        for rep in range(self.n_replications):
            # Use different random_state for each replication
            random_state = 1 + rep

            # === Step 3: Data split (80/20 as per reproduction guide) ===
            idxtrain, ite = train_test_split(np.arange(x.shape[0]), test_size=0.2, random_state=random_state)
            itr, iva = train_test_split(idxtrain, test_size=0.2, random_state=random_state)

            train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
            valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
            test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])

            yield train, valid, test, self.contfeats, self.binfeats
