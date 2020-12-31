import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES

def median_absolute_deviation(X, center=None):
    if center is None:
        return np.nanmedian(np.abs(X), axis=0)
    else:
        return np.nanmedian(np.abs(X - center), axis=0)

def m_estimate_loc_scale(X, loc, var, gamma, max_iter):
    for _ in range(max_iter):
        w = (np.exp(-0.5 * (X-loc)**2 / var) / np.sqrt(2 * np.pi * var))**gamma
        w = w / np.sum(w, axis=0)
        loc = np.sum(w*X, axis=0)
        var = (1 + gamma) * np.sum(w * (X - loc)**2, axis=0)
    return loc, np.sqrt(var)

def m_estimate_scale(X, var, gamma, max_iter):
    for _ in range(max_iter):
        w = (np.exp(-0.5 * X**2 / var) / np.sqrt(2 * np.pi * var))**gamma
        w = w / np.sum(w, axis=0)
        var = (1 + gamma) * np.sum(w * X**2, axis=0)
    return np.sqrt(var)

class RobustScaler(TransformerMixin, BaseEstimator):

    def __init__(self, *, gamma=0.2, with_centering=True, with_scaling=True, max_iter=1000):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.gamma = gamma
        self.max_iter = max_iter

    def fit(self, X, y=None):
        X = self._validate_data(X, accept_sparse=False, dtype=FLOAT_DTYPE, force_all_finite=True)

        if self.with_centering:
            center = np.nanmedian(X, axis=0)
            scale = median_absolute_deviation(X, center=center) / 0.675
            center, scale = m_estimate_loc_scale(X, center, scale**2, self.gamma, self.max_iter)

            self.center_ = center
            if self.with_scaling:
                self.scale_ = scale
            else:
                self.scale_ = None
        else:
            self.center_ = None
            scale = median_absolute_deviation(X) / 0.675
            self.scale_ = m_estimate_scale(X, scale**2, self.gamma, self.max_iter)

        return self
    
    def transform(self, X):
        check_is_fitted(self)

        if self.with_centering:
            X -= self.center_
        if self.with_scaling:
            X /= self.scale_
        
        return X

    def inverse_transform(self, X):
        check_is_fitted(self)

        if self.with_scaling:
            X *= self.scale_
        if self.with_centering:
            X += self.center_
        
        return X

