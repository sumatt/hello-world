from sklearn.preprocessing import PolynomialFeatures

class CustomPolynomialFeatures:
    def __init__(self, degree=2, interaction_only=False, include_bias=True):
        self.transformer = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        return self.transformer.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
