from sklearn.base import BaseEstimator, TransformerMixin

class DropColumns(BaseEstimator, TransformerMixin):
    """Transformer to drop a fixed list of columns."""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        # nothing to learn
        return self

    def transform(self, X):
        # we expect X to be a DataFrame
        return X.drop(columns=self.columns, errors="ignore")

CURRENT_YEAR = 2022

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 current_year=CURRENT_YEAR,
                 vintage_cutoff=30,
                 big_engine_cutoff=7.0):
        self.current_year = current_year

    def fit(self, X, y=None):
        # nothing to learn
        return self

    def transform(self, X):
        # take a DataFrame (raw CSV) and return engineered DataFrame
        df = X.copy()

        # 1) Car age
        df["Car_Age"] = self.current_year - df["Year"]

        # 2) Vintage flag
        df["IsVintage"] = (df["Car_Age"] >= self.vintage_cutoff)

        # 3) Big engine flag ( >7L )
        df["IsBigEngine"] = (df["Engine_Size"] > self.big_engine_cutoff)

        # 4) drop unused
        df = df.drop(columns=["Negotiable", "Mileage_per_Year", 
                              "Unnatural_High_Mileage_Flag", "Year"],
                     errors="ignore")

        return df