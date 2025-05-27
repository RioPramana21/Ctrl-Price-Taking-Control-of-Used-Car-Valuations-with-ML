from sklearn.base import BaseEstimator, TransformerMixin

CURRENT_YEAR = 2022

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, current_year=CURRENT_YEAR):
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
        df["IsVintage"] = (df["Car_Age"] >= 30)

        # 3) Big engine flag ( >7L )
        df["IsBigEngine"] = (df["Engine_Size"] > 7.0)

        # 4) drop unused
        df = df.drop(columns=["Negotiable", "Mileage_per_Year", 
                              "Unnatural_High_Mileage_Flag", "Year"],
                     errors="ignore")

        return df