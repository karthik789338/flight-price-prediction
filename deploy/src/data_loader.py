import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Loads dataset from a given path.
    """
    df = pd.read_csv(path)
    return df

