from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def get_preprocessor(cat_cols):
    """
    Возвращает минимальный препроцессор: One-Hot Encoding для категорий.
    
    Args:
        cat_cols: список названий категориальных столбцов
    
    Returns:
        ColumnTransformer
    """
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'
    )