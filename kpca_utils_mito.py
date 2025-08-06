# kpca_utils.py

import pandas as pd
from sklearn.decomposition import KernelPCA

def split_columns(df):
    float_cols = df.select_dtypes(include=['float64', 'float32']).columns
    non_float_cols = df.select_dtypes(exclude=['float64', 'float32']).columns
    return df[float_cols], df[non_float_cols]

def fit_kernel_pca(float_df, n_components=85, kernel='rbf', **kwargs):
    kpca = KernelPCA(n_components=n_components, kernel=kernel, **kwargs)
    transformed = kpca.fit_transform(float_df)
    return kpca, transformed

def transform_with_kernel_pca(float_df, fitted_kpca):
    return fitted_kpca.transform(float_df)

def concat_transformed_with_nonfloat(transformed_array, non_float_df, index=None):
    df_transformed = pd.DataFrame(
        transformed_array,
        index=index if index is not None else non_float_df.index,
        columns=[f'PCA_{i+1}' for i in range(transformed_array.shape[1])]
    )
    return pd.concat([df_transformed, non_float_df], axis=1)