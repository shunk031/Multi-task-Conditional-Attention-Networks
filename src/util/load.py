import pandas as pd


def load_data(dataset_fpath):

    df = pd.read_csv(str(dataset_fpath))
    print(f'Raw data size: {df.shape}')

    df['cvr'] = df[['conversion', 'click']].apply(
        lambda x: (x[0] / x[1]) * 100 if x[1] > 0 else 0, axis=1)

    return df
