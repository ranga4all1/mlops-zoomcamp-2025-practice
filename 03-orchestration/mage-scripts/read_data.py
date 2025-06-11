if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd

@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your data loading logic here
    color = kwargs.get('global_vars', {}).get('color', 'green')
    year = kwargs.get('global_vars', {}).get('year', 2023)
    month = kwargs.get('global_vars', {}).get('month', 1)


    def read_dataframe(year, month):
        url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet'
        df = pd.read_parquet(url)
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
        return df

    df_train = read_dataframe(year, month)
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    df_val = read_dataframe(next_year, next_month)

    return {'df_train': df_train, 'df_val': df_val}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
