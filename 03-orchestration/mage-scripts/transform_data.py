if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from sklearn.feature_extraction import DictVectorizer

@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here

    df_train = data['df_train']
    df_val = data['df_val']

    def create_X(df, dv=None):
        categorical = ['PU_DO']
        numerical = ['trip_distance']
        dicts = df[categorical + numerical].to_dict(orient='records')
        if dv is None:
            dv = DictVectorizer(sparse=True)
            X = dv.fit_transform(dicts)
        else:
            X = dv.transform(dicts)
        return X, dv

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    return {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'dv': dv
    }


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
