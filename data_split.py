
import sklearn

import warnings
warnings.filterwarnings("ignore")

train_df = None
validation_df = None
test_df = None


def train_test_split():
    for age_range_id in df.age_range_id.unique():
        split_df, tmp_test_df = sklearn.model_selection.train_test_split(df[df.age_range_id == age_range_id],
                                                                         test_size=0.1)
        tmp_train_df, tmp_validation_df = sklearn.model_selection.train_test_split(
            split_df, test_size=0.2)

        if train_df is None:
            train_df = tmp_train_df.copy(deep=True)
        else:
            train_df = train_df.append(tmp_train_df, ignore_index=True)

        if validation_df is None:
            validation_df = tmp_validation_df.copy(deep=True)
        else:
            validation_df = validation_df.append(
                tmp_validation_df, ignore_index=True)

        if test_df is None:
            test_df = tmp_test_df.copy(deep=True)
        else:
            test_df = test_df.append(tmp_test_df, ignore_index=True)


data_split()
