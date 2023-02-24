import pandas as pd
import tensorflow as tf


def predicted_test_data_to_result_csv(df_test, predicted_values, exp_name):
    review_ids = df_test.loc[:, 'review_id']

    max_predicted_values = predicted_values.argmax(axis=1) + 1

    final_df = pd.concat([review_ids, pd.DataFrame(max_predicted_values)], axis=1)

    final_df.columns = ['review_id', 'rating']

    final_df['rating'] = final_df['rating'].apply(int)

    final_df.to_csv(f'res_files/{exp_name}.csv', index_label=False, index=False)


def load_dataset(frac_ratio: float):
    train_path = "data/base/goodreads_train.csv"

    df = pd.read_csv(train_path, sep=",")

    index = df[(df['rating'] == 0)].index
    df.drop(index, inplace=True)
    df.reset_index(inplace=True, drop=True)

    x_train = df.sample(frac=frac_ratio)
    x_val = df.drop(x_train.index)

    y_train = x_train.pop('rating')
    y_train = y_train - 1

    y_val = x_val.pop('rating')
    y_val = y_val - 1

    raw_train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10, reshuffle_each_iteration=False)

    raw_val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(10, reshuffle_each_iteration=False)

    return x_train, x_val, y_train, y_val
