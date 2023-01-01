import pandas as pd


def predicted_test_data_to_result_csv(df_test, predicted_values):
    review_ids = df_test.loc[:, 'review_id']

    max_predicted_values = predicted_values.argmax(axis=1) + 1

    final_df = pd.concat([review_ids, pd.DataFrame(max_predicted_values)], axis=1)

    final_df.columns = ['review_id', 'rating']

    final_df['rating'] = final_df['rating'].apply(int)

    final_df.to_csv('result.csv', index_label=False, index=False)