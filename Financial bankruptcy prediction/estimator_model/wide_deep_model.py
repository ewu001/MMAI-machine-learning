import tensorflow as tf

# Define feature columns
def get_wide_deep(NEMBEDS):
    # Define column types
    is_male,mother_age,plurality,gestation_weeks = \
        [\
            tf.feature_column.categorical_column_with_vocabulary_list('is_male', 
                        ['True', 'False', 'Unknown']),
            tf.feature_column.numeric_column('mother_age'),
            tf.feature_column.categorical_column_with_vocabulary_list('plurality',
                        ['Single(1)', 'Twins(2)', 'Triplets(3)',
                         'Quadruplets(4)', 'Quintuplets(5)','Multiple(2+)']),
            tf.feature_column.numeric_column('gestation_weeks')
        ]

    # Discretize
    age_buckets = tf.feature_column.bucketized_column(mother_age, 
                        boundaries=np.arange(15,45,1).tolist())
    gestation_buckets = tf.feature_column.bucketized_column(gestation_weeks, 
                        boundaries=np.arange(17,47,1).tolist())
      
    # Sparse columns are wide, have a linear relationship with the output
    wide = [is_male,
            plurality,
            age_buckets,
            gestation_buckets]
    
    # Feature cross all the wide columns and embed into a lower dimension
    crossed = tf.feature_column.crossed_column(wide, hash_bucket_size=20000)
    embed = tf.feature_column.embedding_column(crossed, NEMBEDS)
    
    # Continuous columns are deep, have a complex relationship with the output
    deep = [mother_age,
            gestation_weeks,
            embed]
    return wide, deep


def wide_deep_estimator(output_dir, wide, deep, NNSIZE, run_config):
    ## TODO 2b: change the dnn_hidden_units to NNSIZE
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = output_dir,
        linear_feature_columns = wide,
        dnn_feature_columns = deep,
        dnn_hidden_units = NNSIZE,
        config = run_config)
    return estimator