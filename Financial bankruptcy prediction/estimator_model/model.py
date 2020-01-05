import tensorflow as tf
import numpy as np

# Define feature columns
def get_wide_deep(NEMBEDS):
    # Define column types
    
    Tobin_Q = tf.feature_column.numeric_column("Tobin_Q")
    EPS = tf.feature_column.numeric_column("EPS")
    Productivity = tf.feature_column.numeric_column("Productivity")
    Leverage_Ratio = tf.feature_column.numeric_column("Leverage_Ratio")
    Asset_Turnover = tf.feature_column.numeric_column("Asset_Turnover")
    Operational_Margin = tf.feature_column.numeric_column("Operational_Margin")
    Return_Equity = tf.feature_column.numeric_column("Return_Equity")
    Market_Book_Ratio = tf.feature_column.numeric_column("Market_Book_Ratio")
    Assets_Growth = tf.feature_column.numeric_column("Assets_Growth")
    Employee_Growth = tf.feature_column.numeric_column("Employee_Growth")
        
    fiscal_year = tf.feature_column.numeric_column('Fiscal_year')
    profitability = tf.feature_column.numeric_column('Profitability')
    sales_Growth = tf.feature_column.numeric_column('Sales_Growth')


    # Discretize
    year_buckets = tf.feature_column.bucketized_column(fiscal_year, 
                        boundaries=np.arange(1978,2018,1).tolist())
    profitability_buckets = tf.feature_column.bucketized_column(profitability, 
                        boundaries=np.arange(-9,142,3).tolist())
    sales_growth_buckets = tf.feature_column.bucketized_column(sales_Growth, 
                     boundaries=np.arange(-4,39850,500).tolist())
      
    # Sparse columns are wide, have a linear relationship with the output
    wide = [year_buckets,
            profitability_buckets,
            sales_growth_buckets]
    
    # Feature cross all the wide columns and embed into a lower dimension
    crossed = tf.feature_column.crossed_column(wide, hash_bucket_size=20000)
    embed = tf.feature_column.embedding_column(crossed, NEMBEDS)
    
    # Continuous columns are deep, have a complex relationship with the output
    deep = [Tobin_Q,
            EPS,
            Productivity,
            Leverage_Ratio,
            Asset_Turnover,
            Operational_Margin,
            Return_Equity,
            Market_Book_Ratio,
            Assets_Growth,
            Employee_Growth,
            embed]
    return wide, deep


# use Adam optimizer with learning rate decay
def wide_deep_estimator(output_dir, wide, deep, NNSIZE, run_config):

    estimator = tf.estimator.DNNLinearCombinedClassifier(
        model_dir = output_dir,
        linear_feature_columns = wide,
        dnn_feature_columns = deep,
        dnn_hidden_units = NNSIZE,
        config = run_config,
        dnn_optimizer=lambda: tf.train.AdamOptimizer(
                        learning_rate=tf.train.exponential_decay(
                        learning_rate=0.1,
                        global_step=tf.train.get_global_step(),
                        decay_steps=10000,
                        decay_rate=0.96)))
    return estimator


  # use Adam optimizer with learning rate decay
def DNNmodel(output_dir, columns, unit_list, runconfig):
    estimator = tf.estimator.DNNClassifier(
                       model_dir = output_dir,
                       feature_columns = columns,
                       hidden_units = unit_list,
                       config = runconfig,
                       optimizer=lambda: tf.train.AdamOptimizer(
                            learning_rate=tf.train.exponential_decay(
                            learning_rate=0.1,
                            global_step=tf.train.get_global_step(),
                            decay_steps=10000,
                            decay_rate=0.96)))
    return estimator 
