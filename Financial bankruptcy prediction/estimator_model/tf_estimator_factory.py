import shutil
import tensorflow as tf
from . import model

# Declare global variables
# Determine CSV, label, and key columns
CSV_COLUMNS = ['Fiscal_year', 'Tobin_Q', 'EPS', 'Profitability', 'Productivity',
       'Leverage_Ratio', 'Asset_Turnover', 'Operational_Margin',
       'Return_Equity', 'Market_Book_Ratio', 'Assets_Growth', 'Sales_Growth',
       'Employee_Growth', 'BK']

LABEL_COLUMN = "BK"

# Set default values for each CSV column
DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

# Define some hyperparameters
TRAIN_STEPS = 10000
EVAL_STEPS = None
BATCH_SIZE = 512
NEMBEDS = 3
NNSIZE = [64, 32, 8]

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def read_dataset(filename, mode, batch_size = BATCH_SIZE):
    def _input_fn():
        def decode_csv(value_column):
          columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
          features = dict(zip(CSV_COLUMNS, columns))
          label = features.pop(LABEL_COLUMN)
          return features, label

        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

        # Create dataset from file list
        dataset = (tf.data.TextLineDataset(file_list)  # Read text file
                     .map(decode_csv))  # Transform each elem by applying decode_csv fn

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size=10*batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset
    return _input_fn


def get_cols():
  # Define column types
  return [\
          tf.feature_column.numeric_column('Fiscal_year', dtype=tf.float64),
          tf.feature_column.numeric_column("Tobin_Q"),
          tf.feature_column.numeric_column("EPS"),
          tf.feature_column.numeric_column("Profitability"),
          tf.feature_column.numeric_column("Productivity"),
          tf.feature_column.numeric_column("Leverage_Ratio"),
          tf.feature_column.numeric_column("Asset_Turnover"),
          tf.feature_column.numeric_column("Operational_Margin"),
          tf.feature_column.numeric_column("Return_Equity"),
          tf.feature_column.numeric_column("Market_Book_Ratio"),
          tf.feature_column.numeric_column("Assets_Growth"),
          tf.feature_column.numeric_column("Sales_Growth"),
          tf.feature_column.numeric_column("Employee_Growth")
      ]

# Create serving input function to be able to serve predictions later using provided inputs
def serving_input_fn():
    feature_placeholders = {
        "Fiscal_year": tf.placeholder(tf.float64, [None]),
        "Tobin_Q": tf.placeholder(tf.float32, [None]),
        "EPS": tf.placeholder(tf.float32, [None]),
        'Profitability': tf.placeholder(tf.float32, [None]),
        "Productivity": tf.placeholder(tf.float32, [None]),
        "Leverage_Ratio": tf.placeholder(tf.float32, [None]),
        "Asset_Turnover": tf.placeholder(tf.float32, [None]),
        "Operational_Margin": tf.placeholder(tf.float32, [None]),
        "Return_Equity": tf.placeholder(tf.float32, [None]),
        "Market_Book_Ratio": tf.placeholder(tf.float32, [None]),
        "Assets_Growth": tf.placeholder(tf.float32, [None]),
        "Sales_Growth": tf.placeholder(tf.float32, [None]),
        "Employee_Growth": tf.placeholder(tf.float32, [None]),
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)



# Create estimator to train and evaluate
def train_and_evaluate(output_dir):
  shutil.rmtree('logs', ignore_errors = True)
  tf.summary.FileWriterCache.clear()

  EVAL_INTERVAL = 300
  run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL,
                                      save_summary_steps = 100,
                                      keep_checkpoint_max = 3)
  columns = get_cols()
  DNNestimator = model.DNNmodel(output_dir, columns, NNSIZE, run_config)

  train_spec = tf.estimator.TrainSpec(
                       input_fn = read_dataset('data store/training_dataset_processed_tf.csv', mode = tf.estimator.ModeKeys.TRAIN),
                       max_steps = TRAIN_STEPS )
  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
  eval_spec = tf.estimator.EvalSpec(
                       input_fn = read_dataset('data store/evaluation_dataset_processed_tf.csv', mode = tf.estimator.ModeKeys.EVAL),
                       steps = EVAL_STEPS,
                       start_delay_secs = 60, # start evaluating after N seconds
                       throttle_secs = EVAL_INTERVAL,  # evaluate every N seconds
                       exporters = exporter)
  tf.estimator.train_and_evaluate(DNNestimator, train_spec, eval_spec)



# Create wide and deep estimator to train and evaluate
def train_and_evaluate_wide_deep(output_dir):
  shutil.rmtree('logs', ignore_errors = True)
  tf.summary.FileWriterCache.clear()

  EVAL_INTERVAL = 300
  run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL,
                                      save_summary_steps = 100,
                                      keep_checkpoint_max = 3)

  # Define estimator, and inject metrics
  wide, deep = model.get_wide_deep(NEMBEDS)
  wideDeepEstimator = model.wide_deep_estimator(output_dir, wide, deep, NNSIZE, run_config)

  train_spec = tf.estimator.TrainSpec(
                       input_fn = read_dataset('data store/training_dataset_processed_tf.csv', mode = tf.estimator.ModeKeys.TRAIN),
                       max_steps = TRAIN_STEPS )
  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
  eval_spec = tf.estimator.EvalSpec(
                       input_fn = read_dataset('data store/evaluation_dataset_processed_tf.csv', mode = tf.estimator.ModeKeys.EVAL),
                       steps = EVAL_STEPS,
                       start_delay_secs = 60, # start evaluating after N seconds
                       throttle_secs = EVAL_INTERVAL,  # evaluate every N seconds
                       exporters = exporter)
  tf.estimator.train_and_evaluate(wideDeepEstimator, train_spec, eval_spec)



