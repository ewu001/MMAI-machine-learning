import argparse
import os

from . import tf_estimator_model

import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        help = 'location to write checkpoint, event file for tensorboard and model export',
        required = True
    )
    parser.add_argument(
        '--batch_size',
        help = 'Number of examples to compute gradient over.',
        type = int,
        default = 512
    )
    parser.add_argument(
        '--nnsize',
        help = 'Hidden layer sizes to use for DNN feature columns -- provide space-separated layers',
        nargs = '+',
        type = int,
        default=[128, 32, 4]
    )
    parser.add_argument(
        '--train_examples',
        help = 'Number of examples (in thousands) to run the training job over. If this is more than actual # of examples available, it cycles through them. So specifying 1000 here when you have only 100k examples makes this 10 epochs.',
        type = int,
        default = 5000
    )
    parser.add_argument(
        '--eval_steps',
        help = 'Positive number of steps for which to evaluate model. Default to None, which means to evaluate until input_fn raises an end-of-input exception',
        type = int,       
        default = None
    )


    ## parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__    

    ## assign the arguments to the model variables
    output_dir = arguments.pop('output_dir')
    tf_estimator_model.BATCH_SIZE = arguments.pop('batch_size')
    tf_estimator_model.TRAIN_STEPS = (arguments.pop('train_examples') * 1000) / tf_estimator_model.BATCH_SIZE
    tf_estimator_model.EVAL_STEPS = arguments.pop('eval_steps')    
    print ("Will train for {} steps using batch_size={}".format(tf_estimator_model.TRAIN_STEPS, tf_estimator_model.BATCH_SIZE))
    tf_estimator_model.NNSIZE = arguments.pop('nnsize')
    print ("Will use DNN size of {}".format(tf_estimator_model.NNSIZE))


    # Run the training job
    tf_estimator_model.train_and_evaluate(output_dir)