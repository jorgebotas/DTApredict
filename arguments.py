import argparse
import os



def argparser():
  parser = argparse.ArgumentParser()
  # for model
  parser.add_argument(
      '--seq_filter_length',
      type=int,
      default=8,
      help='Filter size for Conv1D layer for protein sequence codification'
  )
  parser.add_argument(
      '--smi_filter_length',
      type=int,
      default=4,
      help='Filter size for Conv1D layer for SMILES codification'
  )
  parser.add_argument(
      '--dom_filter_length',
      type=int,
      default=4,
      help='Filter size for Conv1D layer for Prosite domains codification'
  )
  parser.add_argument(
      '--num_filters',
      type=int,
      default=32,
      help='Number of filters in initial Conv1D layer\n\
            Second layer: 2 * num_filters,\n\
            Third layer: 3 * num_filters\n\
            Default: 32'
  )
  parser.add_argument(
      '--num_hidden',
      type=int,
      default=0,
      help='Number of neurons in hidden layer.'
  )
  parser.add_argument(
      '--num_classes',
      type=int,
      default=0,
      help='Number of classes (families).'
  )
  parser.add_argument(
      '--max_seq_len',
      type=int,
      default=0,
      help='Length of input sequences.'
  )
  parser.add_argument(
      '--max_smi_len',
      type=int,
      default=0,
      help='Length of input sequences.'
  )
  parser.add_argument(
      '--word_representation',
      type=bool,
      default=False,
      help='Word representation of SMILES and sequences\n\
            Default: False. Character representation'
  )
  parser.add_argument(
      '--seq_wordlen',
      type=int,
      default=3,
      help='Length of word in sequence word representation\n\
            Default: 3'
  )
  parser.add_argument(
      '--smi_wordlen',
      type=int,
      default=8,
      help='Length of word in SMILES word representation\n\
            Default: 8'
  )
  parser.add_argument(
       '--deep_smiles',
       type=bool,
       default=True,
       help='Use DeepSMILES representation of drug compounds'
  )
  # for learning
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--num_epoch',
      type=int,
      default=100,
      help='Number of epochs to train.\n\
            Default: 100'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=256,
      help='Batch size. Must divide evenly into the dataset sizes.\n\
            Default: 256'
  )
  parser.add_argument(
      '--extract_domains',
      type=bool,
      default=True,
      help='Whether to extract Prosite domains from protein sequences'
  )
  parser.add_argument(
      '--provided_domains',
      type=bool,
      default=False,
      help='Whether Prosite domains are externally available in data path.'
  )
  parser.add_argument(
      '--dataset_path',
      type=str,
      default='/data/kiba/',
      help='Directory for input data.'
  )
  parser.add_argument(
      '--is_log',
      type=int,
      default=0,
      help='use log transformation for Y'
  )
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='',
      help='Path to write checkpoint file.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp',
      help='Directory for log data.'
  )



  FLAGS, dummy_unparsed = parser.parse_known_args()

  return FLAGS




def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, "log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)
