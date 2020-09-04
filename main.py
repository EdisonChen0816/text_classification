# encoding=utf-8
from src.util.logger import setlogger
from src.util.yaml_util import loadyaml
from src.textcnn import TextCNN
from gensim.models import KeyedVectors
import os
import tensorflow as tf


config = loadyaml('conf/config.yaml')
logger = setlogger(config)

# tf配置
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8


def test_cnntext():
    w2v = KeyedVectors.load(config['w2v_path'])
    cfg = {
        'logger': logger,
        'train_data': config['train_path'],
        'eval_data': config['eval_path'],
        'max_len': 92,
        'w2v': w2v,
        'filters': 16,
        'kernel_size': 3,
        'pool_size': 3,
        'strides': 1,
        'loss': 'adam',
        'rate': 0.01,
        'epoch': 20,
        'batch_size': 4096,
        'dropout': 0.1,
        'model_path': config['model_path'],
        'summary_path': config['summary_path'],
        'tf_config': tf_config
    }
    model = TextCNN(**cfg)
    model.fit()
    model.load(config['predict_path'])
    model.predict('詹姆斯G3决杀，你怎么看？')
    model.close()


if __name__ == '__main__':
    test_cnntext()