import tensorflow as tf
from read_utils import TextConverter
from model import CharRNN
import os

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
flags.DEFINE_boolean('use_embedding', False, 'whether to use embedding')
flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
flags.DEFINE_string('converter_path', '', 'model/name/converter.pkl')
flags.DEFINE_string('checkpoint_path', '', 'checkpoint path')
flags.DEFINE_string('start_string', '', 'use this string to start generating')
flags.DEFINE_integer('max_length', 30, 'max length to generate')


def generate():
    tf.compat.v1.disable_eager_execution()
    converter = TextConverter(filename=FLAGS.converter_path)
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path =\
            tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = CharRNN(converter.vocab_size, sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)

    model.load(FLAGS.checkpoint_path)

    start = converter.text_to_arr(FLAGS.start_string)
    arr = model.sample(FLAGS.max_length, start, converter.vocab_size)

    return converter.arr_to_text(arr)

def call():
    FLAGS.use_embedding = True
    FLAGS.converter_path = os.getenv('converter_path', 'model/poetry/converter.pkl')
    FLAGS.checkpoint_path = os.getenv('checkpoint_path', 'model/poetry/')
    FLAGS.max_length = 300
    return generate()

def main(_):
    print(generate())


if __name__ == '__main__':
    tf.app.run()
