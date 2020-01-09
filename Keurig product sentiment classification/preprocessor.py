from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def tokenization(max_length, comments):
    '''
    Input: max length configuration, and comments

    Return: pre processed sequences that is padded
    '''
    MAX_LENGTH = max_length
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(comments)
    post_seq = tokenizer.texts_to_sequences(comments)
    post_seq_padded = pad_sequences(post_seq, maxlen=MAX_LENGTH)
    return post_seq_padded, tokenizer