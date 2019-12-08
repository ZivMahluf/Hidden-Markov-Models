import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np

START_STATE = '*START*'
START_WORD = '*START*'
END_STATE = '*END*'
END_WORD = '*END*'
RARE_WORD = '*RARE_WORD*'

EPSILON = 0.00000001
BATCH_SIZE = 5


def data_example(data_path='PoS_data.pickle',
                 words_path='all_words.pickle',
                 pos_path='all_PoS.pickle'):
    """
    An example function for loading and printing the Parts-of-Speech data for
    this exercise.
    Note that these do not contain the "rare" values and you will need to
    insert them yourself.

    :param data_path: the path of the PoS_data file.
    :param words_path: the path of the all_words file.
    :param pos_path: the path of the all_PoS file.
    """

    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)

    print("The number of sentences in the data set is: " + str(len(data)))
    print("\nThe tenth sentence in the data set, along with its PoS is:")
    print(data[10][1])
    print(data[10][0])

    print("\nThe number of words in the data set is: " + str(len(words)))
    print("The number of parts of speech in the data set is: " + str(len(pos)))

    print("one of the words is: " + words[34467])
    print("one of the parts of speech is: " + pos[17])

    print(pos)


class Baseline(object):
    '''
    The baseline model.
    '''

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the baseline Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}

        # pos_probs is |pos|, and emissions is size |pos|*|words|
        self.pos_probabilities, self.emissions = baseline_mle(training_set, self)

        pass

    def MAP(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        # create tables
        pos_predictions = []  # list of lists of pos

        for sentence in sentences:
            sentence_pos_prediction = []
            for word in sentence:
                if word not in self.words:
                    word = RARE_WORD
                max_prob, max_pos_index = 0, 0
                for index, pos in enumerate(self.pos_tags):
                    value = self.pos_probabilities[pos] * self.emissions[pos][word]
                    if value > max_prob:
                        max_prob, max_pos_index = value, index
                sentence_pos_prediction.append(self.pos_tags[max_pos_index])
            pos_predictions.append(sentence_pos_prediction)
        return pos_predictions


def baseline_mle(training_set, model):
    """
    a function for calculating the Maximum Likelihood estimation of the
    multinomial and emission probabilities for the baseline model.

    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial baseline model with the pos2i and word2i mappings among other things.
    :return: a mapping of the multinomial and emission probabilities. You may implement
            the probabilities in |PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """
    # get info from data
    X = [training_set[i][1] for i in range(len(training_set))]
    y = [training_set[i][0] for i in range(len(training_set))]

    # update word appearance rate and get amount of words in total
    words_appearance_rate, total_amount_of_words = get_word_appearance_rate(X, model.words, True)

    # get pos tags appearance rate
    pos_appearance, pos_appearance_prob = defaultdict(int), defaultdict(int)
    for pos_labels in y:
        for pos in pos_labels:
            pos_appearance[pos] += 1
    # mean div the pos tags appearance
    for pos in model.pos_tags:
        pos_appearance_prob[pos] = pos_appearance[pos] / total_amount_of_words

    pos_by_word = {pos: defaultdict(int) for pos in model.pos_tags}
    # get P(x|y) by getting all words for each pos
    for i in range(len(X)):
        sentence = X[i]
        pos_of_sentence = y[i]
        for j in range(len(sentence)):  # iterate over each word and it's corresponding pos
            # we divide by the amount of words of the pos
            pos_by_word[pos_of_sentence[j]][sentence[j]] += 1 / pos_appearance[pos_of_sentence[j]]

    return pos_appearance_prob, pos_by_word


class HMM(object):
    '''
    The basic HMM_Model with multinomial transition functions.
    '''

    def __init__(self, pos_tags, words, training_set):
        '''
        The init function of the basic HMM Model.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        '''

        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.i2pos = {i: pos for (i, pos) in enumerate(pos_tags)}
        self.i2word = {i: word for (i, word) in enumerate(words)}

        # transitions is |pos|*|pos|, and emissions is size |pos|*|words|
        self.transition, self.emissions = hmm_mle(training_set, self)

    def sample(self, n):
        '''
        Sample n sequences of words from the HMM.
        :return: A list of word sequences.
        '''
        sequences = []
        for i in range(n):
            current_pos = START_STATE
            seq = [START_WORD]
            while current_pos != END_STATE:
                current_dict = self.transition[current_pos]
                current_values = list(current_dict.values())
                current_keys = list(current_dict.keys())
                # randomly pick the next pos from the current pos options, by the probs of the transition matrix
                next_pos_index = np.random.choice(len(current_dict), 1, p=current_values)[0]
                next_pos = current_keys[next_pos_index]
                next_pos_emission_words = list(self.emissions[next_pos].keys())
                next_word = np.random.choice(next_pos_emission_words, 1)[0]
                seq.append(next_word)
                current_pos = next_pos
            sequences.append(seq)
        return sequences

    def viterbi(self, sentences):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of PoS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :return: iterable sequence of PoS tag sequences.
        '''
        pos_predictions = []  # list of lists of pos
        for sentence in sentences:
            sentence_pos_prediction = []
            dynamic_matrix = {k: defaultdict(int) for k in range(len(sentence))}  # size |sentence| * |pos| * |2|
            # init the matrix with P(y_0 = START_STATE) = 1 and P(y_0 = pos) = 0 for all other pos
            for pos in self.pos_tags:
                dynamic_matrix[0][pos] = (0, pos)
            dynamic_matrix[0][START_STATE] = (1, START_STATE)

            for k in range(1, len(sentence)):
                word = sentence[k]
                if word not in self.words:
                    word = RARE_WORD
                for u in self.pos_tags:
                    # find w which maximizes the equation
                    max_val, max_w = 0.0, START_STATE
                    for w in self.pos_tags:
                        value = dynamic_matrix[k - 1][w][0] * self.transition[w][u] * self.emissions[u][word]
                        if value > max_val:
                            max_val, max_w = value, w
                    dynamic_matrix[k][u] = (max_val, max_w)
            # find the prediction of the sentence by iterating backwards on the dynamic matrix
            max_pos_val = 0.0
            max_pos = START_STATE
            # find max pos
            for pos in self.pos_tags:
                pos_val = dynamic_matrix[len(sentence) - 1][pos][0]
                if pos_val > max_pos_val:
                    max_pos, max_pos_val = pos, pos_val
            for k in range(len(sentence) - 1, -1, -1):  # iterate backwards
                sentence_pos_prediction.append(max_pos)
                _, max_pos = dynamic_matrix[k][max_pos]
            sentence_pos_prediction.reverse()  # because we appended backwards
            pos_predictions.append(sentence_pos_prediction)
        return pos_predictions


def hmm_mle(training_set, model):
    """
    a function for calculating the Maximum Likelihood estimation of the
    transition and emission probabilities for the standard multinomial HMM.

    :param training_set: an iterable sequence of sentences, each containing
            both the words and the PoS tags of the sentence (as in the "data_example" function).
    :param model: an initial HMM with the pos2i and word2i mappings among other things.
    :return: a mapping of the transition and emission probabilities. You may implement
            the probabilities in |PoS|x|PoS| and |PoS|x|Words| sized matrices, or
            any other data structure you prefer.
    """
    # get info from data
    X = [training_set[i][1] for i in range(len(training_set))]
    y = [training_set[i][0] for i in range(len(training_set))]

    # get pos tags appearance rate
    pos_appearance = defaultdict(int)
    for sentence_pos in y:
        for pos in sentence_pos:
            pos_appearance[pos] += 1

    # transition matrix
    transitions = {pos: defaultdict(float) for pos in model.pos_tags}
    for sentence_pos in y:
        for k in range(len(sentence_pos) - 1):
            transitions[sentence_pos[k]][sentence_pos[k + 1]] += 1 / pos_appearance[sentence_pos[k]]

    emissions = {pos: defaultdict(int) for pos in model.pos_tags}
    # get P(x|y) by getting all words for each pos
    for pos_of_sentence, sentence in training_set:
        for j in range(len(sentence)):  # iterate over each word and it's corresponding pos
            # we divide by the amount of words of the pos
            emissions[pos_of_sentence[j]][sentence[j]] += 1 / pos_appearance[pos_of_sentence[j]]

    return transitions, emissions


class MEMM(object):
    '''
    The base Maximum Entropy Markov Model with log-linear transition functions.
    '''

    def __init__(self, pos_tags, words, training_set, phi):
        '''
        The init function of the MEMM.
        :param pos_tags: the possible hidden states (POS tags)
        :param words: the possible emissions (words).
        :param training_set: A training set of sequences of POS-tags and words.
        :param phi: the feature mapping function, which accepts two PoS tags
                    and a word, and returns a list of indices that have a "1" in
                    the binary feature vector.
        '''
        self.words = words
        self.pos_tags = pos_tags
        self.words_size = len(words)
        self.pos_size = len(pos_tags)
        self.pos2i = {pos: i for (i, pos) in enumerate(pos_tags)}
        self.word2i = {word: i for (i, word) in enumerate(words)}
        self.phi = phi
        self.i2pos = {i: pos for (i, pos) in enumerate(pos_tags)}

        self.phi2_vector_size = self.pos_size * self.pos_size + self.pos_size * self.words_size + 3

    def viterbi(self, sentences, w):
        '''
        Given an iterable sequence of word sequences, return the most probable
        assignment of POS tags for these words.
        :param sentences: iterable sequence of word sequences (sentences).
        :param w: a dictionary that maps a feature index to it's weight.
        :return: iterable sequence of POS tag sequences.
        '''
        pos_predictions = []
        for sentence in sentences:
            k = len(sentence)
            dynamic_matrix = np.zeros((k, self.pos_size, 2))
            pos_pred = []
            # init the matrix by using START as the first pos
            for j in range(self.pos_size):
                dynamic_matrix[0, j, 0] = np.sum(w[self.phi(self, sentence[0], self.i2pos[j], START_STATE)])

            for i in range(1, k):
                if sentence[i] not in self.words:
                    sentence[i] = RARE_WORD
                for j in range(self.pos_size):
                    temp = np.zeros(self.pos_size)
                    for t in range(self.pos_size):
                        temp[t] = np.sum(w[self.phi(self, sentence[i], self.i2pos[j], self.i2pos[t])])

                    temp += dynamic_matrix[i - 1, :, 0]
                    dynamic_matrix[i, j, 0] = np.max(temp)
                    dynamic_matrix[i, j, 1] = np.argmax(temp)

            pos_idx = np.argmax(dynamic_matrix[-1, :, 0])
            for i in range(k - 1, -1, -1):  # run backwards
                pos_pred.append(self.i2pos[pos_idx])
                next_pos_idx = dynamic_matrix[i, pos_idx, 1]
                pos_idx = int(next_pos_idx)

            pos_pred.reverse()
            pos_predictions.append(pos_pred)

        return pos_predictions

    def phi_func(self, word, y_new, y_old):
        index_word = self.word2i[word]
        index_y_new = self.pos2i[y_new]
        index_y_old = self.pos2i[y_old]
        # index at transitions matrix, and index at emissions matrix that comes after the transition matrix
        return np.array([index_y_old * self.pos_size + index_y_new,
                         self.pos_size * self.pos_size + index_word * self.pos_size + index_y_new])

    def phi_func2(self, word, y_new, y_old):
        index_word = self.word2i[word]
        index_y_new = self.pos2i[y_new]
        index_y_old = self.pos2i[y_old]
        indices = [index_y_old * self.pos_size + index_y_new,
                   self.pos_size * self.pos_size + index_word * self.pos_size + index_y_new]

        # add index of features
        if word.isupper():
            indices.append(self.phi2_vector_size + 1)
        if word.isdigit():
            indices.append(self.phi2_vector_size + 2)
        if len(word) < 6:
            indices.append(self.phi2_vector_size + 3)

        return np.array(indices)


def perceptron(training_set, initial_model, w0, eta=0.1, epochs=1):
    """
    learn the weight vector of a log-linear model according to the training set.
    :param training_set: iterable sequence of sentences and their parts-of-speech.
    :param initial_model: an initial MEMM object, containing among other things
            the phi feature mapping function.
    :param w0: an initial weights vector.
    :param eta: the learning rate for the perceptron algorithm.
    :param epochs: the amount of times to go over the entire training data (default is 1).
    :return: w, the learned weights vector for the MEMM.
    """
    for epoch in range(epochs):
        for pos, sentence in training_set:
            pos_pred = initial_model.viterbi([sentence], w0)[0]
            w0[initial_model.phi(initial_model, sentence[0], pos[0], START_STATE)] += eta
            w0[initial_model.phi(initial_model, sentence[0], pos_pred[0], START_STATE)] -= eta
            for j in range(1, len(sentence)):
                w0[initial_model.phi(initial_model, sentence[j], pos[j], pos[j - 1])] += eta
                w0[initial_model.phi(initial_model, sentence[j], pos_pred[j], pos_pred[j - 1])] -= eta
    return w0


# --------------- Added functions ---------------
def get_data():
    with open('PoS_data.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('all_words.pickle', 'rb') as f:
        words = pickle.load(f)
    with open('all_PoS.pickle', 'rb') as f:
        pos = pickle.load(f)
    return data, words, pos


def preprocess_data(data, words, pos):
    '''
    add the start and end states, append the start word and end word to every sentence and PoS, change rare words
    :param data: list
    :param words: list
    :param pos: list
    :return: the updated parameters
    '''
    # convert
    words.append(START_WORD)
    words.append(END_WORD)
    words = set(words)
    # append start and end
    pos.insert(0, START_STATE)
    pos.append(END_STATE)
    # add words and states
    for i in range(len(data)):
        data[i][1].insert(0, START_WORD)
        data[i][1].append(END_WORD)
        data[i][0].insert(0, START_STATE)
        data[i][0].append(END_STATE)
    # get rare words
    words_dict = {word: 0 for word in words}
    for i in range(len(data)):
        for word in data[i][1]:
            words_dict[word] += 1
    rare_words_set = {word for (word, count) in words_dict.items() if count <= 2}
    words = words - rare_words_set
    for i in range(len(data)):
        for word_index, word in enumerate(data[i][1]):
            if word not in words:
                data[i][1][word_index] = RARE_WORD

    return data, words, pos


def get_word_appearance_rate(X, words, count_total_amount=False):
    words_appearance_rate = {word: 0 for word in words}
    total_amount_of_words = 0
    for sentence in X:
        if count_total_amount:
            total_amount_of_words += len(sentence)
        for word in sentence:
            words_appearance_rate[word] += 1
    if count_total_amount:
        return words_appearance_rate, total_amount_of_words
    else:
        return words_appearance_rate


if __name__ == '__main__':
    # for MEMM tests:

    data, words, pos = get_data()
    data, words, pos = preprocess_data(data, words, pos)
    # get X and y
    X = [data[i][1] for i in range(len(data))]
    y = [data[i][0] for i in range(len(data))]
    # split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # TESTS -- sample in HMM
    small_train_samples = len(x_train)
    small_test_samples = 10
    small_set = [(y_train[i], x_train[i]) for i in range(small_train_samples)]
    small_x_test = x_test[:small_test_samples]
    small_y_test = y_test[:small_test_samples]
    words_in_small_set = {word for i in range(len(small_set)) for word in small_set[i][1]}
    hmm_model = HMM(pos, words_in_small_set, small_set)
    sampled_sentences = hmm_model.sample(5)
    for sentence in sampled_sentences:
        print(sentence)

    # # TESTS -- MEMM
    # start = time.time()
    # small_train_samples = len(x_train)
    # small_test_samples = len(x_train)
    # small_set = [(y_train[i], x_train[i]) for i in range(small_train_samples)]
    # small_x_test = x_test[:small_test_samples]
    # small_y_test = y_test[:small_test_samples]
    # words_in_small_set = {word for i in range(len(small_set)) for word in small_set[i][1]}
    #
    # memm_model = MEMM(pos, words_in_small_set, small_set, MEMM.phi_func)
    # size_of_transitions_and_emissions = memm_model.pos_size * memm_model.pos_size + \
    #                                     memm_model.words_size * memm_model.pos_size
    # w0 = np.zeros(size_of_transitions_and_emissions)  # init w0
    # w = perceptron(small_set, memm_model, w0)
    # print("finished perceptron")
    # pos_predictions = memm_model.viterbi(small_x_test, w)
    #
    # avg_acc = 0
    # for sample_index in range(len(small_y_test)):
    #     acc = 0
    #     for pos_index in range(len(small_y_test[sample_index])):
    #         if small_y_test[sample_index][pos_index] == pos_predictions[sample_index][pos_index]:
    #             acc += 1
    #     avg_acc += acc / len(small_y_test[sample_index])
    # print("accuracy:" + str(avg_acc / len(small_y_test)))
    # end = time.time()
    # print("total time for memm on 100 test samples: %0.4f seconds" % (end - start))

    # # TESTS -- HMM
    # small_train_samples = len(x_train)
    # small_test_samples = int(len(x_train) / 9)
    # small_set = [(y_train[i], x_train[i]) for i in range(small_train_samples)]
    # small_x_test = x_test[:small_test_samples]
    # small_y_test = y_test[:small_test_samples]
    # words_in_small_set = {word for i in range(len(small_set)) for word in small_set[i][1]}
    # hmm_model = HMM(pos, words_in_small_set, small_set)
    # pos_predictions = hmm_model.viterbi(small_x_test)
    #
    # avg_acc = 0
    # for sample_index in range(len(small_y_test)):
    #     acc = 0
    #     for pos_index in range(len(small_y_test[sample_index])):
    #         if small_y_test[sample_index][pos_index] == pos_predictions[sample_index][pos_index]:
    #             acc += 1
    #     avg_acc += acc / len(small_y_test[sample_index])
    # print("accuracy:" + str(avg_acc / len(small_y_test)))

    # # TESTS -- BASELINE
    # small_train_samples = len(x_train)
    # small_test_samples = int(len(x_test))
    # small_set = [(y_train[i], x_train[i]) for i in range(small_train_samples)]
    # small_x_test = x_test[:small_test_samples]
    # small_y_test = y_test[:small_test_samples]
    # words_in_small_set = {word for i in range(len(small_set)) for word in small_set[i][1]}
    #
    # baseline_model = Baseline(pos, words_in_small_set, small_set)
    # pos_predictions = baseline_model.MAP(small_x_test)
    #
    # avg_acc = 0
    # for sample_index in range(len(small_y_test)):
    #     acc = 0
    #     for pos_index in range(len(small_y_test[sample_index])):
    #         if small_y_test[sample_index][pos_index] == pos_predictions[sample_index][pos_index]:
    #             acc += 1
    #     avg_acc += acc / len(small_y_test[sample_index])
    # print("accuracy:" + str(avg_acc / len(small_y_test)))
    # print("finished")
