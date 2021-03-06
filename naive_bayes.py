import os
import sys

from functools import reduce
from collections import defaultdict

# Spacy imports
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

nlp = English()
tokenizer = Tokenizer(nlp.vocab)

class Dataset():
    def __init__(self, path_to_dataset):
        self.path_to_dataset = path_to_dataset
        self.count_of_words = {'pos': defaultdict(int), 'neg': defaultdict(int)}

    def update_dataset(self, new_path):
        self.path_to_dataset = new_path

    def count_words_from_corpus(self):
        self.total_files = {label: 0 for label in self.path_to_dataset.keys()}

        # Read in binary and decode in ascii
        for label, path_to_label_folder in self.path_to_dataset.items():
            for txt_file in os.listdir(path_to_label_folder):
                self.total_files[label] += 1
                for line in Dataset.load_dataset_line_by_line(os.path.join(path_to_label_folder, txt_file)):
                    try:
                        tokens = Dataset.tokenize(line.decode('ascii').strip())
                    except:
                        continue
                    for token in tokens:
                        self.count_of_words[label][token] += 1

        self.total_words = {label: sum(self.count_of_words[label].values()) for label in self.count_of_words.keys()}

    @staticmethod
    def tokenize(sentence):
        # Remove punctuations
        # punct = ['.', ',', '!', '?']
        # tokens = [token for token in sentence.split(' ') if token not in punct]
        tokens = tokenizer(sentence)
        return tokens

    @staticmethod
    def load_dataset_line_by_line(path_to_dataset):
        # Read in data line by line and return as a generator
        with open(path_to_dataset, 'rb') as f:
            for line in f:
                yield line

def test_model(model, path_to_dataset):
    results = {'correct': 0, 'incorrect': 0}

    # Test the model
    for label, path_to_label_folder in path_to_dataset.items():
        for txt_file in os.listdir(path_to_label_folder):
            for line in Dataset.load_dataset_line_by_line(os.path.join(path_to_label_folder, txt_file)):
                try:
                    tokens = Dataset.tokenize(line.decode('ascii').strip())
                except:
                    continue

                # Calculate P(words) = P(words | pos) + P(words | neg)
                # Likelihood
                evidence_pos = [(model.count_of_words['pos'][token]+1)/(model.total_words['pos'] + (model.total_words['pos'] + model.total_words['neg'])) for token in tokens]
                evidence_neg = [(model.count_of_words['neg'][token]+1)/(model.total_words['neg'] + (model.total_words['pos'] + model.total_words['neg'])) for token in tokens]
                evidence = reduce(lambda x, y: x*y, evidence_pos) + reduce(lambda x, y: x*y, evidence_neg)

                p_pos = (model.total_files['pos'] * reduce(lambda x, y: x*y, evidence_pos)) / evidence
                p_neg = (model.total_files['neg'] * reduce(lambda x, y: x*y, evidence_neg)) / evidence

                if (p_pos > p_neg) and ('pos' == label) or (p_neg > p_pos) and ('neg' == label):
                    print(f'Label: {label}')
                    print(f'Sentence: {line}')
                    results['correct'] += 1
                else:
                    results['incorrect'] += 1

    return results

def main():
    base_path = os.path.join('datasets')
    dataset_path = os.path.join(base_path, 'txt_sentoken')

    path_to_dataset = {'pos': os.path.join(dataset_path, 'pos'), 'neg': os.path.join(dataset_path, 'neg')}

    # Train Naive Bayes model
    dataset = Dataset(path_to_dataset)
    dataset.count_words_from_corpus()

    # Test the trained model
    path_to_test_dataset = {'pos': os.path.join(base_path, 'rt-polaritydata', 'test_pos'), 'neg': os.path.join(base_path, 'rt-polaritydata', 'test_neg')}
    results = test_model(dataset, path_to_test_dataset)

    print(results)
    print(results['correct']/(sum(results.values())))
    print(len(dataset.count_of_words['pos'].values()))
    print(len(dataset.count_of_words['neg'].values()))



if __name__ == '__main__':
    main()
