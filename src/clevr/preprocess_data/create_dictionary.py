from nltk.tokenize import TweetTokenizer
import io
import json
import collections
from clevr.data_provider.clevr_dataset import CLEVRDataset
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Creating dictionary..')

    parser.add_argument("-data_dir", type=str, help="Path to VQA dataset")
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Name of the dictionary file")
    parser.add_argument("-min_occ", type=int, default=1, help='Minimum number of occurences to add word to dictionary (for Human Clevr)')
    args = parser.parse_args()

    dataset = CLEVRDataset(args.data_dir, which_set="train")
    games = dataset.games

    word2i = {'<padding>': 0,
              '<start>': 1,
              '<end>': 2,
              '<unk>': 3
              }

    answer2i = {'<padding>': 0,
                '<start>': 1,
                '<end>': 2,
                '<unk>': 3
                }

    answer2occ = dataset.answer_counter
    word2occ = collections.defaultdict(int)


    # Input words
    tknzr = TweetTokenizer(preserve_case=False)

    for game in games:
        input_tokens = tknzr.tokenize(game.question)
        for tok in input_tokens:
            word2occ[tok] += 1

    # parse the questions
    for word, occ in word2occ.items():
        if occ >= args.min_occ:
            word2i[word] = len(word2i)

    # parse the answers
    for answer in answer2occ.keys():
        answer2i[answer] = len(answer2i)

    print("Number of words): {}".format(len(word2i)))
    print("Number of answers: {}".format(len(answer2i)))

    with io.open(args.dict_file, 'w', encoding='utf8') as f_out:
       data = json.dumps({'word2i': word2i, 'answer2i': answer2i})
       f_out.write(data)
