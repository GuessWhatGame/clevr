import json

import collections

from generic.data_provider.dataset import AbstractDataset
import os

use_100 = False


class Picture:
    def __init__(self, id, filename, image_builder):
        self.id = id
        self.filename = filename

        if image_builder is not None:
            self.image_loader = image_builder.build(id, filename=filename)

    def get_image(self, **kwargs):
        return self.image_loader.get_image(**kwargs)


class Game(object):
    def __init__(self, id, picture, question, answer, question_family_index):
        self.id = id
        self.picture = picture
        self.question = question
        self.answer = answer
        self.question_family_index = question_family_index

    def __str__(self):
        return "[#q:{}, #p:{}] {} - {} ({})".format(self.id, self.picture.id, self.question, self.answer, self.question_family_index)


class CLEVRDataset(AbstractDataset):
    """Loads the dataset."""

    def __init__(self, folder, which_set, image_builder=None):

        question_file_path = '{}/questions/CLEVR_{}_questions.json'.format(folder, which_set)

        games = []
        self.question_family_index = collections.Counter()
        self.answer_counter = collections.Counter()

        with open(question_file_path) as question_file:
            print("Loading questions...")
            data = json.load(question_file)
            info = data["info"]
            samples = data["questions"]

            assert info["split"] == which_set

            print("Successfully Loaded CLEVR v{} ({})".format(info["version"], which_set))

            for sample in samples:

                question_id = int(sample["question_index"])
                question = sample["question"]
                question_family_index = sample.get("question_family_index", -1)  # -1 for test set

                answer = sample.get("answer", None)  # None for test set

                image_id = sample["image_index"]
                image_filename = sample["image_filename"]
                image_filename = os.path.join(which_set, image_filename)

                games.append(Game(id=question_id,
                                  picture=Picture(image_id, image_filename, image_builder),
                                  question=question,
                                  answer=answer,
                                  question_family_index=question_family_index))

                self.question_family_index[question_family_index] += 1
                self.answer_counter[answer] += 1

                if use_100 and len(games) > 100:
                    break

        print('{} games loaded...'.format(len(games)))
        super(CLEVRDataset, self).__init__(games)




if __name__ == '__main__':
    dataset = CLEVRDataset("/home/fstrub/Projects/clevr_data/", which_set="val")

    for d in dataset.games:
        if "How many things are" in d.question:
            print(d)