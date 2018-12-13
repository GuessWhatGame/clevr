import collections
import numpy as np
from generic.data_provider.nlp_utils import padder
from generic.data_provider.batchifier import AbstractBatchifier


class CLEVRBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply(self, games):

        batch = collections.defaultdict(list)
        batch_size = len(games)

        assert batch_size > 0

        for i, game in enumerate(games):

            batch["raw"].append(game)

            # Get question
            question = self.tokenizer.encode_question(game.question)
            batch['question'].append(question)

            # Get answers
            answer = self.tokenizer.encode_answer(game.answer)
            batch['answer'].append(answer)

            # retrieve the image source type
            img = game.image.get_image()
            if "image" not in batch:  # initialize an empty array for better memory consumption
                batch["image"] = np.zeros((batch_size,) + img.shape, dtype=np.float32)
            batch["image"][i] = img

        # pad the questions
        batch['question'], batch['seq_length'] = padder(batch['question'],
                                                        padding_symbol=self.tokenizer.padding_token)

        return batch
