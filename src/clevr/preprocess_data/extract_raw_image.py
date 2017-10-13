#!/usr/bin/env python
import numpy
import os
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from distutils.util import strtobool
import numpy as np
import argparse

import h5py


from generic.data_provider.image_loader import RawImageBuilder
from generic.data_provider.iterator import Iterator
from generic.data_provider.nlp_utils import DummyTokenizer

from clevr.data_provider.clevr_dataset import CLEVRDataset
from clevr.data_provider.clevr_batchifier import CLEVRBatchifier



parser = argparse.ArgumentParser('Feature extractor! ')

parser.add_argument("-img_dir", type=str, required=True, help="Input Image folder")
parser.add_argument("-data_dir", type=str, required=True,help="Dataset folder")
parser.add_argument("-out_dir", type=str, required=True, help="Output directory for h5 files")
parser.add_argument("-set_type", type=list, default=["val", "train", "test"], help='Select the dataset to dump')

parser.add_argument("-subtract_mean", type=lambda x:bool(strtobool(x)), default="True", help="Preprocess the image by substracting the mean")
parser.add_argument("-img_size", type=int, default=224, help="image size (pixels)")

parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
parser.add_argument("-no_thread", type=int, default=2, help="No thread to load batch")



args = parser.parse_args()



# define image properties
if args.subtract_mean:
    channel_mean = np.array([123.68, 116.779, 103.939])
else:
    channel_mean = None

source = 'image'
image_builder = RawImageBuilder(args.img_dir,
                                height=args.img_size,
                                width=args.img_size,
                                channel=channel_mean)
image_shape=[args.img_size, args.img_size, 3]

# CPU/GPU option
cpu_pool = ThreadPool(args.no_thread)

for one_set in args.set_type:

    ############################
    #   LOAD DATASET
    ############################

    print("Load dataset...")
    dataset = CLEVRDataset(args.data_dir, one_set, image_builder=image_builder)

    # hack dataset to only keep one game by image
    image_id_set = {}
    games = []
    for game in dataset.games:
        if game.image.id not in image_id_set:
            games.append(game)
            image_id_set[game.image.id] = 1

    dataset.games = games
    no_images = len(games)

    # prepare batch builder
    dummy_tokenizer = DummyTokenizer()
    batchifier = CLEVRBatchifier(tokenizer=dummy_tokenizer, sources=[source])
    iterator = Iterator(dataset,
                        batch_size=args.batch_size,
                        pool=cpu_pool,
                        batchifier=batchifier)

    filepath = os.path.join(args.out_dir, "{}_features.h5".format(one_set))
    with h5py.File(args.out_file, 'w') as f:

        feat_dataset = f.create_dataset('features', shape=[no_images] + image_shape, dtype=np.float32)
        idx2img = f.create_dataset('idx2img', shape=[no_images], dtype=np.int32)
        pt_hd5 = 0

        for batch in tqdm(iterator):

            # Store dataset
            batch_size = len(batch["raw"])
            feat_dataset[pt_hd5: pt_hd5 + batch_size] = batch["image"].astype(np.float32)

            # Store idx to image.id
            for i, game in enumerate(batch["raw"]):
                idx2img[pt_hd5 + i] = game.image.id

            # update hd5 pointer
            pt_hd5 += batch_size

        print("Start dumping file: {}".format(filepath))
    print("Finished dumping file: {}".format(filepath))

print("Done!")
