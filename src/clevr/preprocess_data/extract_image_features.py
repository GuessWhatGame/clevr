#!/usr/bin/env python
import numpy
import os
import tensorflow as tf
from multiprocessing import Pool
from tqdm import tqdm
from distutils.util import strtobool
import numpy as np
import argparse

import h5py

import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1
import tensorflow.contrib.slim.python.slim.nets.resnet_utils as slim_utils

from generic.data_provider.image_loader import RawImageBuilder
from generic.data_provider.nlp_utils import DummyTokenizer
from generic.data_provider.iterator import Iterator


from clevr.data_provider.clevr_dataset import CLEVRDataset
from clevr.data_provider.clevr_batchifier import CLEVRBatchifier



parser = argparse.ArgumentParser('Feature extractor! ')

parser.add_argument("-img_dir", type=str, required=True, help="Input Image folder")
parser.add_argument("-data_dir", type=str, required=True,help="Dataset folder")
parser.add_argument("-out_dir", type=str, required=True, help="Output directory for h5 files")
parser.add_argument("-set_type", type=list, default=["val", "train", "test"], help='Select the dataset to dump')


parser.add_argument("-ckpt", type=str, required=True, help="Path for network checkpoint: ")
parser.add_argument("-feature_name", type=str, default="block3/unit_22/bottleneck_v1", help="Pick the name of the network features")

parser.add_argument("-subtract_mean", type=lambda x:bool(strtobool(x)), default="True", help="Preprocess the image by substracting the mean")
parser.add_argument("-img_size", type=int, default=224, help="image size (pixels)")
parser.add_argument("-batch_size", type=int, default=64, help="Batch size to extract features")

parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
parser.add_argument("-no_thread", type=int, default=2, help="No thread to load batch")

args = parser.parse_args()


# define image
if args.subtract_mean:
    channel_mean = np.array([123.68, 116.779, 103.939])
else:
    channel_mean = None


# define the image loader
source = 'image'
images = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, 3], name=source)
image_builder = RawImageBuilder(args.img_dir,
                                height=args.img_size,
                                width=args.img_size,
                                channel=channel_mean)



# create network
print("Create network...")
with slim.arg_scope(slim_utils.resnet_arg_scope(is_training=False)):
    _, end_points = resnet_v1.resnet_v1_101(images, 1000)  # 1000 is the number of softmax class

    feature_name = os.path.join("resnet_v1_101", args.feature_name) # define the feature name according slim standard

    ft_output = end_points[feature_name]
    ft_shape = [int(dim) for dim in ft_output.get_shape()[1:]]


#Create a dummy tokenizer for the batchifier
dummy_tokenizer = DummyTokenizer()


# CPU/GPU option
cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

    saver = tf.train.Saver()
    saver.restore(sess, args.ckpt)

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
        batchifier = CLEVRBatchifier(tokenizer=dummy_tokenizer, sources=[source])
        iterator = Iterator(dataset,
                            batch_size=args.batch_size,
                            pool=cpu_pool,
                            batchifier=batchifier,
                            shuffle=False)

        ############################
        #  CREATE FEATURES
        ############################
        print("Start computing image features...")
        filepath = os.path.join(args.out_dir, "{}_features.h5".format(one_set))
        with h5py.File(filepath, 'w') as f:

            feat_dataset = f.create_dataset('features', shape=[no_images] + ft_shape, dtype=np.float32)
            idx2img = f.create_dataset('idx2img', shape=[no_images], dtype=np.int32)
            pt_hd5 = 0

            for batch in tqdm(iterator):
                feat = sess.run(ft_output, feed_dict={images: numpy.array(batch[source])})

                # Store dataset
                batch_size = len(batch["raw"])
                feat_dataset[pt_hd5: pt_hd5 + batch_size] = feat

                # Store idx to image.id
                for i, game in enumerate(batch["raw"]):
                    idx2img[pt_hd5 + i] = game.image.id

                # update hd5 pointer
                pt_hd5 += batch_size
            print("Start dumping file: {}".format(filepath))
        print("Finished dumping file: {}".format(filepath))

print("Done!")
