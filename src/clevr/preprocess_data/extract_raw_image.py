#!/usr/bin/env python
from distutils.util import strtobool
import numpy as np
import argparse

from generic.preprocess_data.extract_img_raw import extract_raw
from generic.data_provider.image_loader import RawImageBuilder

from clevr.data_provider.clevr_dataset import CLEVRDataset
from clevr.data_provider.clevr_batchifier import CLEVRBatchifier



parser = argparse.ArgumentParser('Feature extractor! ')

parser.add_argument("-img_dir", type=str, required=True, help="Input Image folder")
parser.add_argument("-data_dir", type=str, required=True,help="Dataset folder")
parser.add_argument("-out_dir", type=str, required=True, help="Output directory for h5 files")
parser.add_argument("-set_type", type=list, default=["val", "train", "test"], help='Select the dataset to dump')

parser.add_argument("-subtract_mean", type=lambda x:bool(strtobool(x)), default="True", help="Preprocess the image by substracting the mean")
parser.add_argument("-img_size", type=int, required=True, help="image size (pixels)")

parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
parser.add_argument("-no_thread", type=int, default=2, help="No thread to load batch")

args = parser.parse_args()



# define image properties
if args.subtract_mean:
    channel_mean = np.array([123.68, 116.779, 103.939])
else:
    channel_mean = None

source_name = 'image'
image_builder = RawImageBuilder(args.img_dir,
                                height=args.img_size,
                                width=args.img_size,
                                channel=channel_mean)
image_shape=[args.img_size, args.img_size, 3]

extract_raw(
    image_shape=image_shape,
    dataset_cstor=CLEVRDataset,
    dataset_args={"folder": args.data_dir, "year": args.year, "image_builder":image_builder},
    batchifier_cstor=CLEVRBatchifier,
    source_name=source_name,
    out_dir=args.out_dir,
    set_type=args.set_type,
    no_threads=args.no_thread,
)