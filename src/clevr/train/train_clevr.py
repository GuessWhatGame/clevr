import argparse
import logging
import os
import tensorflow as tf
from distutils.util import strtobool

from multiprocessing import Pool

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.tf_utils.ckpt_loader import load_checkpoint
from generic.utils.config import load_config
from generic.utils.file_handlers import pickle_dump
from generic.data_provider.image_loader import get_img_builder

from clevr.data_provider.clevr_tokenizer import CLEVRTokenizer
from clevr.data_provider.clevr_dataset import CLEVRDataset
from clevr.data_provider.clevr_batchifier import CLEVRBatchifier
from clevr.models.clevr_film_network import FiLMNetwork


###############################
#  LOAD CONFIG
#############################

parser = argparse.ArgumentParser('CLEVR network baseline!')

parser.add_argument("-data_dir", type=str, help="Directory with data")
parser.add_argument("-img_dir", type=str, help="Directory with image")
parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
parser.add_argument("-config", type=str, help='Config file')
parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
parser.add_argument("-continue_exp", type=lambda x:bool(strtobool(x)), default="False", help="Continue previously started experiment?")
parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")
parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How many GPU ram is required? (ratio)")


args = parser.parse_args()

config, exp_identifier, save_path = load_config(args.config, args.exp_dir)
logger = logging.getLogger()


# Load config
resnet_version = config['model'].get('resnet_version', 50)
finetune = config["model"]["image"].get('finetune', list())
lrt = config['optimizer']['learning_rate']
batch_size = config['optimizer']['batch_size']
clip_val = config['optimizer']['clip_val']
no_epoch = config["optimizer"]["no_epoch"]
merge_dataset = config.get("merge_dataset", False)


# Load images
logger.info('Loading images..')
image_loader = get_img_builder(config['model']['image'], args.img_dir)
use_resnet = image_loader.is_raw_image()


# Load dictionary
logger.info('Loading dictionary..')
tokenizer = CLEVRTokenizer(os.path.join(args.data_dir, config["dico_name"]))

# Load data
logger.info('Loading data..')
trainset = CLEVRDataset(args.data_dir, which_set="train", image_builder=image_loader)
validset = CLEVRDataset(args.data_dir, which_set="val", image_builder=image_loader)
testset = CLEVRDataset(args.data_dir, which_set="test", image_builder=image_loader)



# Build Network
logger.info('Building network..')
network = FiLMNetwork(config=config["model"],
                       no_words=tokenizer.no_words,
                       no_answers=tokenizer.no_answers)

# Build Optimizer
logger.info('Building optimizer..')
optimizer, loss = create_optimizer(network, network.loss_decay, config, finetune=finetune)
outputs = [loss, network.accuracy]


###############################
#  START  TRAINING
#############################

# create a saver to store/load checkpoint
saver = tf.train.Saver()
resnet_saver = None

# Retrieve only resnet variabes
if use_resnet:
    start = len(network.scope_name)+1
    resnet_vars = {v.name[start:-2]: v for v in network.get_resnet_parameters()}
    resnet_saver = tf.train.Saver(resnet_vars)


# CPU/GPU option
cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

    # retrieve incoming sources
    sources = network.get_sources(sess)
    logger.info("Sources: " + ', '.join(sources))


    # Load checkpoints or pre-trained networks
    sess.run(tf.global_variables_initializer())
    start_epoch = load_checkpoint(sess, saver, args, save_path)
    if use_resnet:
        resnet_saver.restore(sess, os.path.join(args.data_dir,'resnet_v1_{}.ckpt'.format(resnet_version)))


    # Create evaluation tools
    evaluator = Evaluator(sources, network.scope_name, network=network, tokenizer=tokenizer)
    train_batchifier = CLEVRBatchifier(tokenizer, sources, optim_param=config["config"])
    eval_batchifier = CLEVRBatchifier(tokenizer, sources)


    # start actual training
    best_val_acc, best_train_acc = 0, 0
    for t in range(start_epoch, no_epoch):

        logger.info('Epoch {}/{}..'.format(t + 1,no_epoch))

        train_iterator = Iterator(trainset,
                                  batch_size=batch_size,
                                  batchifier=train_batchifier,
                                  shuffle=True,
                                  pool=cpu_pool)
        [train_loss, train_accuracy] = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer])


        valid_loss, valid_accuracy = 0,0
        if not merge_dataset:
            valid_iterator = Iterator(validset,
                                      batch_size=batch_size*2,
                                      batchifier=eval_batchifier,
                                      shuffle=True,
                                      pool=cpu_pool)

            [valid_loss, valid_accuracy] = evaluator.process(sess, valid_iterator, outputs=outputs)

        logger.info("Training loss: {}".format(train_loss))
        logger.info("Training accuracy: {}".format(train_accuracy))
        logger.info("Validation loss: {}".format(valid_loss))
        logger.info("Validation accuracy: {}".format(valid_accuracy))

        if valid_accuracy >= best_val_acc:
            best_train_acc = train_accuracy
            best_val_acc = valid_accuracy
            saver.save(sess, save_path.format('params.ckpt'))
            logger.info("checkpoint saved...")

            pickle_dump({'epoch': t}, save_path.format('status.pkl'))

    # Dump test file to upload on VQA website
    # logger.info("Compute final {} results...".format(args.test_set))

    # vqa_file_name = "vqa_OpenEnded_mscoco_{}{}_cbn_results.json".format(args.test_set, args.year, config["model"]["name"])
    # dumper_eval_listener = VQADumperListener(tokenizer, os.path.join(args.exp_dir, save_path.format(vqa_file_name)),
    #                                               require=network.prediction)
    #
    # saver.restore(sess, save_path.format('params.ckpt'))
    # test_iterator = Iterator(testset,
    #                          batch_size=batch_size*2,
    #                          batchifier=eval_batchifier,
    #                          shuffle=False,
    #                          pool=cpu_pool)
    # evaluator.process(sess, test_iterator, outputs=[], listener=dumper_eval_listener)
    # logger.info("File dump at {}".format(dumper_eval_listener.out_path))