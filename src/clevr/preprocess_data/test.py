from multiprocessing.pool import ThreadPool
from tqdm import tqdm

from generic.data_provider.image_loader import h5FeatureBuilder
from generic.data_provider.iterator import Iterator
from generic.data_provider.nlp_utils import DummyTokenizer

from clevr.data_provider.clevr_dataset import CLEVRDataset
from clevr.data_provider.clevr_batchifier import CLEVRBatchifier

if __name__ == "__main__":

    feat_dir = "/media/datas2/tmp"
    data_dir = "/home/sequel/fstrub/clevr_data"

    image_builder = h5FeatureBuilder(img_dir=feat_dir, bufferize=False)

    print("Load datasets...")
    dataset = CLEVRDataset(folder=data_dir, which_set="val", image_builder=image_builder)

    cpu_pool = ThreadPool(1)

    dummy_tokenizer = DummyTokenizer()

    batchifier = CLEVRBatchifier(tokenizer=dummy_tokenizer, sources=["image"])
    iterator = Iterator(dataset,
                        batch_size=64,
                        pool=cpu_pool,
                        batchifier=batchifier)

    for batch in tqdm(iterator):
        pass

    print("Done!")
