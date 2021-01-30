import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from transformers.modeling_bert import BertConfig
from pretrained.tokenization import BertTokenizer as ETRITokenizer
from gluonnlp.data import SentencepieceTokenizer
from model.net import SentenceClassifier
from model.data1 import Corpus
from model.utils1-1 import PreProcessor, PadSequence
from model.metric import evaluate, acc
from utils import Config, CheckpointManager, SummaryManager
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# for reproducibility
torch.manual_seed(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing config.json of data")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing config.json of model")
parser.add_argument('--type', default='skt', choices=['skt', 'etri'], required=True,  type=str)


if __name__ == '__main__':
    args = parser.parse_args()
    ptr_dir = Path('pretrained')
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    ptr_config = Config(ptr_dir / 'config_{}.json'.format(args.type))
    data_config = Config(data_dir / 'config.json')
    model_config = Config(model_dir / 'config.json')

    # vocab
    with open(ptr_config.vocab, mode='rb') as io:
        vocab = pickle.load(io)

    # tokenizer
    if args.type == 'etri':
        ptr_tokenizer = ETRITokenizer.from_pretrained(ptr_config.tokenizer, do_lower_case=False)
        pad_sequence = PadSequence(length=model_config.length, pad_val=vocab.to_indices(vocab.padding_token))
        preprocessor = PreProcessor(vocab=vocab, split_fn=ptr_tokenizer.tokenize, pad_fn=pad_sequence)
    elif args.type == 'skt':
        ptr_tokenizer = SentencepieceTokenizer(ptr_config.tokenizer)
        pad_sequence = PadSequence(length=model_config.length, pad_val=vocab.to_indices(vocab.padding_token))
        preprocessor = PreProcessor(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=pad_sequence)

    # model
    config = BertConfig(ptr_config.config)
    model = SentenceClassifier(config, num_classes=model_config.num_classes, vocab=preprocessor.vocab)
    bert_pretrained = torch.load(ptr_config.bert)
    model.load_state_dict(bert_pretrained, strict=False)

    # training
    tr_ds = Corpus("data/my_train.txt", preprocessor.preprocess)