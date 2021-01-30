import argparse
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from transformers.modeling_bert import BertConfig
from pretrained.tokenization import BertTokenizer as ETRITokenizer
from gluonnlp.data import SentencepieceTokenizer
from model.net import SentenceClassifier
from model.data import Corpus
from model.utils import PreProcessor, PadSequence
from model.metric2 import evaluate, acc
from utils import Config, CheckpointManager, SummaryManager
from IPython import embed

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing config.json of data")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing config.json of model")
parser.add_argument('--dataset', default='test', help="name of the data in --data_dir to be evaluate")
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

    # model (restore)
    checkpoint_manager = CheckpointManager(model_dir)
    checkpoint = checkpoint_manager.load_checkpoint('best_{}.tar'.format(args.type))
    config = BertConfig(ptr_config.config)
    model = SentenceClassifier(config, num_classes=model_config.num_classes, vocab=preprocessor.vocab)
    model.load_state_dict(checkpoint['model_state_dict'])

    # evaluation
    filepath = getattr(data_config, args.dataset)
    ds = Corpus(filepath, preprocessor.preprocess)
    dl = DataLoader(ds, batch_size=model_config.batch_size, num_workers=4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    summary_manager = SummaryManager(model_dir)
    res = evaluate(model, dl, {'loss': nn.CrossEntropyLoss(), 'acc': acc}, device)
    df_test = pd.read_csv('data/public_test.csv')	
    df_new = pd.DataFrame({'label':res})	
    df_new['id'] = df_test.id	
    df_new['smishing'] = df_new.label	
    df_new.to_csv('submission.csv', index=False)	
    df_for_us = pd.DataFrame({'id':df_new.id, 'text':df_test.text, 'smishing':df_new.smishing, 'binary_smishing':(df_new.smishing > 0.5).astype('int')})	
    df_for_us.to_csv('submission_for_us.csv', index=False, encoding='cp949', sep='\t')    
