import argparse
import os
import sys
import subprocess
import time
import pickle
import numpy as np
import torch
from torch.nn import functional as F

from src.data.dictionary import Dictionary
from src.data.dataset import ParallelDataset
from src.data.loader import load_binarized
from src.model import build_model
from src.utils import truncate, concat_batches, to_cuda

TEST_REVERSE = False

def load_data(dir_data, params):
    """
    Load parallel data.
    """
    files = os.listdir(dir_data)
    lang_pairs = set()
    for f in files:
        if f[-4:] == ".pth":
            f = f[:-4]
        langs, _ = f.split(".")
        src, tgt = langs.split("-")
        lang_pairs.add((src,tgt))
    data = {}
    data["para"] = {}
    for src, tgt in lang_pairs:
        path_src = os.path.join(dir_data, "{}-{}.{}.pth".format(src, tgt, src))
        path_tgt = os.path.join(dir_data, "{}-{}.{}.pth".format(src, tgt, tgt))
        assert os.path.isfile(path_src)
        assert os.path.isfile(path_tgt)

        # load binarized datasets
        src_data = load_binarized(path_src, params)
        tgt_data = load_binarized(path_tgt, params)

        # create ParallelDataset
        dataset = ParallelDataset(
            src_data['sentences'], src_data['positions'],
            tgt_data['sentences'], tgt_data['positions'],
            params, labels=None
        )
        data["para"][(src, tgt)] = dataset
    return data

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Predict class of sentence pairs")

    # main parameters
    parser.add_argument("--load_model", type=str, required=True,
                        help="Model to load")
    parser.add_argument("--dump_path", type=str, required=True,
                        help="Experiment dump path")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Data path")
    parser.add_argument("--log_file", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of sentences per batch")
    return parser

def main(args):
    rng = np.random.RandomState(0)

    # Make dump path
    if not os.path.exists(args.dump_path):
        subprocess.Popen("mkdir -p %s" % args.dump_path, shell=True).wait()
    else:
        if os.listdir(args.dump_path):
            m = "Directory {} is not empty.".format(args.dump_path)
            raise ValueError(m)
    if len(args.log_file) and os.path.isfile(args.log_file):
        write_log = True
    else:
        write_log = False

    # load model parameters
    model_dir = os.path.dirname(args.load_model)
    params_path = os.path.join(model_dir, 'params.pkl')
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    
    # load data parameters and model parameters from checkpoint
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pth')
    assert os.path.isfile(checkpoint_path)
    data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
    for k,v in data["params"].items():
        params.__dict__[k] = v
    dico = Dictionary(data["dico_id2word"], data["dico_word2id"], data["dico_counts"])

    # Print score
    for k,v in data["best_metrics"].items():
        print("- {}: {}".format(k,v))

    # Fix some of the params we pass to load_data
    params.debug_train = False
    params.max_vocab = -1
    params.min_count = 0
    params.tokens_per_batch = -1
    params.max_batch_size = args.batch_size
    params.batch_size = args.batch_size

    # load data
    data = load_data(args.data_path, params)

    # Print data summary
    for (src, tgt), dataset in data['para'].items():
        datatype = "Para data (%s)" % ("WITHOUT labels" if dataset.labels is None else "WITH labels")
        m = '{: <27} - {: >12}:{: >10}'.format(datatype, '%s-%s' % (src, tgt), len(dataset))
        print(m)

    # Fix some of the params we pass to the model builder
    params.reload_model = args.load_model

    # build model
    if params.encoder_only:
        model = build_model(params, dico)
    else:
        encoder, decoder = build_model(params, dico)
        model = encoder

    # Predict
    model = model.module if params.multi_gpu else model
    model.eval()
    start = time.time()
    for (src, tgt), dataset in data['para'].items():
        path = os.path.join(args.dump_path, "{}-{}.pred".format(src, tgt))
        scores_file = open(path, "w")
        lang1_id = params.lang2id[src]
        lang2_id = params.lang2id[tgt]
        diffs = []
        nb_written = 0
        for batch in dataset.get_iterator(False, group_by_size=False, n_sentences=-1, return_indices=False):
            (sent1, len1), (sent2, len2), labels = batch
            sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
            sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)
            x, lengths, positions, langs = concat_batches(sent1, len1, lang1_id, sent2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=True)
            x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)
            with torch.no_grad():
                # Get sentence pair embedding
                h = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)[0]
                CLF_ID1, CLF_ID2 = 8, 9  # very hacky, use embeddings to make weights for the classifier
                emb = (model.module if params.multi_gpu else model).embeddings.weight
                pred = F.linear(h, emb[CLF_ID1].unsqueeze(0), emb[CLF_ID2, 0])
                pred = torch.sigmoid(pred)
                pred = pred.view(-1).cpu().numpy().tolist()
            for p in pred:
                scores_file.write("{:.8f}\n".format(p))
            nb_written += len(pred)
            if nb_written % 10000 == 0:
                elapsed = int(time.time() - start)
                lpss = elapsed % 60
                lpsm = elapsed // 60
                lpsh = lpsm // 60
                lpsm = lpsm % 60
                msg = "[{:02d}:{:02d}:{:02d} {}-{}]"
                msg += " {}/{} ({:.2f}%) sentences processed".format(lpsh, lpsm, lpss, src, tgt, nb_written, len(dataset), 100*nb_written/len(dataset))
                print(msg)
                if write_log:
                    with open(args.log_file, "a") as fout:
                        fout.write(msg + "\n")
            # Try reversing order
            if TEST_REVERSE:
                x, lengths, positions, langs = concat_batches(sent2, len2, lang2_id, sent1, len1, lang1_id, params.pad_index, params.eos_index, reset_positions=True)
                x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)
                with torch.no_grad():
                    # Get sentence pair embedding
                    h = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)[0]
                    CLF_ID1, CLF_ID2 = 8, 9  # very hacky, use embeddings to make weights for the classifier
                    emb = (model.module if params.multi_gpu else model).embeddings.weight
                    pred_rev = F.linear(h, emb[CLF_ID1].unsqueeze(0), emb[CLF_ID2, 0])
                    pred_rev = torch.sigmoid(pred_rev)
                    pred_rev = pred_rev.view(-1).cpu().numpy().tolist()
                    for p, pp in zip(pred, pred_rev):
                        diffs.append(p-pp)

        if TEST_REVERSE:
            print("Average absolute diff between score(l1,l2) and score(l2,l1): {}".format(np.mean(np.abs(diffs))))

        scores_file.close()

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    args = parser.parse_args()

    # run prediction
    main(args)
