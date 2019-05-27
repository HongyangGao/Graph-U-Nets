import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append('%s/../s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from embedding import EmbedMeanField, EmbedLoopyBP
from mlp import MLPClassifier

from util import cmd_args, load_data

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        self.s2v = model(latent_dim=cmd_args.latent_dim, 
                        output_dim=cmd_args.out_dim,
                        num_node_feats=cmd_args.feat_dim, 
                        num_edge_feats=0,
                        max_lv=cmd_args.max_lv)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class)

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        concat_feat = []
        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            concat_feat += batch_graph[i].node_tags
        
        concat_feat = torch.LongTensor(concat_feat).view(-1, 1)
        node_feat = torch.zeros(n_nodes, cmd_args.feat_dim)
        node_feat.scatter_(1, concat_feat, 1)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda() 
            labels = labels.cuda()

        return node_feat, labels

    def forward(self, batch_graph): 
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, None)
        
        return self.mlp(embed, labels)

def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        _, loss, acc = classifier(batch_graph)
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()

        loss = loss.data.cpu().numpy()[0]
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )

        total_loss.append( np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    return avg_loss


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    train_graphs, test_graphs = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()
    
    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1]))
        
        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1]))

        # if best_loss is None or test_loss[0] < best_loss:
        #     best_loss = test_loss[0]
        #     print('----saving to best model since this is the best valid loss so far.----')
        #     torch.save(classifier.state_dict(), cmd_args.save_dir + '/epoch-best.model')
        #     save_args(cmd_args.save_dir + '/epoch-best-args.pkl', cmd_args)
