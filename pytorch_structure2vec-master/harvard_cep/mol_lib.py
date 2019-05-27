import ctypes
import numpy as np
import os
import sys
import torch
from tqdm import tqdm

class _mol_lib(object):

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/build/dll/libmol.so' % dir_path)

        # self.lib.Smiles2Graph.restype = ctypes.c_void_p
        self.lib.PrepareBatchFeature.restype = ctypes.c_int
        self.lib.DumpFeatures.restype = ctypes.c_int
        self.lib.LoadMolGraph.restype = ctypes.c_int

        self.lib.NodeFeatDim.restype = ctypes.c_int
        self.lib.EdgeFeatDim.restype = ctypes.c_int
        self.lib.NumNodes.restype = ctypes.c_int
        self.lib.NumEdges.restype = ctypes.c_int
        self.lib.EdgeList.restype = ctypes.c_void_p

        self.num_node_feats = self.lib.NodeFeatDim()
        self.num_edge_feats = self.lib.EdgeFeatDim()

    def PrepareFeatureLabel(self, molgraph_list):
        c_list = (ctypes.c_void_p * len(molgraph_list))()
        total_num_nodes = 0
        total_num_edges = 0
        for i in range(len(molgraph_list)):
            c_list[i] = molgraph_list[i].handle
            total_num_nodes += molgraph_list[i].num_nodes
            total_num_edges += molgraph_list[i].num_edges

        torch_node_feat = torch.zeros(total_num_nodes, self.num_node_feats)
        torch_edge_feat = torch.zeros(total_num_edges * 2, self.num_edge_feats)
        torch_label = torch.zeros(len(molgraph_list), 1)

        node_feat = torch_node_feat.numpy()
        edge_feat = torch_edge_feat.numpy()    
        label = torch_label.numpy()

        self.lib.PrepareBatchFeature(len(molgraph_list), ctypes.cast(c_list, ctypes.c_void_p),
                                    ctypes.c_void_p(node_feat.ctypes.data), 
                                    ctypes.c_void_p(edge_feat.ctypes.data))

        for i in range(len(molgraph_list)):
            label[i] = molgraph_list[i].pce

        return torch_node_feat, torch_edge_feat, torch_label

    def DumpFeatures(self, fname):
        p = ctypes.cast(fname, ctypes.c_char_p)
        self.lib.DumpFeatures(p)

    def LoadMolGraph(self, phase, str_pce_tuples):
        fname = 'data/%s.txt.bin' % phase
        assert os.path.isfile(fname)
        
        fname = ctypes.cast(fname, ctypes.c_char_p)
        num_graphs = len(str_pce_tuples)
        c_list = (ctypes.c_void_p * num_graphs)()
        t = self.lib.LoadMolGraph(fname, ctypes.cast(c_list, ctypes.c_void_p))
        assert t == num_graphs

        molgraph_list = []
        for i in tqdm(range(0, t)):
            g = MolGraph(c_list[i], str_pce_tuples[i][0], str_pce_tuples[i][1])            
            molgraph_list.append(g)

        return molgraph_list
    # def __CtypeNetworkX(self, g):
    #     edges = g.edges()
    #     e_list_from = (ctypes.c_int * len(edges))()
    #     e_list_to = (ctypes.c_int * len(edges))()

    #     if len(edges):
    #         a, b = zip(*edges)
    #         e_list_from[:] = a
    #         e_list_to[:] = b

    #     return (len(g.nodes()), len(edges), ctypes.cast(e_list_from, ctypes.c_void_p), ctypes.cast(e_list_to, ctypes.c_void_p))

    # def TakeSnapshot(self):
    #     self.lib.UpdateSnapshot()

    # def ClearTrainGraphs(self):
    #     self.ngraph_train = 0
    #     self.lib.ClearTrainGraphs()

    # def InsertGraph(self, g, is_test):
    #     n_nodes, n_edges, e_froms, e_tos = self.__CtypeNetworkX(g)
    #     if is_test:
    #         t = self.ngraph_test
    #         self.ngraph_test += 1
    #     else:
    #         t = self.ngraph_train
    #         self.ngraph_train += 1

    #     self.lib.InsertGraph(is_test, t, n_nodes, n_edges, e_froms, e_tos)

    # def LoadModel(self, path_to_model):
    #     p = ctypes.cast(path_to_model, ctypes.c_char_p)
    #     self.lib.LoadModel(p)

    # def GetSol(self, gid, maxn):
    #     sol = (ctypes.c_int * (maxn + 10))()
    #     val = self.lib.GetSol(gid, sol)
    #     return val, sol

dll_path = '%s/build/dll/libmol.so' % os.path.dirname(os.path.realpath(__file__))
if os.path.exists(dll_path):
    MOLLIB = _mol_lib()

    class MolGraph(object):

        def __init__(self, handle, smiles, pce):
            self.smiles = smiles
            self.handle = ctypes.c_void_p(handle)
            self.num_nodes = MOLLIB.lib.NumNodes(self.handle)
            self.num_edges = MOLLIB.lib.NumEdges(self.handle)
            # self.edge_pairs = np.ctypeslib.as_array(MOLLIB.lib.EdgeList(self.handle), shape=( self.num_edges * 2, ))
            self.edge_pairs = ctypes.c_void_p(MOLLIB.lib.EdgeList(self.handle))
            self.pce = pce
else:
    MOLLIB = None
    MolGraph = None

if __name__ == '__main__':
    MOLLIB.DumpFeatures('data/train.txt')
    MOLLIB.DumpFeatures('data/valid.txt')
    MOLLIB.DumpFeatures('data/test.txt')