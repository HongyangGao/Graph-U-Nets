import ctypes
import numpy as np
import os
import sys
import torch

class _s2v_lib(object):

    def __init__(self, args):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/build/dll/libs2v.so' % dir_path)

        self.lib.GetGraphStruct.restype = ctypes.c_void_p
        self.lib.PrepareBatchGraph.restype = ctypes.c_int
        self.lib.PrepareMeanField.restype = ctypes.c_int
        self.lib.PrepareLoopyBP.restype = ctypes.c_int
        self.lib.NumEdgePairs.restype = ctypes.c_int

        if sys.version_info[0] > 2:
            args = [arg.encode() for arg in args]  # str -> bytes for each element in args
        arr = (ctypes.c_char_p * len(args))()
        arr[:] = args
        self.lib.Init(len(args), arr)

        self.batch_graph_handle = ctypes.c_void_p(self.lib.GetGraphStruct())

    def _prepare_graph(self, graph_list, is_directed=0):    
        edgepair_list = (ctypes.c_void_p * len(graph_list))()
        list_num_nodes = np.zeros((len(graph_list), ), dtype=np.int32)
        list_num_edges = np.zeros((len(graph_list), ), dtype=np.int32)        
        for i in range(len(graph_list)):
            if type(graph_list[i].edge_pairs) is ctypes.c_void_p:
                edgepair_list[i] = graph_list[i].edge_pairs
            elif type(graph_list[i].edge_pairs) is np.ndarray:
                edgepair_list[i] = ctypes.c_void_p(graph_list[i].edge_pairs.ctypes.data)
            else:
                raise NotImplementedError

            list_num_nodes[i] = graph_list[i].num_nodes
            list_num_edges[i] = graph_list[i].num_edges
        total_num_nodes = np.sum(list_num_nodes)
        total_num_edges = np.sum(list_num_edges)

        self.lib.PrepareBatchGraph(self.batch_graph_handle, 
                                len(graph_list), 
                                ctypes.c_void_p(list_num_nodes.ctypes.data),
                                ctypes.c_void_p(list_num_edges.ctypes.data),
                                ctypes.cast(edgepair_list, ctypes.c_void_p),
                                is_directed)

        return total_num_nodes, total_num_edges

    def PrepareMeanField(self, graph_list, is_directed=0):
        assert not is_directed
        total_num_nodes, total_num_edges = self._prepare_graph(graph_list, is_directed)
        
        n2n_idxes = torch.LongTensor(2, total_num_edges * 2)
        n2n_vals = torch.FloatTensor(total_num_edges * 2)

        e2n_idxes = torch.LongTensor(2, total_num_edges * 2)
        e2n_vals = torch.FloatTensor(total_num_edges * 2)

        subg_idxes = torch.LongTensor(2, total_num_nodes)
        subg_vals = torch.FloatTensor(total_num_nodes)

        idx_list = (ctypes.c_void_p * 3)()
        idx_list[0] = n2n_idxes.numpy().ctypes.data
        idx_list[1] = e2n_idxes.numpy().ctypes.data
        idx_list[2] = subg_idxes.numpy().ctypes.data

        val_list = (ctypes.c_void_p * 3)()
        val_list[0] = n2n_vals.numpy().ctypes.data
        val_list[1] = e2n_vals.numpy().ctypes.data
        val_list[2] = subg_vals.numpy().ctypes.data

        self.lib.PrepareMeanField(self.batch_graph_handle,
                                ctypes.cast(idx_list, ctypes.c_void_p),
                                ctypes.cast(val_list, ctypes.c_void_p))
        
        n2n_sp = torch.sparse.FloatTensor(n2n_idxes, n2n_vals, torch.Size([total_num_nodes, total_num_nodes]))
        e2n_sp = torch.sparse.FloatTensor(e2n_idxes, e2n_vals, torch.Size([total_num_nodes, total_num_edges * 2]))
        subg_sp = torch.sparse.FloatTensor(subg_idxes, subg_vals, torch.Size([len(graph_list), total_num_nodes]))
        return n2n_sp, e2n_sp, subg_sp

    def PrepareLoopyBP(self, graph_list, is_directed=0):
        assert not is_directed
        total_num_nodes, total_num_edges = self._prepare_graph(graph_list, is_directed)
        total_edge_pairs = self.lib.NumEdgePairs(self.batch_graph_handle)

        n2e_idxes = torch.LongTensor(2, total_num_edges * 2)
        n2e_vals = torch.FloatTensor(total_num_edges * 2)

        e2e_idxes = torch.LongTensor(2, total_edge_pairs)
        e2e_vals = torch.FloatTensor(total_edge_pairs)

        e2n_idxes = torch.LongTensor(2, total_num_edges * 2)
        e2n_vals = torch.FloatTensor(total_num_edges * 2)

        subg_idxes = torch.LongTensor(2, total_num_nodes)
        subg_vals = torch.FloatTensor(total_num_nodes)

        idx_list = (ctypes.c_void_p * 4)()
        idx_list[0] = ctypes.c_void_p(n2e_idxes.numpy().ctypes.data)
        idx_list[1] = ctypes.c_void_p(e2e_idxes.numpy().ctypes.data)
        idx_list[2] = ctypes.c_void_p(e2n_idxes.numpy().ctypes.data)
        idx_list[3] = ctypes.c_void_p(subg_idxes.numpy().ctypes.data)

        val_list = (ctypes.c_void_p * 4)()
        val_list[0] = ctypes.c_void_p(n2e_vals.numpy().ctypes.data)
        val_list[1] = ctypes.c_void_p(e2e_vals.numpy().ctypes.data)
        val_list[2] = ctypes.c_void_p(e2n_vals.numpy().ctypes.data)
        val_list[3] = ctypes.c_void_p(subg_vals.numpy().ctypes.data)

        self.lib.PrepareLoopyBP(self.batch_graph_handle,
                                ctypes.cast(idx_list, ctypes.c_void_p),
                                ctypes.cast(val_list, ctypes.c_void_p))

        n2e_sp = torch.sparse.FloatTensor(n2e_idxes, n2e_vals, torch.Size([total_num_edges * 2, total_num_nodes]))
        e2e_sp = torch.sparse.FloatTensor(e2e_idxes, e2e_vals, torch.Size([total_num_edges * 2, total_num_edges * 2]))
        e2n_sp = torch.sparse.FloatTensor(e2n_idxes, e2n_vals, torch.Size([total_num_nodes, total_num_edges * 2]))
        subg_sp = torch.sparse.FloatTensor(subg_idxes, subg_vals, torch.Size([len(graph_list), total_num_nodes]))

        return n2e_sp, e2e_sp, e2n_sp, subg_sp

dll_path = '%s/build/dll/libs2v.so' % os.path.dirname(os.path.realpath(__file__))
if os.path.exists(dll_path):
    S2VLIB = _s2v_lib(sys.argv)
else:
    S2VLIB = None

if __name__ == '__main__':
    sys.path.append('%s/../harvard_cep' % os.path.dirname(os.path.realpath(__file__)))
    from util import resampling_idxes, load_raw_data
    from mol_lib import MOLLIB, MolGraph

    raw_data_dict = load_raw_data()
    test_data = MOLLIB.LoadMolGraph('test', raw_data_dict['test'])

    batch_graph = test_data[0:10]

    S2VLIB.PrepareLoopyBP(batch_graph)
