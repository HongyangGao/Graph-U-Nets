#include "s2v_lib.h"
#include "config.h"
#include "msg_pass.h"
#include "graph_struct.h"
#include <random>
#include <algorithm>
#include <cstdlib>
#include <signal.h>

int Init(const int argc, const char **argv)
{
    cfg::LoadParams(argc, argv);
    return 0;
}

void *GetGraphStruct()
{
    auto *batch_graph = new GraphStruct();
    return batch_graph;
}

int PrepareBatchGraph(void *_batch_graph,
                      const int num_graphs,
                      const int *num_nodes,
                      const int *num_edges,
                      void **list_of_edge_pairs,
                      int is_directed)
{
    GraphStruct *batch_graph = static_cast<GraphStruct *>(_batch_graph);
    std::vector<unsigned> prefix_sum;
    prefix_sum.clear();
    unsigned edge_cnt = 0, node_cnt = 0;

    for (int i = 0; i < num_graphs; ++i)
    {
        node_cnt += num_nodes[i];
        edge_cnt += num_edges[i];
        prefix_sum.push_back(num_nodes[i]);
        if (i)
            prefix_sum[i] += prefix_sum[i - 1];
    }
    for (int i = (int)prefix_sum.size() - 1; i > 0; --i)
        prefix_sum[i] = prefix_sum[i - 1]; // shift
    prefix_sum[0] = 0;

    batch_graph->Resize(num_graphs, node_cnt);

    for (int i = 0; i < num_graphs; ++i)
    {
        for (int j = 0; j < num_nodes[i]; ++j)
        {
            batch_graph->AddNode(i, prefix_sum[i] + j);
        }
    }

    int x, y, cur_edge = 0;
    for (int i = 0; i < num_graphs; ++i)
    {
        int *edge_pairs = static_cast<int *>(list_of_edge_pairs[i]);
        for (int j = 0; j < num_edges[i] * 2; j += 2)
        {
            x = prefix_sum[i] + edge_pairs[j];
            y = prefix_sum[i] + edge_pairs[j + 1];
            batch_graph->AddEdge(cur_edge, x, y);
            if (!is_directed)
                batch_graph->AddEdge(cur_edge + 1, y, x);
            cur_edge += 2;
        }
    }

    return 0;
}

int NumEdgePairs(void *_graph)
{
    GraphStruct *graph = static_cast<GraphStruct *>(_graph);
    int cnt = 0;
    for (uint i = 0; i < graph->num_nodes; ++i)
    {
        auto in_cnt = graph->in_edges->head[i].size();
        cnt += in_cnt * (in_cnt - 1); 
	}
    return cnt;
}

int PrepareMeanField(void *_batch_graph,
                     void **list_of_idxes,
                     void **list_of_vals)
{
    GraphStruct *batch_graph = static_cast<GraphStruct *>(_batch_graph);
    n2n_construct(batch_graph,
                  static_cast<long long *>(list_of_idxes[0]),
                  static_cast<Dtype *>(list_of_vals[0]));
    e2n_construct(batch_graph,
                  static_cast<long long *>(list_of_idxes[1]),
                  static_cast<Dtype *>(list_of_vals[1]));
    subg_construct(batch_graph,
                   static_cast<long long *>(list_of_idxes[2]),
                   static_cast<Dtype *>(list_of_vals[2]));

    return 0;
}

int PrepareLoopyBP(void *_batch_graph,
                   void **list_of_idxes,
                   void **list_of_vals)
{
    GraphStruct *batch_graph = static_cast<GraphStruct *>(_batch_graph);
    n2e_construct(batch_graph,
                  static_cast<long long *>(list_of_idxes[0]),
                  static_cast<Dtype *>(list_of_vals[0]));
    e2e_construct(batch_graph,
                  static_cast<long long *>(list_of_idxes[1]),
                  static_cast<Dtype *>(list_of_vals[1]));                  
    e2n_construct(batch_graph,
                  static_cast<long long *>(list_of_idxes[2]),
                  static_cast<Dtype *>(list_of_vals[2]));
    subg_construct(batch_graph,
                   static_cast<long long *>(list_of_idxes[3]),
                   static_cast<Dtype *>(list_of_vals[3]));
    return 0;                   
}