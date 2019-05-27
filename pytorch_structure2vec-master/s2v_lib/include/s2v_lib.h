#ifndef S2V_LIB_H
#define S2V_LIB_H

#include "config.h"

extern "C" int Init(const int argc, const char **argv);

extern "C" void *GetGraphStruct();

extern "C" int PrepareBatchGraph(void *_batch_graph,
                                 const int num_graphs,
                                 const int *num_nodes,
                                 const int *num_edges,
                                 void **list_of_edge_pairs,
                                 int is_directed);

extern "C" int PrepareMeanField(void *_batch_graph,
                                void **list_of_idxes,
                                void **list_of_vals);

extern "C" int PrepareLoopyBP(void *_batch_graph,
                              void **list_of_idxes,
                              void **list_of_vals);

extern "C" int NumEdgePairs(void *_graph);

#endif