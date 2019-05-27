#ifndef MSG_PASS_H
#define MSG_PASS_H

#include "graph_struct.h"
#include "config.h"

void n2n_construct(GraphStruct* graph, long long* idxes, Dtype* vals);

void e2n_construct(GraphStruct* graph, long long* idxes, Dtype* vals);

void n2e_construct(GraphStruct* graph, long long* idxes, Dtype* vals);

void e2e_construct(GraphStruct* graph, long long* idxes, Dtype* vals);

void subg_construct(GraphStruct* graph, long long* idxes, Dtype* vals);

#endif