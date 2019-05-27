#include "msg_pass.h"

void n2n_construct(GraphStruct* graph, long long* idxes, Dtype* vals)
{
    int nnz = 0;    
    long long* row_ptr = idxes;
    long long* col_ptr = idxes + graph->num_edges;

	for (uint i = 0; i < graph->num_nodes; ++i)
	{
		auto& list = graph->in_edges->head[i];
		for (size_t j = 0; j < list.size(); ++j)
		{            
            vals[nnz] = cfg::msg_average ? 1.0 / list.size() : 1.0;
            row_ptr[nnz] = i;
			col_ptr[nnz] = list[j].second;
			nnz++;
		}
	}
	assert(nnz == (int)graph->num_edges);
}

void e2n_construct(GraphStruct* graph, long long* idxes, Dtype* vals)
{
    int nnz = 0;
    long long* row_ptr = idxes;
    long long* col_ptr = idxes + graph->num_edges;

	for (uint i = 0; i < graph->num_nodes; ++i)
	{
		auto& list = graph->in_edges->head[i];
		for (size_t j = 0; j < list.size(); ++j)
		{
			vals[nnz] = cfg::msg_average ? 1.0 / list.size() : 1.0;
			row_ptr[nnz] = i;
			col_ptr[nnz] = list[j].first;
			nnz++;			
		}
	}
	assert(nnz == (int)graph->num_edges);
}

void n2e_construct(GraphStruct* graph, long long* idxes, Dtype* vals)
{
    int nnz = 0;
    long long* row_ptr = idxes;
    long long* col_ptr = idxes + graph->num_edges;

	for (uint i = 0; i < graph->num_edges; ++i)
	{
	    row_ptr[nnz] = i;
	    vals[nnz] = 1.0;
	    col_ptr[nnz] = graph->edge_list[i].first;
	    nnz++;
	}
	assert(nnz == (int)graph->num_edges);
}

void e2e_construct(GraphStruct* graph, long long* idxes, Dtype* vals)
{
    size_t cnt = 0;
    for (uint i = 0; i < graph->num_nodes; ++i)
    {
        auto in_cnt = graph->in_edges->head[i].size();
        cnt += in_cnt * (in_cnt - 1);
    }
    long long* row_ptr = idxes;
    long long* col_ptr = idxes + cnt;
    int nnz = 0;
    for (uint i = 0; i < graph->num_edges; ++i)
    {
        int node_from = graph->edge_list[i].first, node_to = graph->edge_list[i].second;
        auto& list = graph->in_edges->head[node_from]; 
        for (size_t j = 0; j < list.size(); ++j)
        {
            if (list[j].second == node_to)
                continue;
            row_ptr[nnz] = i;
            vals[nnz] = cfg::msg_average ? 1.0 / (list.size() - 1) : 1.0;
            col_ptr[nnz] = list[j].first;
            nnz++;
        }
    }
    assert(nnz == (int)cnt); 
}

void subg_construct(GraphStruct* graph, long long* idxes, Dtype* vals)
{
    int nnz = 0;    
    long long* row_ptr = idxes;
	long long* col_ptr = idxes + graph->num_nodes;

	for (uint i = 0; i < graph->num_subgraph; ++i)
	{
		auto& list = graph->subgraph->head[i];
		for (size_t j = 0; j < list.size(); ++j)
		{
			vals[nnz] = cfg::msg_average ? 1.0 / list.size() : 1.0;
			row_ptr[nnz] = i;
			col_ptr[nnz] = list[j];
			nnz++;
		}
	}	
	assert(nnz == (int)graph->num_nodes);	
}
