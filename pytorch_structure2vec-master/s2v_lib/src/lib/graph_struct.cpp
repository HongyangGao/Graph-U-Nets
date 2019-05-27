#include "graph_struct.h"

template<typename T>
LinkedTable<T>::LinkedTable()
{
	n = ncap = 0;
	head.clear();	
}

template<typename T>
void LinkedTable<T>::AddEntry(int head_id, T content)
{			
	if (head_id >= n)
	{				
		if (head_id + 1 > ncap)
		{
			ncap = std::max(ncap * 2, head_id + 1);
			head.resize(ncap);	
			for (int i = n; i < head_id + 1; ++i)
				head[i].clear();
		}
		n = head_id + 1;
	}
	
	head[head_id].push_back(content);
}

template<typename T>
void LinkedTable<T>::Resize(int new_n)
{
	if (new_n > ncap)
	{
		ncap = std::max(ncap * 2, new_n);
		head.resize(ncap);
	}
	n = new_n;
	for (size_t i = 0; i < head.size(); ++i)
		head[i].clear();
}

template class LinkedTable<int>;
template class LinkedTable< std::pair<int, int> >;

GraphStruct::GraphStruct()
{
	out_edges = new LinkedTable< std::pair<int, int> >();
    in_edges = new LinkedTable< std::pair<int, int> >();
	subgraph = new LinkedTable< int >();
    edge_list.clear();
}

GraphStruct::~GraphStruct()
{
	delete out_edges;
    delete in_edges;
	delete subgraph;
}

void GraphStruct::AddEdge(int idx, int x, int y)
{
    out_edges->AddEntry(x, std::pair<int, int>(idx, y));
    in_edges->AddEntry(y, std::pair<int, int>(idx, x));         
	num_edges++;
    edge_list.push_back(std::make_pair(x, y));
    assert(num_edges == edge_list.size());
    assert(num_edges - 1 == (unsigned)idx);
}

void GraphStruct::AddNode(int subg_id, int n_idx)
{
	subgraph->AddEntry(subg_id, n_idx);
}

void GraphStruct::Resize(unsigned _num_subgraph, unsigned _num_nodes)
{
	num_nodes = _num_nodes;
	num_edges = 0;
    edge_list.clear();
	num_subgraph = _num_subgraph;
	
	in_edges->Resize(num_nodes);
    out_edges->Resize(num_nodes);
	subgraph->Resize(num_subgraph);
}
