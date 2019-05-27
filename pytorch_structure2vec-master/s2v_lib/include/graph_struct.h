#ifndef GRAPH_STRUCT_H
#define GRAPH_STRUCT_H

#include <vector>
#include <map>
#include <iostream>
#include <cassert>

/**
 * @brief      Class for linked table (an array of linked list)
 *
 * @tparam     T     element type in the linked list
 */
template<typename T>
class LinkedTable
{
public:
		/**
		 * @brief      constructor; create an empty linked table
		 */
		LinkedTable();

		/**
		 * @brief      add an entry to the head_id-th linked list with content
		 *
		 * @param[in]  head_id  The linked list identifier
		 * @param[in]  content  The content to be append
		 */
		void AddEntry(int head_id, T content);
		
		/**
		 * @brief      preserve the space for header array
		 *
		 * @param[in]  new_n  new array size
		 */
		void Resize(int new_n);
		
		/**
		 * header array size
		 */
		int n;

		/**
		 * array of linked list (vector of vector)
		 */
		std::vector< std::vector<T> > head;

private:
		/**
		 * capacity of the head vector; when doing resize, we won't clean up the head,
		 * rather, we will only increat the size when ncap is not enough; 
		 */
		int ncap;
};

/**
 * @brief      represents a (directed) graph
 */
class GraphStruct
{
public:
	/**
	 * @brief      construct an empty graph
	 */
	GraphStruct();

	/**
	 * @brief      delete the pointers
	 */
	~GraphStruct();
	
	/**
	 * @brief      Adds an edge.
	 *
	 * @param[in]  idx   The edge index, must add in the increasing order of idx (start from 0)
	 * @param[in]  x     edge start
	 * @param[in]  y     edge end
	 */
	void AddEdge(int idx, int x, int y);

	/**
	 * @brief      Add a node to the graph
	 *
	 * @param[in]  subg_id  The subgraph identifier (used for mini-batch)
	 * @param[in]  n_idx    The node index
	 */
	void AddNode(int subg_id, int n_idx);
	
	/**
	 * @brief      preserve the space
	 *
	 * @param[in]  _num_subgraph  The number of subgraph
	 * @param[in]  _num_nodes     The number of nodes
	 */
	void Resize(unsigned _num_subgraph, unsigned _num_nodes = 0);
	
	/**
	 * adjacent list representation; stores out going edges of a node
	 */
	LinkedTable< std::pair<int, int> > *out_edges;
	/**
	 * adjacent list representation; stores incoming edges of a node
	 */
	LinkedTable< std::pair<int, int> > *in_edges;

	/**
	 * track the node lists of each subgraph
	 */
	LinkedTable< int >* subgraph;

	/**
	 * edge list ({x->y}), where the position in the edge_list corresponds to the edge index
	 */
	std::vector< std::pair<int, int> > edge_list;    

	/**
	 * total # nodes 
	 */
	unsigned num_nodes;
	/**
	 * total # edges
	 */
	unsigned num_edges;
	/**
	 * # subgraphs
	 */
	unsigned num_subgraph;	
};

extern GraphStruct batch_graph; // global variable

#endif