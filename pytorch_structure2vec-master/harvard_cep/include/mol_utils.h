#ifndef UTILS_H
#define UTILS_H

#include "mol_lib.h"
#include <GraphMol/RingInfo.h>
#include <vector>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <map>


/*
'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
'Cr', 'Pt', 'Hg', 'Pb'*/
const unsigned atom_nums[] = {6, 7, 8, 16, 9, 14, 15, 17, 35, 12, 11, 
							  20, 26, 33, 13, 53, 5, 23, 19, 81, 70,
							  51, 50, 47, 85, 27, 34, 22, 30, 1, 
							  3, 32, 29, 79, 28, 48, 49, 25, 40, 
							  64, 78, 80, 82};
							  
struct MolFeat
{
	static void InitIdxMap();
	
	static void ParseAtomFeat(Dtype* arr, int feat);
	
	static int AtomFeat(const RDKit::Atom* cur_atom);
	
	static int EdgeFeat(const RDKit::Bond* bond);
	
	static void ParseEdgeFeat(Dtype* arr, int feat);
	
	static const int nodefeat_dim = 62;
	static const int edgefeat_dim = 6;
	static std::map<unsigned, unsigned> atom_idx_map;
};
				
struct MolGraph
{
		MolGraph(int _num_nodes, int _num_edges);
		
		void GetDegrees();
		
		inline int node_feat_at(int node_idx)
		{
			return node_features[node_idx];
		}
		
		inline int edge_feat_at(int edge_idx)
		{
			return edge_features[edge_idx];
		}
		
		std::vector< std::vector<int> > adj_list;
		int* edge_pairs, *degrees;
		int num_nodes, num_edges;
		int* node_features, *edge_features;
};

#endif