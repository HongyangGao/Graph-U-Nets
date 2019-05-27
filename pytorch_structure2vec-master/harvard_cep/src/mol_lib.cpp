#include "mol_lib.h"
#include "mol_utils.h"

#include <random>
#include <algorithm>
#include <cstdlib>
#include <signal.h>

// int Init(const int argc, const char** argv)
// {
//     MolFeat::InitIdxMap();
//     return 0;
// }

int NodeFeatDim()
{
    return MolFeat::nodefeat_dim;
}

int EdgeFeatDim()
{
    return MolFeat::edgefeat_dim;
}

int NumNodes(void* _g)
{
    MolGraph* g = static_cast<MolGraph*>(_g);
    return g->num_nodes;
}

int NumEdges(void* _g)
{
    MolGraph* g = static_cast<MolGraph*>(_g);
    return g->num_edges;
}

void* EdgeList(void* _g)
{
    MolGraph* g = static_cast<MolGraph*>(_g);
    return g->edge_pairs;
}

int PrepareBatchFeature(const int num_graphs, void** g_list, Dtype* node_input, Dtype* edge_input)
{    
    unsigned edge_cnt = 0, node_cnt = 0;
    
    for (int i = 0; i < num_graphs; ++i)
    {
        MolGraph* g = static_cast<MolGraph*>(g_list[i]);
		node_cnt += g->num_nodes;
		edge_cnt += g->num_edges;
    }
    
    Dtype* ptr = node_input;
    for (int i = 0; i < num_graphs; ++i)
    {
        MolGraph* g = static_cast<MolGraph*>(g_list[i]);

		for (int j = 0; j < g->num_nodes; ++j)
		{
			MolFeat::ParseAtomFeat(ptr, g->node_feat_at(j));
			ptr += MolFeat::nodefeat_dim;
		}
    }

	ptr = edge_input;
	for (int i = 0; i < num_graphs; ++i)
	{
        MolGraph* g = static_cast<MolGraph*>(g_list[i]);
		for (int j = 0; j < g->num_edges * 2; j += 2)
		{
			// two directions have the same feature
			MolFeat::ParseEdgeFeat(ptr, g->edge_feat_at(j / 2));
			ptr += MolFeat::edgefeat_dim;
			MolFeat::ParseEdgeFeat(ptr, g->edge_feat_at(j / 2));
			ptr += MolFeat::edgefeat_dim;
		}
	}

    return 0;
}

int DumpFeatures(const char* raw_csv)
{
	char outputfile[1000]; 
	sprintf(outputfile, "%s.bin", raw_csv);

	std::cerr << "filename = " << raw_csv << std::endl;
	RDKit::SmilesMolSupplier supplier(raw_csv, " ", 0, 0, false, true);
	RDKit::ROMol* mol;
	MolFeat::InitIdxMap();
	std::vector<int> node_feats, edge_feats;
	FILE* debug = fopen("logwrite", "w"); 
	std::cerr << "parsing " << raw_csv << std::endl;
	FILE* fid = fopen(outputfile, "wb");
	
	node_feats.clear();
	edge_feats.clear();
	int cur_cnt = 0, total_nodes = 0, total_edges = 0;
	while (!supplier.atEnd())
	{
		mol = supplier.next();
		int num_nodes = mol->getNumAtoms(), num_edges = mol->getNumBonds();
		fwrite(&num_nodes, sizeof(int), 1, fid);
		fwrite(&num_edges, sizeof(int), 1, fid);
		fprintf(debug, "%d %d\n", num_nodes, num_edges);
		total_nodes += num_nodes;
		total_edges += num_edges;
		for (unsigned int i = 0; i < mol->getNumAtoms(); ++i)
		{
			const RDKit::Atom* cur_atom = mol->getAtomWithIdx(i);
			node_feats.push_back(MolFeat::AtomFeat(cur_atom));
			fprintf(debug, "%d\n", MolFeat::AtomFeat(cur_atom));
		}
		for (unsigned i = 0; i < mol->getNumBonds(); ++i)
		{
			const RDKit::Bond* bond = mol->getBondWithIdx(i);
			unsigned int x = bond->getBeginAtomIdx();
			unsigned int y = bond->getEndAtomIdx();
			fwrite(&x, sizeof(int), 1, fid);			
			fwrite(&y, sizeof(int), 1, fid);
			edge_feats.push_back(MolFeat::EdgeFeat(bond));
		}
		delete mol;		
		
		cur_cnt++;		
		if (cur_cnt % 10000 == 0)
			std::cerr << cur_cnt << std::endl;
	}
	std::cerr << "totally " << cur_cnt << " molecules processed" << std::endl;
	printf("num_nodes: %d, num_edges: %d\n", (int)node_feats.size(), (int)edge_feats.size());
	cur_cnt = 0;
	fwrite(&cur_cnt, sizeof(int), 1, fid);
	fwrite(&cur_cnt, sizeof(int), 1, fid);
	fclose(debug);
	fwrite(node_feats.data(), sizeof(int), node_feats.size(), fid);
	fwrite(edge_feats.data(), sizeof(int), edge_feats.size(), fid);
	fclose(fid);

	return 0;
}

int LoadMolGraph(const char* filename, void** mol_list)
{
    assert(filename);
    FILE* fid = fopen(filename, "rb");
    unsigned num_nodes, num_edges;
    unsigned total_nodes = 0, total_edges = 0;
    std::cerr << "loading graph from " << filename << std::endl; 
    int num_mols = 0;
    while (true)
    {
        assert(fread(&num_nodes, sizeof(int), 1, fid) == 1);
        assert(fread(&num_edges, sizeof(int), 1, fid) == 1);

        if (num_nodes <= 0 || num_edges <= 0)
            break;
        MolGraph* g = new MolGraph(num_nodes, num_edges);

        assert(fread(g->edge_pairs, sizeof(int), num_edges * 2, fid) == num_edges * 2);
        for (int j = 0; j < g->num_edges * 2; j += 2)
        {
            auto x = g->edge_pairs[j];
            auto y = g->edge_pairs[j + 1];
            g->adj_list[x].push_back(y);
            g->adj_list[y].push_back(x);
        }

        total_nodes += num_nodes;
        total_edges += num_edges; 
		mol_list[num_mols] = g;
		num_mols++;
	}
    std::cerr << "num_nodes: " << total_nodes << "\tnum_edges: " << total_edges << std::endl;
    auto* node_features = new int[total_nodes];
    auto* edge_features = new int[total_edges];

    assert(fread(node_features, sizeof(int), total_nodes, fid) == total_nodes);			
    assert(fread(edge_features, sizeof(int), total_edges, fid) == total_edges);
    fclose(fid);

    total_nodes = 0; total_edges = 0;
    for (int i = 0; i < num_mols; ++i)
    {
        MolGraph* g = static_cast<MolGraph*>(mol_list[i]);
        g->node_features = node_features + total_nodes;
        g->edge_features = edge_features + total_edges;
        g->GetDegrees();
        total_nodes += g->num_nodes;
		total_edges += g->num_edges;			
	}

    return num_mols;
}