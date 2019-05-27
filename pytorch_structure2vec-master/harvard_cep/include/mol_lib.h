#ifndef MOL_LIB_H
#define MOL_LIB_H

typedef float Dtype;

extern "C" int Init(const int argc, const char** argv);

extern "C" int DumpFeatures(const char* raw_smiles);

extern "C" int LoadMolGraph(const char* filename, void** mol_list);

extern "C" int NodeFeatDim();

extern "C" int EdgeFeatDim();

extern "C" int NumNodes(void* _g);

extern "C" int NumEdges(void* _g);

extern "C" void* EdgeList(void* _g);

extern "C" int PrepareBatchFeature(const int num_graphs, void** g_list, Dtype* node_input, Dtype* edge_input);

#endif