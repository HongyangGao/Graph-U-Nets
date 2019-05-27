#include "mol_utils.h"

void MolFeat::InitIdxMap()
{
    atom_idx_map.clear();
    for (unsigned i = 0; i < sizeof(atom_nums) / sizeof(unsigned); ++i)
        atom_idx_map[atom_nums[i]] = i;
}

int MolFeat::AtomFeat(const RDKit::Atom* cur_atom)
{
    // getIsAromatic
    int feat = cur_atom->getIsAromatic();
    feat = feat << 4;
    // getImplicitValence
    if (cur_atom->getImplicitValence() <= 5)
        feat |= cur_atom->getImplicitValence();
    else	
        feat |= 5;
    feat = feat << 4;
    // getTotalNumHs
    if (cur_atom->getTotalNumHs() <= 4)
        feat |= cur_atom->getTotalNumHs();
    else
        feat |= 4;
    feat = feat << 4;
    // getDegree
    feat |= cur_atom->getDegree();
    feat = feat << 8;
    // atom_idx_map
    unsigned x = cur_atom->getAtomicNum();
    if (atom_idx_map.count(x))
        feat |= atom_idx_map[x];				
    else
        feat |= atom_idx_map.size();
        
    return feat;	
}

void MolFeat::ParseAtomFeat(Dtype* arr, int feat)
{
    // atom_idx_map
    int t = feat & ((1 << 8) - 1);
    arr[t] = 1.0;
    feat >>= 8;		
    // getDegree
    int mask = (1 << 4) - 1;
    t = feat & mask;
    arr[44 + t] = 1.0;
    feat >>= 4;
    // getTotalNumHs
    t = feat & mask;
    arr[50 + t] = 1.0;
    feat >>= 4;
    // getImplicitValence
    t = feat & mask;
    arr[55 + t] = 1.0;
    feat >>= 4;
    // getIsAromatic
    if (feat & mask)
        arr[61] = 1.0;
}
	
int MolFeat::EdgeFeat(const RDKit::Bond* bond)
{			
    int t = 0;
    auto bt = bond->getBondType();
    if (bt == RDKit::Bond::SINGLE)
        t = 0;
    if (bt == RDKit::Bond::DOUBLE)
        t = 1;
    if (bt == RDKit::Bond::TRIPLE)
        t = 2;				
    if (bt == RDKit::Bond::AROMATIC)
        t = 3;
                        
    int feat = (bond->getOwningMol().getRingInfo()->numBondRings(bond->getIdx()) != 0);  
    feat = (feat << 8) | bond->getIsConjugated();
    feat = (feat << 8) | t;
    return feat;
}

void MolFeat::ParseEdgeFeat(Dtype* arr, int feat)
{
    int mask = (1 << 8) - 1;
    // getBondType
    arr[feat & mask] = 1.0;
    feat >>= 8;		
    // getIsConjugated
    if (feat & mask)
        arr[4] = 1.0;
    feat >>= 8;		
    // is ring
    if (feat & mask)
        arr[5] = 1.0;
}

std::map<unsigned, unsigned> MolFeat::atom_idx_map;


MolGraph::MolGraph(int _num_nodes, int _num_edges)
{
    num_nodes = _num_nodes;
    num_edges = _num_edges;
    edge_pairs = new int[num_edges * 2];
    degrees = new int[num_nodes];
    adj_list.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
        adj_list[i].clear();
}

void MolGraph::GetDegrees()
{
    memset(degrees, 0, sizeof(int) * num_nodes);
    for (int i = 0; i < num_edges * 2; i += 2)
    {
        degrees[edge_pairs[i]]++;
        degrees[edge_pairs[i + 1]]++;
    }
}