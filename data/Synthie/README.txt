README for dataset Synthie


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Description === 

Synthie is a synthetic data sets consisting of 400 graphs. The data set is subdivided into four classes. Each node has a real-valued attribute vector of dimension 15 and no labels. We used the following procedure to generate the data set: First, we generated two Erdös-Rényi graphs using the G(n,p) model with p=0.2 and n = 10. For each graph we generated a seed set S_i for i in {1,2 of 200 graphs by randomly adding or deleting 25% of the edges. From these seed sets we generated two classes C_1 and C_2 of 200 graphs each by randomly sampling 10 graphs from S_1 cup S_2 and randomly connecting these graphs. For C_1 we choose a seed graph with probability 0.8 from S_1 and with probability 0.2 from S_2. The class C_2 was generated the same way but with interchanged probabilities. Finally, we generated a set of real-valued vectors of dimension 15 subdivided into two classes A and B using the make_classification method Scikit-learn. We then subdivided C_i into two classes C^A_i and C^B_i by drawing a random attribute from A or B for each node. For class C^A_i, we drew an attribute from A if the node belonged to a seed graph of seed set S_1, and from B otherwise. Class C^B_i was created the same way but with interchanged seed sets.

=== Previous Use of the Dataset ===

Christopher Morris, Nils M. Kriege, Kristian Kersting, Petra Mutzel. Faster Kernels for Graphs with Continuous Attributes via Hashing, IEEE International Conference on Data Mining (ICDM) 2016 

=== References ===

Christopher Morris, Nils M. Kriege, Kristian Kersting, Petra Mutzel. Faster Kernels for Graphs with Continuous Attributes via Hashing, IEEE International Conference on Data Mining (ICDM) 2016 

