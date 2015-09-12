package nl.marcvanzee.neuralnetwork;

//Marc van Zee (F093385)   Creating a Neural Network
//http://www.marcvanzee.nl    marcvanzee@hotmail.com
//University Utrecht         Artificial Intelligence

/*********************************
 * NeuralNetwork.java
 * 
 * an interface for a neural network; contains minimum collection of
 * methods that a network should contain.
 * 
 * A neural network is defined in this context with the following characteristics:
 * 1. a collection of Nodes, divided over several layers.
 *    minimal an input layer and an output layer, but an infinite amount of hidden layers
 *    
 * 2. between every layer of nodes there are Lines that connect the Nodes
 * 
 * 3. the network can be trained with an example image
 * 
 * 4. the network can be tested with an example image
 * 
 * 5. the error level of the current run can be retrieved
 */

import nl.marcvanzee.neuralnetwork.elements.LineLayer;
import nl.marcvanzee.neuralnetwork.elements.NodeLayer;

public interface NeuralNetwork {
	
	/************************************
     * addNodes(int[] networkLayers)
     *---------------------------------------
	 * (pre)  int[] networklayers: an array for the layers of the network
	 *                                  for example: 3 5 20 can be a network with 3 input nodes,
	 *                                  5 hidden nodes and 20 output nodes.
	 * (post) the nodes have been created an been put in a collection
	 */
	void addNodes(int[] networkLayers);
	
	/************************************
     * connectNodes()
     *---------------------------------------
	 * (pre)  Nodes are added to the network
	 * (post) connection between the Nodes are made
	 * 
	 * for every layer, the method connectLayers(layer1, layer2) is called
	 */
	void connectNodes();
	
	/************************************
     * connectLayers(NodeLayer layer1, NodeLayer layer2)
     *---------------------------------------
	 * (pre)  NodeLayer layer1 and layer2 are not empty
	 * 
	 * Connect the nodes of the two layers. 
	 * No restriction, this method can be implemented as a FeedForward network
	 * (every node in a layer connects to all the nodes in the previous layer), or a FullyConnected
	 * network, or something else.
	 */
	LineLayer connectLayers(NodeLayer layer1, NodeLayer layer2);
	
	// test and train function
	void train(double[] image, int answer);
	void test(double[] image, int answer);
}
