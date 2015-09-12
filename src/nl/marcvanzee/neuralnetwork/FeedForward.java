package nl.marcvanzee.neuralnetwork;

//Marc van Zee (F093385)   Creating a Neural Network
//http://www.marcvanzee.nl    marcvanzee@hotmail.com
//University Utrecht         Artificial Intelligence

/*********************************
* FeedForward.java
* 
* The Feed Forward implementation of a Neural Network
* This network topology has at least two layers where every Node in a layer,
* is connected to all the Nodes of the previous layer.
* 
* This specific class contains exactly that implementation, and it also has
* a method returning a string with the current topology.
* 
* Furthermore, it contain a method that inputs and example into the network,
* and a method to reset all the Nodes in the network. These are added because
* they seem to be practical for different implementations of the Feed Forward technique.
* 
* THE ARRAY/VECTOR DILEMMA
* 
* I first coded the entire structure of the network using Vectors. I am a big fan of
* Vectors because of their dynamic properties, but now I have learned that this property
* comes with a price. When I used the Vectors, the algorithm worked well, but the results
* were impossibly slow. I took around 2 hours to run all the examples 3 times!
* 
* When I changed everything to array, I gained huge speed profits.
* As you can see now, the algorithm is relatively fast.
* 
* So, although I see Vectors as more elegant than arrays, in this context they were not 
* fitting as a solution.
*/

import nl.marcvanzee.neuralnetwork.elements.*;

// FFANN = feed forward artificial neural network
public abstract class FeedForward implements NeuralNetwork {	
	// the layers of the network
	protected NodeLayer inputLayer;
	protected NodeLayer[] hiddenLayers;
	protected NodeLayer outputLayer;
	
	protected LineLayer[] lineLayers;
	
	// inserts an image into the network
	// abstract: to be implemented by the algorithm that inherits this class
	public abstract void train(double[] image, int answer);
	public abstract void test(double[] image, int answer);
		
	// constructor: build the network
	public FeedForward(int[] networkLayers) {
		addNodes(networkLayers);
		connectNodes();
	}

	public void addNodes(int[] networkLayers) {
		// first initialize all the layers with their corresponding size (this is
		// what i hate about arrays)
		inputLayer = new NodeLayer(networkLayers[0]);
		hiddenLayers = new NodeLayer[networkLayers.length-2];
		outputLayer = new NodeLayer(networkLayers[networkLayers.length-1]);
		
		// now add the Nodes to the input layer
		for (int i=0; i<networkLayers[0]; i++) {
			inputLayer.add(new Node(), i);
		}

		// then all hidden nodes to their corresponding layer
		for (int i=1; i<networkLayers.length - 1; i++) {
			NodeLayer tempNodeLayer = new NodeLayer(networkLayers[i]);
			
			for (int j=0; j<networkLayers[i]; j++) {
				tempNodeLayer.add(new Node(), j);
			}
			hiddenLayers[i-1] = tempNodeLayer;
		}
		
		// finally add output nodes
		for (int i=0; i<networkLayers[networkLayers.length-1]; i++) {
			outputLayer.add(new OutputNode(), i);
		}
	}
	
	public void connectNodes() {
		NodeLayer connectingOutputLayer;
		
		// lineLayers: an array of LineLayer that contain all the layers
		// of the lines. first determine the size of this array
		//
		// connect at least the input to the output (1), for every
		// hidden layer we need an extra line layer
		lineLayers = new LineLayer[hiddenLayers.length + 1];
		
		// connect input node to next layer, is this a hidden layer?
		if (hiddenLayers.length > 0) {
			// there is at least one hidden node layer, so connect every node in the first hidden layer to all input nodes
			lineLayers[0] = connectLayers(inputLayer, hiddenLayers[0]);
			
			// now iterate over all hidden layers and connect them to each other
			for (int i=1; i<hiddenLayers.length; i++) {				
				lineLayers[i] = connectLayers(hiddenLayers[i-1], hiddenLayers[i]);
			}
			
			// make sure that the output layer will connect to the last hidden layer			
			connectingOutputLayer = hiddenLayers[hiddenLayers.length - 1];
		} else {
			// there is no hidden layer, so let the output layer connect to the input layer
			connectingOutputLayer = inputLayer;
		}
		
		// connect every output node to all the nodes in the connectingOutputLayer
		lineLayers[lineLayers.length - 1] = connectLayers(connectingOutputLayer, outputLayer);
	}
		
	// the core of the FeedForward algorithm:
	// connect every node of layer1 to all the nodes of layer2
	public LineLayer connectLayers(NodeLayer layer1, NodeLayer layer2) {
		// first determine the size of the LineLayer
		LineLayer tempLineLayer = new LineLayer(layer1.size() * layer2.size());
		
		for (int i=0; i<layer2.size(); i++) {
			for (int j=0; j<layer1.size(); j++) {
				// iterate over all the Nodes in layer1 and connect them to every node
				// of layer 2. first determine position in the array.
				int curPos = layer1.size() * i + j;
				// then add the line to the LineLayer
				tempLineLayer.add(new Line(layer1.get(j), layer2.get(i)), curPos);
			}	
		}
		
		return tempLineLayer;	
	}
				
	public void inputExample(double image[]) {	
		for (int i=0; i<image.length; i++) {
			inputLayer.get(i).setOutput(image[i]);
		}
	}
		
	public String printTopology() {
		String str = "";
		
		str += "input nodes: " + inputLayer.size();
		str += "\n-- connections: " + lineLayers[0].size();
		for (int i=0; i<hiddenLayers.length; i++) {
			str += "\nhidden layer (" + i + "): " + hiddenLayers[i].size();
			str += "\n-- connections: " + lineLayers[i + 1].size();
		}
		str += "\noutput nodes: " + outputLayer.size() + "\n\n";
		return str;
	}
	
	protected void resetNodes() {
		// NOTE: input nodes need no reset.
		//       their ouput value will be equal to the input of the image
		//       and their error value is irrelevant
		
		// iterate over all hidden layers
		for (int i=0; i<hiddenLayers.length; i++) {
			// empty all nodes in this layer
			for (int j=0; j<hiddenLayers[i].size(); j++) {
				hiddenLayers[i].get(j).reset();
			}
		}
		
		// now empty all nodes in the output layer
		for (int i=0; i<outputLayer.size(); i++) {
			this.outputLayer.get(i).reset();
		}
	}
}
