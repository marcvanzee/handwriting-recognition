package nl.marcvanzee.neuralnetwork.backpropagation;

//Marc van Zee (F093385)   Creating a Neural Network
//http://www.marcvanzee.nl    marcvanzee@hotmail.com
//University Utrecht         Artificial Intelligence

/*********************************
* HalfConnected.java
* 
* An example extension of the backpropagation class, not implemented in the GUI.
* The connectLayers class is overwritten from the FeedForward class, and instead
* of connecting every node of one layer to all nodes of the previous layer, we
* connect it to only half of the previous layer Nodes, randomly chosen.
* 
* Interesting, because double line connections are possible then.
* 
* To implement it: replace every occurrence of BackPropagation in GUI and
* Runner to HalfConnected.
* 
*/

import java.util.Random;
import nl.marcvanzee.neuralnetwork.elements.*;

public class HalfConnected extends BackPropagation {
	public HalfConnected(int[] networkLayers) {
		super(networkLayers);
	}

	public LineLayer connectLayers(NodeLayer layer1, NodeLayer layer2) {
		// first determine the size of the LineLayer
		LineLayer tempLineLayer = new LineLayer(layer1.size() * (layer2.size()/2));
		
		Random generator = new Random();

		for (int i=0; i<layer2.size(); i++) {
			for (int j=0; j<layer1.size()/2; j++) {
				
				int curPos = layer1.size()/2 * i + j;
				// connect layer2 to a random node in layer 1
				tempLineLayer.add(new Line(layer1.get(generator.nextInt(layer1.size())), layer2.get(i)), curPos);
			}	
		}
		
		return tempLineLayer;	
	}
	
}
