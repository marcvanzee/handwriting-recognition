package nl.marcvanzee.neuralnetwork.elements;

//Marc van Zee (F093385)   Creating a Neural Network
//http://www.marcvanzee.nl    marcvanzee@hotmail.com
//University Utrecht         Artificial Intelligence

/*********************************
* NodeLayer.java
* 
* The NodeLayer is simply a collection of nodes.
* The size is static (not dynamic like a VectorList), because of efficiency consideration.
* Please read FeedForward.java for more details on this subject.
*/

public class NodeLayer {
	private Node[] nodes;
	
	public NodeLayer(int length) {
		nodes = new Node[length];
	}
	
	public void add(Node node, int index) {
		nodes[index] = node;
	}
	
	public Node get(int i) {
		return nodes[i];
	}
	
	public int size() {
		return nodes.length;
	}
}
