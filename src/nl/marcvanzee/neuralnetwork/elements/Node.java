package nl.marcvanzee.neuralnetwork.elements;

//Marc van Zee (F093385)   Creating a Neural Network
//http://www.marcvanzee.nl    marcvanzee@hotmail.com
//University Utrecht         Artificial Intelligence

/*********************************
* Node.java
* 
* A node in the network.
* My goal was to keep the nodes as simple as possible.
* That is why I have not implemented any activation function.
* This will allow the Nodes to be implemented in as many different neural networks as possible.
* If one wants to specialize, these nodes can of course be extended, but the current class should
* already be a solid base for a Neural Network.
*/

public class Node  {	
	// the Node possesses only an output and an error value
	// please read BackPropagation for more information about the propagation technique.
	protected double output  = 0;
	protected double error   = 0;
	
	public Node() { }
		
	public void reset() {
		this.output = 0;
		this.error  = 0;
	}
	
	public void incOutput(double value) {
		this.output += value;
	}
	
	public void incError(double value) {
		this.error += value;
	}
	
	public void setError(double value) {
		this.error = value;
	}
	
	public void setOutput(double value) {
		this.output = value;
	}
	
	public double getOutput() {
		return this.output;
	}
	
	public double getError() {
		return error;
	}
}
