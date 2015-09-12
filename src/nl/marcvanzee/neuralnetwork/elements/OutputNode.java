package nl.marcvanzee.neuralnetwork.elements;

//Marc van Zee (F093385)   Creating a Neural Network
//http://www.marcvanzee.nl    marcvanzee@hotmail.com
//University Utrecht         Artificial Intelligence

/*********************************
* OutputNode.java
* 
* Simple extension to the Node class
* an output node should also contain a target value
*/

public class OutputNode extends Node  {	
	private double target  = 0;
	
	public OutputNode() { }

	public void setTarget(double target) {
		this.target = target;
	}
	
	public double getTarget() {
		return target;
	}
}
