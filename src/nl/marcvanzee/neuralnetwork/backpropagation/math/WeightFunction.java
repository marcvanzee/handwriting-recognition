package nl.marcvanzee.neuralnetwork.backpropagation.math;

//Marc van Zee (F093385)   Creating a Neural Network
//http://www.marcvanzee.nl    marcvanzee@hotmail.com
//University Utrecht         Artificial Intelligence

/*********************************
* WeightFunction.java
* 
* The standard weight function to calculate the new weight values of the lines
* Momentum can be ignored by entering 0 for it. All other values (except previousWeight) are required.
*/

public final class WeightFunction {
	
	public static final double compute(double learningRate, double momentum, double toError, 
										double fromOutput, double lineWeight, double previousWeight) {
		
		double deltaWeight = learningRate * toError * fromOutput;
	    return lineWeight + deltaWeight + momentum * previousWeight;
	}
}
