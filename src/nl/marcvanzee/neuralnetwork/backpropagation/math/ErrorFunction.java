package nl.marcvanzee.neuralnetwork.backpropagation.math;

//Marc van Zee (F093385)   Creating a Neural Network
//http://www.marcvanzee.nl    marcvanzee@hotmail.com
//University Utrecht         Artificial Intelligence

/*********************************
* ErrorFunction.java
* 
* The standard error function to calculate the error values of the nodes.
* Notice that the output nodes primarily define the error values as the
* difference between the target value and the actual output.
* 
* All the other nodes simply propagate these errors base on their own output.
*/

public final class ErrorFunction {	
	public ErrorFunction() { }
	
	public static double computeOutput(double output, double target) {
		return output * (1 - output) * (target - output);
	}
	
	public static double compute(double output, double error) {
		return output * (1 - output) * error;
	}
}
