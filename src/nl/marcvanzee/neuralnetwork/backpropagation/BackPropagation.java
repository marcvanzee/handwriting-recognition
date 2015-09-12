package nl.marcvanzee.neuralnetwork.backpropagation;

//Marc van Zee (F093385)   Creating a Neural Network
//http://www.marcvanzee.nl    marcvanzee@hotmail.com
//University Utrecht         Artificial Intelligence

/*********************************
* BackPropagation.java
* 
* BackPropagation is an implementation of the FeedForward Neural Network that
* propagates the input through the network, depending on the weights of the
* Lines that connect the Nodes. Then it determines an error term at the output 
* and propagates this error term back through the network. From these error terms, 
* the weights of the Lines are being updated. This process iterates until an optimal
* values has been found.
* 
* As you can see, this implementation has a lot more functionality that the FeedForward
* algorithm. I want to make clear that all these functionalities are not a property of the
* Neural Network, but a property of Back propagation. That is why I have not implemented
* this method in the Node and Line classes. They only have very simple values, and all
* the calculations are done in this class.
* 
* I did use additional classes for the error, sigmoid and weight functions. This allows
* one to easily change them without having to dig in this class.
* 
*/

import java.util.Random;

import nl.marcvanzee.neuralnetwork.FeedForward;
import nl.marcvanzee.neuralnetwork.backpropagation.math.ErrorFunction;
import nl.marcvanzee.neuralnetwork.backpropagation.math.SigmoidFunction;
import nl.marcvanzee.neuralnetwork.backpropagation.math.WeightFunction;
import nl.marcvanzee.neuralnetwork.elements.*;

public class BackPropagation extends FeedForward {
	
	// ---- algorithm variables - are being initialized by the GUI
	protected double LEARNING_RATE = 0.5;
	protected double MOMENTUM = 0.3;
	protected double MIN_WEIGHT = -0.05;
	protected double MAX_WEIGHT = 0.05;
	
	// ---- error feedback variables
	// rounds:    the number of runs since the last increase in step size
	//            is being used to calculated the average error term
	// step size: for more information see GUI.java
	protected int rounds = 0, stepSize = 100;

	// previous change in weight: used for momentum
	protected double[][] previousDeltaWeight;
	
	// the average weight of this step round
	protected double[][] averageWeight;
	
	// the best configuration so far
	protected double[][] minErrorWeight;
	
	// the lowest error value so far
	protected double minError;
	
	//the error value of this run
	protected double curStepError;
	
	/************************************
     * BackPropagation(int[] networkLayers)
     *---------------------------------------
     * create network topology based on networkLayers
     * initialize the momentum and the counters for error feedback
     */
	public BackPropagation(int[] networkLayers) {
		super(networkLayers);
		initMomentum();
		initCounters();
		initWeights();
	}

	/************************************
     * train(double[] image, int answer)
     */
	public void train(double[] image, int answer) {
		inputExample(image);

		// set target of all output values
		setOutputTargets(answer);

		// now computer output of every unit
		// first reset all the netto values and errors in the nodes to 0
		resetNodes();
		
		// propagate input forward through the network
		propagateInput();
				
		// propagate errors backward through the network
		propagateError();
		
		// process these error for error feedback
		updateStepError();		
	}
	
	/************************************
     * test(double[] image, int answer)
     */
	public void test(double[] image, int answer) {	
		// very similar to train() method, except that we don't
		// propagate the error back since we are only interested in
		// the error values at the output Nodes.
		inputExample(image);
		
		setOutputTargets(answer);
		
		propagateInput();
		
		// different update step because we don't keep track of anything
		updateTestStep();
	}
	
	/************************************
     * propagateInput()
     * -----------------------------
     * propagate the output value of the input Nodes all the way
     * to the output Nodes using the Sigmoid function
     */
	private void propagateInput() {
		
		// for every line layer
		for (int i=0; i<lineLayers.length; i++) {			
			// for every line
			for (int j=0; j<lineLayers[i].size(); j++) {
				Line curLine = lineLayers[i].get(j);
				
				// propagate the output of the fromNode to the toNode
				// the output depends on the weight of this line.
				// toNode(output) += fromNode(output) * weight
				
				// NOTE: the values are being added for all lines in this Line Layer,
				//       only after all the lines have done this, the sigmoid function
				//       is being used. this has to advantage that the sum of the inputs
				//       of the Nodes and the output can be stored in one variable.
				Node toNode = curLine.getToNode();
				Node fromNode = curLine.getFromNode();
				
				toNode.incOutput(fromNode.getOutput() * curLine.getWeight());				
			}
			
			// now use the sigmoid function to calculate output values of the Nodes in this Layer
			// if there are no more hidden layers, we are connecting it to the output layer.
			NodeLayer curLayer = (i < hiddenLayers.length) ? hiddenLayers[i] : outputLayer;

			for (int j=0; j<curLayer.size(); j++) {
				// sigmoid function
				
				double netto = curLayer.get(j).getOutput();
				curLayer.get(j).setOutput(SigmoidFunction.compute(netto));
			}
		}
	}

	/************************************
     * propagateError()
     * -----------------------------
     * propagate the errors back through the network, starting at the output nodes
     */
	private void propagateError() {		
		// for each network output unit, calculate its error term
		for (int i=0; i<outputLayer.size(); i++) {
			OutputNode curNode = ((OutputNode) outputLayer.get(i));
			double curOutput   = curNode.getOutput();
			double errorValue =  ErrorFunction.computeOutput(curOutput, curNode.getTarget());
			curNode.setError(errorValue);
		}
		
		LineLayer curLineLayer;
		for (int i=(lineLayers.length-1); i>=0; i--) {
			// calculate error values of the nodes by propagating previous error using the lines
			curLineLayer = lineLayers[i];
			for (int j=0; j<curLineLayer.size(); j++) {
				Line curLine = curLineLayer.get(j);
				
				// propagate error function
				Node toNode = curLine.getToNode();
				Node fromNode = curLine.getFromNode();
				
				fromNode.incError(toNode.getError() * curLine.getWeight());
				
				// and update the weight of this line
				// NOTE: the line update function was previously in a separated method.
				//       I have put it here because a network can consist of many Lines, and
				//       it is a big bottleneck if I would use another method to iterate over
				//       all the lines again, while I can also do it here and catch two flies
				//       in once (is that an english expression?)
				double deltaWeight = LEARNING_RATE * toNode.getError() * fromNode.getOutput();
			    curLine.setWeight(WeightFunction.compute(LEARNING_RATE, MOMENTUM, toNode.getError(), fromNode.getOutput(), curLine.getWeight(), previousDeltaWeight[i][j]));
			    previousDeltaWeight[i][j] = deltaWeight;
			}
			
			// use error function to calculate error values of the nodes of this layer
			// if we are at the last lineLayer, then use the inputLayer. else, if there
			// are more than one lineLayers, we must have hiddenLayers.
			NodeLayer curLayer = (i > 0) ? hiddenLayers[i - 1] : inputLayer;
			
			for (int j=0; j<curLayer.size(); j++) {
				// error function
				Node curNode = curLayer.get(j);				
				curNode.setError(ErrorFunction.compute(curNode.getOutput(), curNode.getError()));
			}
		}
	}

	/************************************
     * initMomentum
     * -----------------------------
     * initialize all momentum values with 0
     */
	private void initMomentum() {
		int maxLines = 0;
		
		// find the maximal amount of Lines in a Line Layer
		for (int i=0; i<lineLayers.length; i++) {
			maxLines = (lineLayers[i].size() > maxLines) ? lineLayers[i].size() : maxLines;
		}
		
		// intialize the array to this value
		// NOTE: this leads to empty calls in the array, but still it is much
		//       faster than using a VectorList
		previousDeltaWeight = new double[lineLayers.length][maxLines];

		// initialize all previous weights to 0
		for (int i=0; i<lineLayers.length; i++) {			
			for (int j=0; j<lineLayers[i].size(); j++) {
				previousDeltaWeight[i][j] = 0.0;
			}
		}
	}

	/************************************
     * initCounters()
     * -----------------------------
     * initialize counter for error feedback
     */
	private void initCounters() {
		curStepError = 0;
		rounds = 0;
		stepSize = 100;
		minError = 1;
		
		// same as in initMomentum(): this causes empty cells
		averageWeight = new double[lineLayers.length][lineLayers[0].size()];
		minErrorWeight = new double[lineLayers.length][lineLayers[0].size()];
	}

	/************************************
     * updateStepError()
     * -----------------------------
     * update the current error value and the maximum error value
     */
	private void updateStepError() {
		// update only if we have run for <stepSize> times
		// rounds > 0, because 0 % x = 0
		if ((rounds % stepSize == 0) && (rounds > 0)) {
			
			// see if this round is the best so far
			if (curStepError < minError) {
				// save the configuration of the weights
				saveCurrentWeight();
				
				// and the value of the error
				minError = curStepError;
			}
			// update error
			curStepError = getCurrentError();
			rounds = 0;
			
		} else {
			double curError = getCurrentError();
			
			// update the average error by multiplying the previous error level
			// with the amount of runs, adding the current error level to it
			// and dividing this by the number of rounds+1
			curStepError = (curError + curStepError * rounds) / (rounds + 1);
			increaseAverageWeight();
		}
		rounds++;
	}
	
	/************************************
     * updateTestStep()
     * -----------------------------
     * simplified version of the updatStepError() for test run, and not training run
     */
	private void updateTestStep() {
		if (rounds % stepSize == 0) {
			curStepError = getCurrentError();
			rounds = 0;
		} else {
			curStepError = (getCurrentError() + curStepError * rounds) / (rounds + 1);
			increaseAverageWeight();
		}
		
		rounds++;
	}
	
	/************************************
     * getCurrentError()
     * -----------------------------
     * get the overall error of this network by applying the standard error function: 0.5 (target - output) ^ 2
     */
	public double getCurrentError() {
		double sum = 0;
		
		for (int i=0; i<outputLayer.size(); i++) {
			sum += Math.pow(((OutputNode) outputLayer.get(i)).getTarget() - ((OutputNode) outputLayer.get(i)).getOutput(), 2);
		}
		return sum * 0.5;
	}

	/************************************
     * initWeights()
     * -----------------------------
     * initialize the weights of the Lines using the weight boundaries
     */
	public void initWeights() {
		Random generator = new Random();
		
		for (int i=0; i<lineLayers.length; i++) {
			for (int j=0; j<lineLayers[i].size(); j++) {
				double weight = MIN_WEIGHT + generator.nextDouble() * (Math.abs(MIN_WEIGHT) + Math.abs(MAX_WEIGHT));
				lineLayers[i].get(j).setWeight(weight);
			}
		}
	}
	
	/************************************
     * the following functions are all very straightforward and need no extra commenting.
     */
	private void saveCurrentWeight() {
		for (int i=0; i<lineLayers.length; i++) {
			for (int j=0; j<lineLayers[i].size(); j++) {
				minErrorWeight[i][j] = averageWeight[i][j] / stepSize;
			}
		}
	}
	
	private void setOutputTargets(int answer) {
		for (int i=0; i<outputLayer.size(); i++) {
			((OutputNode) outputLayer.get(i)).setTarget((i==answer)?1:0);
		}
	}

	public double getCurStepError() {
		return curStepError;
	}
	
	public void setStepSize(int stepSize) {
		this.stepSize = stepSize;
	}
		
	public String printTopology() {
		String str = super.printTopology();
		str += "momentum: " + MOMENTUM + "\n\n";
		
		return str;
	}
	
	public void setMomentum(double momentum) {
		this.MOMENTUM = momentum;
	}
	
	public void increaseAverageWeight() {
		for (int i=0; i<lineLayers.length; i++) {
			for (int j=0; j<lineLayers[i].size(); j++) {
				averageWeight[i][j] += lineLayers[i].get(j).getWeight();
			}
		}
	}
	
	public void loadOptimalWeights() {
		for (int i=0; i<lineLayers.length; i++) {
			for (int j=0; j<lineLayers[i].size(); j++) {
				lineLayers[i].get(j).setWeight(minErrorWeight[i][j]);
			}
		}
	}
	
	public double[][] getOptimalWeights() {
		return minErrorWeight;
	}
	
	public void setInitWeights(double weights[]) {
		this.MIN_WEIGHT = weights[0];
		this.MAX_WEIGHT = weights[1];
	}
}
