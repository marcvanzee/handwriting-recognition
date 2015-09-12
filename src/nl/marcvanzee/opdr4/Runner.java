package nl.marcvanzee.opdr4;

//Marc van Zee (F093385)   Creating a Neural Network
//http://www.marcvanzee.nl    marcvanzee@hotmail.com
//University Utrecht         Artificial Intelligence

/*********************************
* Runner.java
* 
* Thread object that will work the neural network.
* Is capable of both training and testing and does image processing as well.
* It reports results back to its parent, which is a GUI.
* Gets all the required settings for the network from the parent.
* 
* The thread dies when finished, or when interrupted.
*/ 

import java.io.IOException;

import mnist.tools.MnistManager;
import nl.marcvanzee.neuralnetwork.backpropagation.BackPropagation;

public class Runner extends Thread {
	private GUI parent;
	private MnistManager m;
	private boolean train;
	//BackPropagation network;
	BackPropagation network;
	
	public Runner(GUI parent, BackPropagation network) {
		// find out who the parent is, and use the network of the parent
		this.parent = parent;
		this.network = network;
	}

	public void run() {
		// open the image files
		String files[] = parent.getFiles();
		try {
			m = new MnistManager(files[0], files[1]);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// get the loop values
		int nrExamples = parent.getExamples();
		int nrRuns     = parent.getRuns();
		
		// find out what to do with the images
		double image[];
    	boolean[] preProcessor = parent.getPreProcessors();
    	
    	// value will contain the correct answers (what the network should answer)
    	int value = 0;
    	
    	// start by sending a message to the parent what we are going to do and
    	// what the network looks like
    	String modeTxt = (train) ? "training" : "testing";
    	
    	parent.Log("---- " + modeTxt + " mode\n");
    	parent.Log(network.printTopology());
    	parent.Log("--- printing error rate. graph is updated after training.");
    	
    	// if we are testing, we should tell the network to load the optimal weight values
    	// which are saved from the training session.
    	if (!train) 
    		network.loadOptimalWeights();

    	for (int runs=0; runs<nrRuns; runs++) {
	    	for (int example=1; example<nrExamples; example++) { 
	    		// load image
	    		m.setCurrent(example);

				try {
					// pre-process image according to setting defined by user
					image = processImage(m.readImage(), preProcessor);
					value = m.readLabel();
					
					if (train) {
						network.train(image, value);
					} else {
						network.test(image, value);
					}
					
					// collect the average error value of the network every [stepSize]
					// times and send these values to the plotter and the debug window in the GUI
					if (example % parent.getStepSize() == 0) {
						parent.Log(example + ": " + network.getCurStepError());
						parent.addValue(network.getCurStepError(), train);
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
				
				// if the user wants to stop the simulation, we should die.
				if (interrupted()) {
					parent.Log("--- run halted by user.");
					return;
				}
			}
    	}
    	// when finished, enable all items on the GUI again.
    	parent.enableAll(true);
    }
	
	 /************************************
     * processImage(int image[][], boolean[] preProcessor)
     *---------------------------------------
     * preprocess the image according to the values of the preProcessor
     * see GUI.java for explanation about this technique
     */
    private double[] processImage(int image[][], boolean[] preProcessor) {
    	// preProcess[1] = true => scale image
    	int imgScaled[][] = preProcessor[1] ? scaleImage(image) : image;
    	
    	// preProcess[0] = true => canvas image
    	int imgCanvas = preProcessor[0] ? 4 : 0;
    	imgCanvas = preProcessor[1] ? imgCanvas/2 : imgCanvas;
    	
    	int dim1 = imgScaled.length-(imgCanvas*2);
    	int dim2 = imgScaled[0].length-(imgCanvas*2);
		double ret[] = new double[(dim1*dim2)];

		for (int i=0; i<dim1; i++) {
			for (int j=0; j<dim2; j++) {
				// first scale values to [0,1]
				double inputValue = scale(imgScaled[i+imgCanvas][j+imgCanvas]);
				
				// enter values in the input nodes
				ret[j + dim2*i] = inputValue;
			}
		}
		return ret;
	}
    
    /************************************
     * scaleImage(int image[][])
     *---------------------------------------
     * put image in one long array and half its size
     * see GUI.java for explanation about this technique
     */
    private int[][] scaleImage(int image[][]) {
    	int img[][] = new int[image.length/2][image[0].length/2];
    	
    	// skip one pixel every step in both directions
    	for (int i=0; i<img.length; i+=2) {
    		for (int j=0; j<img[0].length; j+=2) {
    			// for every pixel, look at the three other pixels in all positive directions
    			// sum these for pixels and take the average
    	  		int values[] = { image[i][j], image[i+1][j], image[i][j+1], image[i+1][j+1] };
    	  		int average = (values[0] + values[1] + values[2] + values[3])/4;
    	  		img[i][j] = average;
    		}
    	}
    	
    	return img;
    }
    
    // use this to scale the inputs to [0,1]
	private double scale(double num) {
		return num / 255;
	}
	
	public void setTrain(boolean train) {
		this.train = train;	
	}
}
