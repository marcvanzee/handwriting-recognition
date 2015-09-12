package nl.marcvanzee.opdr4;

// Marc van Zee (F093385)   Creating a Neural Network
// http://www.marcvanzee.nl    marcvanzee@hotmail.com
// University Utrecht         Artificial Intelligence

/*********************************
 *   GUI.java - RUN THIS METHOD  *
 *********************************
 *
 * The start method of the application.
 * Contains the GUI, where the user can set the properties of the neural network.
 * 
 * ---------- FUNCTIONS OF THE GUI EXPLAINED
 *
 * [1] NETWORK TOPOLOGY
 * 
 * Input nodes:   not editable by the user. the number of nodes depend on the input image.
 *                can be changed by preprocessing the image (enabling the checkboxes will
 *                change the input node value);
 *              
 * Hidden nodes:  the number of hidden nodes per layer, examples:
 *                3       creates 1 hidden layer of 3 nodes.
 *                []      not entering a value will create a network without any hidden layers.
 *                       this will lead to underfitting; the nodes cannot hold all the information.
 *                2 4 1   creates 3 hidden layers of respectively 2, 4 and 1 nodes, where the first
 *                       layer will connect to the input layer and the last to the output
 *              
 * Output nodes:  the number of output nodes, is always 10.
 *                every number from the input file is represented in 10 output nodes, where for every
 *                number a different output node will have a target value of 1 and all the others a 
 *                target value of 0.
 * 
 * [2] ALGORTIHM SPECIFICATIONS
 * 
 * Algorithm:     backprop & other
 * 
 * Momentum:      Making the weight update on the nth iteration depend partially on the update that
 *                occured during the (n-1)th iteration.
 *                Set to 0 for no momentum. Values between 0.1 and 0.3 are suggested.
 * 
 * Learning rate: values between 0.05 and 0.1 are suggested
 *                a lower values makes the simulation slower, but more stable.
 *      
 * Initial Weight:the boundaries of the initial values of the weights of the lines.
 *                values separated by a ,
 *                [0,5]
 *                [-1,1]
 *                [13,24.1]
 * 
 * Learning method: x
 * 
 * [3] SIMULATION SPECIFICATIONS
 * 
 * Simulation method: Training or test. first perform a training before doing any test.
 * 					  the correct files for the simulation method are selected automatically, but can
 *                    be changed by choosing another file.
 * 
 * Nr of Examples:    number of examples that the network should traing/test.
 * Nr of Runs:        how often these examples should be inserted into the network.
 * 
 * Decrease canvas:   crop the canvas from 28x28 pixels to 20x20 pixels.
 * 						" the original black and white (bilevel) images from NIST were size normalized to fit in a 
 *     					  20x20 pixel box while preserving their aspect ratio. the images were centered in a 28x28 
 *     					  image by computing the center of mass of the pixels, and translating the image so as to 
 *     					  position this point at the center of the 28x28 field."
 *  
 *     					--- source [http://yann.lecun.com/exdb/mnist/]
 *     
 * Scale image:       take a square of four pixels and reduce it to one pixel.
 *                    this was actually an experiment to speed up the training process,
 *                    it turns out that the results are very poor though.
 *                    
 * Start:             start simulation
 *                    when started, press again to stop. 
 *                    when stopped, press again to resume.
 *
 * Reset:             erase all the values of the current network, aswell as the plot.
 * 
 * [4] PLOT
 * 
 * The plot is an instance of the PTPlot Java Plotter
 * http://ptolemy.eecs.berkeley.edu/java/ptplot5.7/ptolemy/plot/doc/index.htm
 * 
 * The plot will auto adjust to the domain of the current values, and the user can zoom by click-drag-releasing
 * on the plot.
 * 
 * [5] DEBUG WINDOW
 * 
 * Will print the error values every [x] steps. These values are also being plotted.
 * It is suggested to set this to a value around 100, since a value of 1 will plot every individual result.
 * This will create a very messy plot. When a step size of 100 is entered, the average error term over 100
 * examples will be printed and will thus be much more constant.
 * 
 * Also, additional messages are printed in this window.
 * 
 * stepSize can be set in the variables below (global int stepSize)
 * 
 */

import java.applet.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;

import nl.marcvanzee.neuralnetwork.backpropagation.BackPropagation;

public class GUI extends Applet implements  ActionListener, ItemListener {
    private static final long serialVersionUID = 1L;
    
    //-------------------------- the initial values of the setting of the neural network.
    
    // these are the values that are being presented to the user when the application starts
    
    // and array of int where each number represents the amount of nodes in that layer.
    // the first number represents the input layer, and the last one the hidden.
    // the number of hidden layers is undetermined and unlimited.
    private final int[] NODES = { 400, 250, 10 };

    private final double MOMENTUM = 0.3;
    
    // learning rate
    private final double LEARNING_RATE = 0.05;
    
    // the boundaries of the initial values of the weights of the lines.
    private final double INIT_WEIGHT[] = { -0.05, 0.05 };
    
    // files to use as image database, both for training and testing
    private final String DIGIT_PATH = "../digit_db/";
    private final String FILE_TRAIN[] = { DIGIT_PATH + "train-images-idx3-ubyte", DIGIT_PATH + "train-labels-idx1-ubyte" };
    private final String FILE_TEST[] = { DIGIT_PATH + "t10k-images-idx3-ubyte", DIGIT_PATH + "t10k-labels-idx1-ubyte" };
    
    // number of examples that the network should traing/test
    private final int NR_EXAMPLES = 60000;
    
    // how often these examples should be inserted into the netwerk
    private final int NR_ROUNDS   = 3;   
    
    // two checkboxes that will preprocess the image in a way
    private boolean imgCanvas = true;
    private boolean imgScale  = false;
        
    //--------------------------  variables for program flow
    
    // whether we are doing a training. if false: we are testing the network.
    private boolean training  = true;
    
    // how many steps to perform before printing and plotting the error value.
    private int stepSize = 100;
    
    private int networkLayers[];
    
    //--------------------------  interface variables
    private Plotter plot;
    private NetworkDrawing drawingPanel;
    
    private Button imageFileButton, labelFileButton, startButton, resetButton;
	
    private TextField[] topologyNodes;
    private TextField momentumTxt, learningRateTxt, weightBoundaries, imageFile, labelFile, nrExamples, nrRuns;
    private TextArea results;
	
    private Panel body, footer, settingsContainer, plotPanel, simulationContainer, feedbackPanel;
	
    private Label topologyHeadLbl, algorithmHeadLbl, trainingHeadLbl, header;
    private Label[] topologyBodyLbl, algorithmBodyLbl;
	
    private Choice algorithmCombo, learningMethodCombo, runMethodCombo;
	
    private Checkbox[] imageProcess = new Checkbox[2];
	
    private Font h1 = new Font("Helvetica", Font.BOLD, 20);
    private Font h2 = new Font("Helvetica", Font.BOLD, 14);
    
    //--------------------------  neural network variables
    // an object that represents the network we are using: backpropagation
    //BackPropagation network;
    private BackPropagation network;
    
    // the object that actually does the calculations.
    // since this is a GUI, we want the user to be able to stop the simulation (or
    // do other things) while the simulation is running. therefore this object is
    // a thread; it will run in the background and communicate with the [network]
    // object but send results to this class.
    private Runner runThread;
    
    private Drawer drawThread;
	
    /************************************
     * initiate the GUI
     * 
     * methods: init()
     *          addHeader
     *          addTopology
     *          addAlgorithmSpecs
     *
     * this method basically sets up the entire GUI.
     * it is not very useful to comment a lot on it, since this is not the
     * scope of the assignment.
     ************************************ 
     */
    public void init() {     	
    	this.setLayout(new BorderLayout());
    	
    	addHeader();
    	
    	body = new Panel(new GridLayout(1, 2, 20, 1));  
    	body.setSize(100, 600);
    	footer = new Panel(new BorderLayout());
    	
    	addTopology();
    	addAlgorithmSpecs();
    	addTrainingSettings(); 	
    	
        resize(1000,900);  
        
        plotPanel = new Panel();
        plot = new Plotter(stepSize);
        plotPanel.add(plot);
        
        drawingPanel = new NetworkDrawing();
        
        footer.add(plotPanel, BorderLayout.NORTH);

        body.add(settingsContainer);
        body.add(footer);

        results = new TextArea("Choose your settings and Train or Test the network.\n" +
        							"First Train and then Test.\n\n", 5, 20, TextArea.SCROLLBARS_VERTICAL_ONLY);
        
        feedbackPanel = new Panel(new BorderLayout());
        feedbackPanel.add(results, BorderLayout.SOUTH);
        feedbackPanel.add(simulationContainer, BorderLayout.NORTH);

        add(body, BorderLayout.NORTH);
        add(feedbackPanel, BorderLayout.SOUTH);
        add(drawingPanel);
        
        startButton.addActionListener(this);
        resetButton.addActionListener(this);
        imageFileButton.addActionListener(this);
        labelFileButton.addActionListener(this);
        
        imageProcess[0].addItemListener(this);
        imageProcess[1].addItemListener(this);
        
        runMethodCombo.addItemListener(this);
    }
     
    private void addHeader() {
    	header = new Label("feedforward Artificial Neural Network");
    	header.setFont(h1);
    	header.setBackground(new Color(200, 200, 200));
    	add(header, BorderLayout.NORTH);
    }
    
    private void addTopology() {
    	settingsContainer = new Panel(new GridLayout(0, 2, 5, 5));
    	
    	topologyHeadLbl = new Label("Network Topology");
    	topologyHeadLbl.setFont(h2);
       	
    	topologyBodyLbl = new Label[3];
    	topologyBodyLbl[0] = new Label("Input nodes: ");
    	topologyBodyLbl[1] = new Label("Hidden nodes: ");
    	topologyBodyLbl[2] = new Label("Output nodes: ");

    	topologyNodes = new TextField[3];
    	
    	topologyNodes[0] = new TextField(Integer.toString(NODES[0]), 5);
    	topologyNodes[1] = new TextField(Integer.toString(NODES[1]), 15);
    	topologyNodes[2] = new TextField(Integer.toString(NODES[2]), 5);
    	
    	// hide the input layer because it depends on the image and process settings
    	topologyNodes[0].setEnabled(false);
    	topologyNodes[2].setEnabled(false);
    	
    	settingsContainer.add(topologyHeadLbl);
    	settingsContainer.add(new Label(" "));
    	
    	for (int i=0; i<3; i++) {
    		settingsContainer.add(topologyBodyLbl[i]);
    		settingsContainer.add(topologyNodes[i]);
    	}
    	
    	settingsContainer.add(new Label(" "));
    	settingsContainer.add(new Label(" "));
    }
    
    private void addAlgorithmSpecs() {    	
    	algorithmHeadLbl = new Label("Algorithm Specifications");
    	algorithmHeadLbl.setFont(h2);   	
    	algorithmBodyLbl = new Label[5];
    	
    	algorithmBodyLbl[0] = new Label("Algorithm: ");
    	algorithmBodyLbl[1] = new Label("Momentum (0 = off): ");
    	algorithmBodyLbl[2] = new Label("Learning Rate: ");
    	algorithmBodyLbl[3] = new Label("Init Weight (w1,w2): ");
    	algorithmBodyLbl[4] = new Label("Learning Method: ");

        algorithmCombo = new Choice();
        algorithmCombo.addItem("BackPropagation");
    	
        momentumTxt = new TextField(Double.toString(MOMENTUM), 5);
        learningRateTxt = new TextField(Double.toString(LEARNING_RATE), 5);
        weightBoundaries = new TextField(INIT_WEIGHT[0] + "," + INIT_WEIGHT[1], 5);
        
        learningMethodCombo = new Choice();
        learningMethodCombo.addItem("Online");
        
        settingsContainer.add(algorithmHeadLbl);
    	settingsContainer.add(new Label(" "));
        settingsContainer.add(algorithmBodyLbl[0]);
        settingsContainer.add(algorithmCombo);
        settingsContainer.add(algorithmBodyLbl[1]);
        settingsContainer.add(momentumTxt);
        settingsContainer.add(algorithmBodyLbl[2]);
        settingsContainer.add(learningRateTxt);
        settingsContainer.add(algorithmBodyLbl[3]);
        settingsContainer.add(weightBoundaries);
        settingsContainer.add(algorithmBodyLbl[4]);
        settingsContainer.add(learningMethodCombo);
        
        settingsContainer.add(new Label(" "));
        settingsContainer.add(new Label(" "));
    }
    
    private void addTrainingSettings() {
    	simulationContainer = new Panel(new GridLayout(0, 4, 5, 5));
    	
    	trainingHeadLbl = new Label("Simulation method");
    	trainingHeadLbl.setFont(h2);
    	
    	runMethodCombo = new Choice();
        runMethodCombo.addItem("Train");
        runMethodCombo.addItem("Test");
        runMethodCombo.addItem("Test Single Example");
    	
    	imageFile = new TextField(FILE_TRAIN[0], 30);
    	labelFile = new TextField(FILE_TRAIN[1], 30);
    	
    	imageFileButton = new Button("browse image file");
    	labelFileButton = new Button("browse label file");
    	
    	imageFile.setEnabled(false);
    	labelFile.setEnabled(false);
    	
    	imageProcess[0] = new Checkbox("decrease canvas to 20x20");
    	imageProcess[1] = new Checkbox("scale down pixels");
    	
    	imageProcess[0].setState(imgCanvas);
    	imageProcess[1].setState(imgScale);
    	
    	nrExamples = new TextField(Integer.toString(NR_EXAMPLES));
    	nrRuns = new TextField(Integer.toString(NR_ROUNDS));
    	
    	startButton = new Button("START");
    	resetButton = new Button("RESET");
    	
    	simulationContainer.add(trainingHeadLbl);
    	simulationContainer.add(runMethodCombo);
    	
    	simulationContainer.add(imageProcess[0]);
    	simulationContainer.add(imageProcess[1]);
    	
    	simulationContainer.add(imageFile);
    	simulationContainer.add(imageFileButton);
    	
    	simulationContainer.add(new Label("number of examples: "));
    	simulationContainer.add(nrExamples);
    	
    	simulationContainer.add(labelFile);
    	simulationContainer.add(labelFileButton);
    	
    	simulationContainer.add(new Label("number of runs: "));
    	simulationContainer.add(nrRuns);
    	
    	simulationContainer.add(new Label(" "));
    	simulationContainer.add(startButton);
    	simulationContainer.add(resetButton);
    	simulationContainer.add(new Label(" "));
    }
    
    /************************************
     * Log(String log)
     *---------------------------------------
     * logging function for the debug window.
     * <log> is being appended
     */
    public void Log(String log) {
    	results.append(log + "\n");
    }
    
    /************************************
     * addValue(value, train)
     *---------------------------------------
     * plot value on the plotter.
     * if boolean train is true, the training color will
     * be used in the plot (red). else, the test color (blue) will be used
     */
    public void addValue(double value, boolean train) {
    	int dataset = (train) ? 0 : 1;
    	plot.addToPlot(value, dataset);
    	
    	drawThread = new Drawer(networkLayers, network.getOptimalWeights(), drawingPanel);
    	drawThread.start();
    }
    
    /************************************
     * startTraining()
     *---------------------------------------
     * run the training.
     */
    private void startTraining() {
		networkLayers = getNetworkLayers();

		// if this is the first time we are doing a training, or if
		// the reset just has been pressed: create a new neural network
		if (network == null) {
			//network = new BackPropagation(networkLayers);
			network = new BackPropagation(networkLayers);
			
			// and initiate it with the GUI values
			network.setStepSize(getStepSize());
			network.setInitWeights(getInitWeights());
			network.setMomentum(getMomentum());
			
			network.printTopology();
			drawingPanel.addNodes(networkLayers);
		}
    	
		// if we were previously doing a test, we need to reset plot counter
		// in this way the plotter will start at 0 again.
		if (!training) {
			plot.resetCounter();
			training = true;
		}
		
		// now start train the network as a thread
    	runThread = new Runner(this, network);
    	runThread.setTrain(true);
    	runThread.start();
    }
    
    /************************************
     * startTest
     *---------------------------------------
     * test the netwerk
     */
    private void startTest() {
    	// make sure that we have a network to test
    	if (network == null) {
    		Log("Cannot test without training first!");
    	} else {
    		// if we were previously doing a training, we need to reset the plot counter
    		if (training) {
    			plot.resetCounter();
    			training = false;
    		}
    		
    		// run the test in a thread
    		runThread = new Runner(this, network);
        	runThread.setTrain(false);
        	runThread.start();
    	}
    }
    
    /************************************
     * enableAll(boolean enable)
     *---------------------------------------
     * enable/disable all the items in the GUI.
     * used when pressing start (disable) or stop (enable)
     */
    public void enableAll(boolean enable) {
    	topologyNodes[1].setEnabled(enable);
    	
    	algorithmCombo.setEnabled(enable);
    	momentumTxt.setEnabled(enable);
    	learningRateTxt.setEnabled(enable);
    	weightBoundaries.setEnabled(enable);
    	learningMethodCombo.setEnabled(enable);
    	runMethodCombo.setEnabled(enable);
    	nrExamples.setEnabled(enable);
    	nrRuns.setEnabled(enable);
    	imageFileButton.setEnabled(enable);
    	labelFileButton.setEnabled(enable);
    	
    	imageProcess[0].setEnabled(enable);
    	imageProcess[1].setEnabled(enable);
    	
    	resetButton.setEnabled(enable);
    }

    /************************************
     * getStepSize()
     *---------------------------------------
     * returns step size
     */
    public int getStepSize() {
    	return stepSize;
    }

    /************************************
     * getNetworkLayers()
     *---------------------------------------
     * an algorithm to translate the nodes in the GUI to an array of nodes
     */
    public int[] getNetworkLayers() {
    	int nodes[];
    	//regular expression to ignore spaces
    	String hiddens[] = topologyNodes[1].getText().split("\\s+");

    	// make sure an empty string is treated well
    	int numHiddens = (hiddens.length == 1 && hiddens[0].length() == 0) ? 0 : hiddens.length;
    	
    	// count layers
    	nodes = new int[numHiddens + 2];
    	nodes[0] = Integer.parseInt(topologyNodes[0].getText());
    	for (int i=0; i<numHiddens; i++) {
    			nodes[i+1] = Integer.parseInt(hiddens[i]);
    	}
    	
    	// add output layer
    	nodes[nodes.length-1] = Integer.parseInt(topologyNodes[2].getText());
    	
    	return nodes;
    }
    
    // the rest of the getters are used to process information from the GUI
    public double getMomentum() {
    	return Double.parseDouble(momentumTxt.getText());
    }
    
    public double getLearningRate() {
    	return Double.parseDouble(learningRateTxt.getText());
    }
    
    public double[] getInitWeights() {
    	String[] weights = weightBoundaries.getText().split("\\s*,\\s*");
    	double[] weightsNum = { Double.parseDouble(weights[0]), Double.parseDouble(weights[1]) };
    	
    	return weightsNum;
    }
    
    public String[] getFiles() {
    	String files[] = { imageFile.getText(), labelFile.getText() };
    	return files;
    }
    
    public String getLearningMethod() {
    	return learningMethodCombo.getSelectedItem();
    }
    
    public int getExamples() {
    	return Integer.parseInt(nrExamples.getText());
    }
    
    public int getRuns() {
    	return Integer.parseInt(nrRuns.getText());
    }
    
    public boolean[] getPreProcessors() {
    	return new boolean[] { imageProcess[0].getState(), imageProcess[1].getState() }; 
    }

	public void paint( Graphics g ) { }
	
	/************************************
     * actionPerformed(ActionEvent evt)
     *---------------------------------------
     * action listener for the buttons.
     * pretty straightforward
     */
	public void actionPerformed(ActionEvent evt) {
		if (evt.getSource() == startButton) {
			if (startButton.getLabel() == "START") {
				enableAll(false);
				startButton.setLabel("STOP");
				if (runMethodCombo.getSelectedItem() == "Train") {
					startTraining();
				} else {
					startTest();
				}
			} else {
				enableAll(true);
				startButton.setLabel("START");
				runThread.interrupt();
			}			
		} else if (evt.getSource() == imageFileButton || evt.getSource() == labelFileButton) {			
			Frame parent = new Frame();
			FileDialog fd = new FileDialog(parent, "Please choose a file:",
			    FileDialog.LOAD);
			fd.setVisible(true);
			String selectedItem = fd.getFile();
			if (selectedItem == null) {
				// no file selected
			} else {
				File ffile = new File( fd.getDirectory() + File.separator +
				                     fd.getFile());
			    if (evt.getSource() == imageFileButton) {
			    	imageFile.setText(ffile.getAbsolutePath());
			    } else if (evt.getSource() == imageFileButton) {
			    	imageFile.setText(ffile.getAbsolutePath());
			    }
			}		
		} else if (evt.getSource() == resetButton) {
			network = null;
			plot.resetAll();
			results.setText("");
			drawingPanel.reset();
		}
	}

	/************************************
     * itemStateChanged(ItemEvent e)
     *---------------------------------------
     * action listener for the combo boxes.
     * also very straightforward
     */
	public void itemStateChanged(ItemEvent e) {
		if ((e.getSource() == imageProcess[0]) || (e.getSource() == imageProcess[1])) {
			int inputs = (imageProcess[0].getState()) ? 400 : 784;
			inputs = (imageProcess[1].getState()) ? inputs/2 : inputs;
			
			topologyNodes[0].setText(Integer.toString(inputs));
		} else if (e.getSource() == runMethodCombo) {
			if (runMethodCombo.getSelectedItem() == "Train") {
				imageFile.setText(FILE_TRAIN[0]);
				labelFile.setText(FILE_TRAIN[1]);
				nrExamples.setText("60000");
			} else if (runMethodCombo.getSelectedItem() == "Test") {
				imageFile.setText(FILE_TEST[0]);
				labelFile.setText(FILE_TEST[1]);
				nrExamples.setText("10000");
			} else {
				plotPanel = null;
				repaint();
			}
		}
	}
}