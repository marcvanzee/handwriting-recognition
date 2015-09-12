package nl.marcvanzee.opdr4;

//Marc van Zee (F093385)   Creating a Neural Network
//http://www.marcvanzee.nl    marcvanzee@hotmail.com
//University Utrecht         Artificial Intelligence

/*********************************
 * Plotter.java
 * 
 * The plot is an instance of the PTPlot Java Plotter
 * http://ptolemy.eecs.berkeley.edu/java/ptplot5.7/ptolemy/plot/doc/index.htm
 * 
 * The plot will auto adjust to the domain of the current values, and the user can zoom by click-drag-releasing
 * on the plot.
 */

import ptolemy.plot.PlotLive;

public class Plotter extends PlotLive {
	private static final long serialVersionUID = 1L;
	
	// variable to determine how many examples (x-axis) we've had already
	private int counter;
	
	// how much the x-axis should increase at every step
	private int stepSize;
	
	public Plotter(int stepSize) {
		resetAll();
		this.setXLabel("examples");
		this.setYLabel("error rate");
		
		// start at a range of 0 to 300 examples
		this.setXRange(0, 300);
		this.stepSize = stepSize;

		this.addLegend(0, "training");
		this.addLegend(1, "test");		
	}
	
	public void addToPlot(double value, int dataset) {
		// automatically zoom out 200% when the x-axis reaches its maximum
		if (counter >= this.getXRange()[1]) {
			this.zoom(0, 0, counter*2, 1);
		}
		addPoint(dataset, counter, value, true);
		counter += stepSize;
	}

	public void addPoints() {
		
	}
	
	public void resetAll() {
		counter = 0;
		this.setYRange(0, 1);
		clear(0);
		clear(1);
	}
	
	public void resetCounter() {
		counter = 0;
	}
}
