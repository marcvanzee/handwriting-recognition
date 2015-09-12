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

import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.util.Random;
import java.util.Vector;

public class Drawer extends Thread {
	private NetworkDrawing drawPanel;
	private double weights[][];
	private int layers[];
	private Point2D nodes[][];
	
	private Vector<Line2D> lines = new Vector<Line2D>();
	private Vector<Integer> lineWeights = new Vector<Integer>();
	
	public Drawer(int layers[], double weights[][], NetworkDrawing drawPanel) {
		// find out who the parent is, and use the drawing of the parent
		this.drawPanel = drawPanel;
		this.layers = layers;
		this.weights = weights;
	}

	public void run() {
		int maxNodeLayer = 0;
		for (int i=0; i<layers.length; i++) {
			maxNodeLayer = (layers[i] > maxNodeLayer) ? layers[i] : maxNodeLayer;
		}
		
		nodes = drawPanel.getNodes();
							
		for (int i=1; i<layers.length; i++) {
			// find maxium weight of this layer
			double bestWeight = 0;

			for (int j=0; j<weights[i-1].length; j++) {
				if (weights[i-1][j] > bestWeight) bestWeight = weights[i-1][j];
			}
				
			for (int j=0; j<layers[i]; j++) {
				for (int k=0; k<layers[i-1]; k++) {
					if (weights[i-1][j*k+k] >= bestWeight*0.5) {
						double curWeight = weights[i-1][j*k+k];
						double min = bestWeight*0.5;
						double max = bestWeight;
						
						curWeight = (curWeight-min)/(max-min) * 255;
						int alpha = (int)curWeight;
						if (alpha > 255) alpha = 255;
						if (alpha < 0)   alpha = 0;
						
						lines.add(new Line2D.Double(nodes[i][j].getX(), nodes[i][j].getY(), nodes[i-1][k].getX(), nodes[i-1][k].getY()));
						lineWeights.add(alpha);
					}
				}
			}
		}
		Random gen = new Random();
		
		while (lines.size() > 1000) {
			int r = gen.nextInt(lines.size()-1);
			lines.remove(r);
			lineWeights.remove(r);
		}
		
		drawPanel.addLines(lines, lineWeights);
	}
}
