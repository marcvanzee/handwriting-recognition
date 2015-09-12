package nl.marcvanzee.opdr4;


import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Panel;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.util.Vector;

public class NetworkDrawing extends Panel {
	private static final long serialVersionUID = 1L;
	
	private int width, height;
	int layers[];
	private Point2D nodes[][];
	private Vector<Line2D> lines;
	private Vector<Integer> alpha;
	private Graphics2D g2;
	
	public void addNodes(int networkLayers[]) {		
		this.layers = networkLayers;
		repaint();
	}
	
	public void addLines(Vector<Line2D> lines, Vector<Integer> alpha) {
		this.lines = lines;
		this.alpha = alpha;

		repaint();
	}
	
	public void paint(Graphics g) {
		g2 = (Graphics2D) g;
		if (layers != null) {
			drawNodes();
			
			if ((lines != null) && (alpha != null)) {
				drawLines();
			}
		}
	}

	private void drawNodes() {
		this.width = getSize().width;
		this.height = getSize().height;
		
		int maxNodeLayer = 0;
		for (int i=0; i<layers.length; i++) {
			maxNodeLayer = (layers[i] > maxNodeLayer) ? layers[i] : maxNodeLayer;
		}
		
		nodes = new Point2D[layers.length][maxNodeLayer];
		
		int nrLayers = layers.length;
		double nodeStep, nodeSize, layerStep = height / (nrLayers - 1);
		int nrNodes;
		
		g2.setColor(Color.BLACK);
		
		for (int y=0; y<nrLayers; y++) {
			nrNodes = layers[y];
			nodeStep = (double)width / ((double)nrNodes + 1);
			nodeSize = (nodeStep > 10) ? 10 : (nodeStep-1);
			for (int x=0; x<nrNodes; x++) {
				double xCor = x*nodeStep+nodeStep;
				double yCor = y*layerStep;
				//if (y==0) yCor += nodeSize;
				if (y==nrLayers-1) yCor -= nodeSize;
				nodes[y][x] = new Point2D.Double(xCor, yCor);
				g2.draw(new Ellipse2D.Double(xCor, yCor, nodeSize, nodeSize));
			}
		}
	}

	private void drawLines() {
		for (int i=0; i<lines.size(); i++) {				
			g2.setColor(new Color(0, 0, 0, alpha.get(i)));
			g2.draw(lines.get(i));
		}
	}

	public void reset() {
		layers = null;
		lines = null;
		repaint();
	}
	
	public Point2D[][] getNodes() {
		return nodes;
	}
}
