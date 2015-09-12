package nl.marcvanzee.neuralnetwork.elements;

//Marc van Zee (F093385)   Creating a Neural Network
//http://www.marcvanzee.nl    marcvanzee@hotmail.com
//University Utrecht         Artificial Intelligence

/*********************************
* LineLayer.java
* 
* The LineLayer is simply a collection of lines.
* The size is static (not dynamic like a VectorList), because of efficiency consideration.
* Please read FeedForward.java for more details on this subject.
*/

public class LineLayer {
	private Line[] lines;
	
	public LineLayer(int length) {
		lines = new Line[length];
	}
	
	public void add(Line line, int index) {
		lines[index] = line;
	}
	
	public Line get(int i) {
		return lines[i];
	}
	
	public int size() {
		return lines.length;
	}
}
