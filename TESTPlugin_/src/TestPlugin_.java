import java.util.Arrays;
import java.util.stream.Collectors;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.plugin.PlugIn;
import ij.plugin.frame.RoiManager;
import ij.text.TextPanel;

public class TestPlugin_ implements PlugIn{ 
	public void run (String arg) {
		ImagePlus imp = IJ.getImage();
		ij.IJ.run("Subtract Background...", "rolling=200 stack");
		ij.IJ.run("Median...", "radius=1 stack");
		ij.IJ.setAutoThreshold(imp,"Default dark");
		//run("Threshold...");
		ij.IJ.setThreshold(220, 65535);
		
//		ij.ImageStack.("BlackBackground", false);
		ij.IJ.run("Convert to Mask", "method=Default background=Dark");
		ij.IJ.run("Skeletonize", "stack");
		ij.IJ.setThreshold(255, 255);
		
		ij.IJ.run("Analyze Particles...", "size=10-Infinity display exclude clear add stack");
		
		RoiManager rm = RoiManager.getInstance2();
		
		TextPanel tp = new TextPanel("rois");
		
		for (Roi roi : rm.getRoisAsArray()) {			
			float[] xsFloat;
			xsFloat = roi.getContainedFloatPoints().xpoints;
			int[] xsInt;
			xsInt = new int[xsFloat.length];			
			int c = 0;
			for (float xs : xsFloat) {
				xsInt[c] = (int) xs;
				c++;
			}
			String xs_ = Arrays.stream(xsInt)
			        .mapToObj(String::valueOf)
			        .collect(Collectors.joining(","));
			tp.append(xs_);
			
			float[] ysFloat;
			ysFloat = roi.getContainedFloatPoints().ypoints;
			int[] ysInt;
			ysInt = new int[ysFloat.length];
			c = 0;
			for (float ys : ysFloat) {
				ysInt[c] = (int) ys;
				c++;
			}
			String ys_ = Arrays.stream(ysInt)
			        .mapToObj(String::valueOf)
			        .collect(Collectors.joining(","));
			tp.append(ys_);
			
			int z = roi.getZPosition();
			tp.append(Integer.valueOf(z).toString());	
        }
		tp.saveAs("E:/TIRF/210728_/roi.csv");
		
	}
}