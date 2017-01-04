package com.fshuai.Count.Test_Sift;

import mpi.cbg.fly.*;
import java.util.Collections;
import java.awt.Image;
import java.awt.image.PixelGrabber;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
//import org.apache.log4j.Logger;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

/**
 * The JavaSIFT class is a wrapper Class around the JavaSIFT implementation written by Stephan Saalfeld.
 * Its not very fast but plain Java and indenpendent from other libraries (you even don't need ImageJ).
 */
public class SIFT implements ImageFeatureExtractor,Serializable
{
	// steps
	private int steps = 3;
	// initial sigma
	private float initial_sigma = 1.6f;
	// background colour
	//private double bg = 0.0;
	// feature descriptor size
	private int fdsize = 4;
	// feature descriptor orientation bins
	private int fdbins = 8;
	// size restrictions for scale octaves, use octaves < max_size and > min_size only
	private int min_size = 64;
	private int max_size = 1024;
	/**
	 * Set true to double the size of the image by linear interpolation to
	 * ( with * 2 + 1 ) * ( height * 2 + 1 ).  Thus we can start identifying
	 * DoG extrema with $\sigma = INITIAL_SIGMA / 2$ like proposed by
	 * \citet{Lowe04}.
	 * 
	 * This is useful for images scmaller than 1000px per side only. 
	 */ 
	   private boolean upscale = true;
        private static float normTo1(int b) {
            return (float) (b / 255.0f);
        }
        
        private static int RGB2Grey(int argb) {
           // int a = (argb >> 24) & 0xff;
            int r = (argb >> 16) & 0xff;
            int g = (argb >> 8) & 0xff;
            int b = (argb) & 0xff;

            //int rgb=(0xff000000 | ((r<<16)&0xff0000) | ((g<<8)&0xff00) | (b&0xff));
            int y = (int) Math.round(0.299f * r + 0.587f * g + 0.114f * b);
            return y;
        }

        private FloatArray2D convert(RenderedImage img)
        {
            
            FloatArray2D image;
            PixelGrabber grabber=new PixelGrabber((Image) img, 0, 0, -1,-1, true);
            try {
                grabber.grabPixels();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            int[] data = (int[]) grabber.getPixels();
            
            image = new FloatArray2D(grabber.getWidth(),  grabber.getHeight());
            for (int d=0;d<data.length;d++)
                        image.data[d] = normTo1(RGB2Grey(data[d]));
            return image;
        }
        
        private List<ImageFeature> convert(List<Feature> features)
        {
            List<ImageFeature> res=new ArrayList<ImageFeature>();
            for (Feature f:features)
            {
                ImageFeature imageFeature=new ImageFeature();
                imageFeature.setDescriptor(( f.descriptor));
                imageFeature.setOrientation(f.orientation);
                imageFeature.setScale(f.scale);
                res.add(imageFeature);
            }
            return res;
        }
	
//        private float[] convert(float[] desc)
//        {
//            for (int i=0;i<desc.length;i++)
//            {
//               int int_val = (int)(512 * desc[i]);
//               int_val = Math.min( 255, int_val ); 
//               desc[i]=int_val;
//            }
//            return desc;
//        }
    public List<ImageFeature> getSift(Image img){
    	String preamb=this.getClass()+": ";
        List<Feature> fs;
	
        FloatArray2DSIFT sift = new FloatArray2DSIFT( fdsize, fdbins );		
        FloatArray2D fa = convert((RenderedImage)img);              
        Filter.enhance( fa, 1.0f );

        if ( upscale )
        {
        	FloatArray2D fat = new FloatArray2D( fa.width * 2 - 1, fa.height * 2 - 1 ); 
        	FloatArray2DScaleOctave.upsample( fa, fat );
        	fa = fat;
        	fa = Filter.computeGaussianFastMirror( fa, ( float )Math.sqrt( initial_sigma * initial_sigma - 1.0 ) );
        }
        else
        	fa = Filter.computeGaussianFastMirror( fa, ( float )Math.sqrt( initial_sigma * initial_sigma - 0.25 ) );

        	long start_time = System.currentTimeMillis();
        	System.out.println(preamb+"processing SIFT ..." );
        	sift.init( fa, steps, initial_sigma, min_size, max_size );
        	fs = sift.run( max_size );
        	Collections.sort( fs );
        	System.out.println(preamb+"took " + ( System.currentTimeMillis() - start_time ) + "ms" );		
        	System.out.println(preamb+ fs.size() + " features identified and processed" );     
        	return convert(fs);
    }
    
    public List<ImageFeature> getSift1(RenderedImage img){
    	String preamb=this.getClass()+": ";
        List<Feature> fs;
	
        FloatArray2DSIFT sift = new FloatArray2DSIFT( fdsize, fdbins );		
        FloatArray2D fa = convert(img);              
        Filter.enhance( fa, 1.0f );

        if ( upscale )
        {
        	FloatArray2D fat = new FloatArray2D( fa.width * 2 - 1, fa.height * 2 - 1 ); 
        	FloatArray2DScaleOctave.upsample( fa, fat );
        	fa = fat;
        	fa = Filter.computeGaussianFastMirror( fa, ( float )Math.sqrt( initial_sigma * initial_sigma - 1.0 ) );
        }
        else
        	fa = Filter.computeGaussianFastMirror( fa, ( float )Math.sqrt( initial_sigma * initial_sigma - 0.25 ) );

        	long start_time = System.currentTimeMillis();
        	System.out.println(preamb+"processing SIFT ..." );
        	sift.init( fa, steps, initial_sigma, min_size, max_size );
        	fs = sift.run( max_size );
        	Collections.sort( fs );
        	System.out.println(preamb+"took " + ( System.currentTimeMillis() - start_time ) + "ms" );		
        	System.out.println(preamb+ fs.size() + " features identified and processed" );     
        	return convert(fs);
    }
    
   

	public JavaRDD<List<ImageFeature>> getFeatures1(JavaRDD<RenderedImage> filepaths)
	{   
		return filepaths.map(new Function<RenderedImage,List<ImageFeature>>(){
			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			public List<ImageFeature> call(RenderedImage arg0) throws Exception {
				// TODO Auto-generated method stub
				List<ImageFeature> list=null;
				list=getSift1(arg0);
				return list;
			}
		});
	}
	public static JavaRDD<RenderedImage> getImage(String path,JavaSparkContext sc) throws IOException{
		//根据路径读取文件
		File file=new File(path);
		File[] tempList=file.listFiles();
		List<RenderedImage> imageList=new ArrayList<RenderedImage>();
		for(int i=0;i<tempList.length;i++){
			if(tempList[i].getName().endsWith(".jpg") || tempList[i].getName().endsWith(".png")){
				RenderedImage tmp=ImageIO.read(tempList[i]);
				imageList.add(tmp);
				//imageList.add(new RenderedImage(ImageIO.read(tempList[i])));
			}
		}
		JavaRDD<RenderedImage> result=sc.parallelize(imageList);
		return result;
	}
	/*public static JavaRDD<Image> getImage(String path,JavaSparkContext sc) throws IOException{
		//根据路径读取文件
		File file=new File(path);
		File[] tempList=file.listFiles();
		List<Image> imageList=new ArrayList<Image>();
		for(int i=0;i<tempList.length;i++){
			if(tempList[i].getName().endsWith(".jpg") || tempList[i].getName().endsWith(".png")){
				imageList.add(ImageIO.read(tempList[i]));
			}
		}
		JavaRDD<Image> result=sc.parallelize(imageList);
		return result;
	}*/

   public static void main(String args[]){
	   String[] jars={"Jama-1.0.2","Java_SIFT.jar"};
	   SparkConf conf=new SparkConf().setAppName("Simple Application");
	   JavaSparkContext sc=new JavaSparkContext(conf);
	   String filepath="/root/image/";//存放图片目录
	   JavaRDD<RenderedImage> images=null;
	   JavaRDD<List<ImageFeature>> points=null;
		try {
			//SeriImage im=new SeriImage(ImageIO.read(new File(filepath)));
			//new SIFT().getFeatures1(im);
			images = getImage(filepath, sc);
			images.cache();
			points=new SIFT().getFeatures1(images);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("len:"+images.count());
	   //JavaRDD<List<ImageFeature>> points=new SIFT().getFeatures1(images);
	   //System.out.println("content:"+points.take(0).indexOf(0));
   }

  

   public List<ImageFeature> getFeatures(Image img) {
	   // TODO Auto-generated method stub
	   return null;
   }
}