import java.awt.*;
import java.awt.event.*;
import java.util.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import javax.swing.filechooser.*;
import java.io.*;
class Matrix{
    public Dimension getDimensions(){
        return new Dimension(rows,cols);
    }

    public double getElement(int row,int col){
            return values[cols*(row-1)+col-1];

    }

    public void setElement(int row,int col,double value){
            values[cols*(row-1)+col-1]=value;

    }

    public double[] toArray(){
        return values;
    }



    public Matrix removeElement(){
        double[] temp=Arrays.copyOf(values,values.length-1);
        if (cols==1)
            return new Matrix(temp,rows-1,cols);
        else if (rows==1)
            return new Matrix(temp,rows,cols-1);
        else{
            System.out.println("Cannot remove");
            return null;
        }


    }

    public Matrix addElement(double ele){
        double[] temp=Arrays.copyOf(values,values.length+1);
        temp[values.length]=ele;
        if (cols==1)
            return new Matrix(temp,rows+1,cols);
        else if(rows==1)
            return new Matrix(temp,rows,cols+1);
        else{
            System.out.println("Cannot add");
            return null;
        }

    }


    public static Matrix multiply(Matrix a,Matrix b){
        Dimension dimA=a.getDimensions();
        Dimension dimB=b.getDimensions();
        double[] value=new double[dimA.width*dimB.height];
        int count=0;
        double sum;
        if (dimA.height!=dimB.width){
            System.out.println("Multiplication not possible!");
            return null;
        }
        else{

            for (int i=1;i<=dimA.width;++i){
                for (int j=1;j<=dimB.height;++j){
                    sum=0;
                    for (int k=1;k<=dimA.height;++k){

                        sum+=a.getElement(i,k)*b.getElement(k,j);
                    }
                    value[count++]=sum;
                }
            }
            return new Matrix(value,dimA.width,dimB.height);
        }
    }

    public static Matrix add(Matrix a,Matrix b,boolean subtract){
        Dimension d=a.getDimensions();
        Dimension d2=b.getDimensions();
        if (d.width!=d2.width||d.height!=d2.height){
             System.out.println("Addition not possible!");
             return null;
        }

        double[] value=new double[d.width*d.height];
        int count=0;
        for (int i=1;i<=d.width;++i){
            for (int j=1;j<=d.height;++j){
                if (subtract)
                    value[count++]=a.getElement(i,j)-b.getElement(i,j);
                else
                    value[count++]=a.getElement(i,j)+b.getElement(i,j);
            }
        }
        return new Matrix(value,d.width,d.height);
    }

    public static Matrix scalarMultiply(Matrix a,double scalar){
        Dimension d=a.getDimensions();
        double[] value=new double[d.width*d.height];
        int count=0;
        for (int i=1;i<=d.width;++i){
            for (int j=1;j<=d.height;++j){
                value[count++]=a.getElement(i,j)*scalar;
            }
        }
        return new Matrix(value,d.width,d.height);
    }


    public static Matrix transpose(Matrix a){
        Dimension dim=a.getDimensions();
        double values[]=new double[dim.width*dim.height];
        int count=0;
        for (int i=1;i<=dim.height;++i){
            for (int j=1;j<=dim.width;++j){
                values[count++]=a.getElement(j,i);
            }
        }
        return new Matrix(values,dim.height,dim.width);
    }


    public void display(JTextArea status){
        String t=new String();

        for (int i=0;i<=this.rows+1;++i){
            if (i==0)
                t+="\u250C";
            else if (i==this.rows+1)
                t+="\u2514";
            else
                t+="\u2502";
            for (int j=1;j<=this.cols;++j){
                if (i!=0&&i!=this.rows+1)
                    t+=String.format("%+-8.3f ",getElement(i,j));
                else
                    t+=String.format("%-8s "," ");
            }
            if (i==0)
                t+="\u2510";
            else if (i==this.rows+1)
                t+="\u2518";
            else
                t+="\u2502";

            t+="\n";
        }


        t+="\n";
        status.append(t);
    }



    public Matrix(double[] values,int rows,int cols){
        this.values=values;
        this.rows=rows;
        this.cols=cols;
    }

    public Matrix(double maxValue,int rows,int cols){
        this.rows=rows;
        this.cols=cols;
        values=new double[rows*cols];
        int count=0;
        for (int i=1;i<=rows;++i){
            for (int j=1;j<=cols;++j){
                values[count++]=Math.random()*(maxValue-Neural.MIN_INITIAL_WEIGHT)+Neural.MIN_INITIAL_WEIGHT;
            }
        }
    }

    private int rows;
    private int cols;
    private double[] values;

}

class Neural{

    public double applyFunction(double x){
        if (ACT_FUNC_TYPE==0){
            return ((2/(1+Math.exp(-LAMBDA*x)))-1);
        }else{
            return ((1/(1+Math.exp(-LAMBDA*x))));
        }
    }

    public double applyDerivative(double x){
        if (ACT_FUNC_TYPE==0){
            return LAMBDA*(1-Math.pow(applyFunction(x),2))/2;
        }else{
            return LAMBDA*applyFunction(x)*(1-applyFunction(x));
        }


    }




    class Layer{

        public Matrix getOutput(Matrix mInput){
            Matrix mNet=Matrix.multiply(mWeights,mInput);
            Dimension dimNet=mNet.getDimensions();
            for (int i=1;i<=dimNet.width;++i){
                for(int j=1;j<=dimNet.height;++j){
                    mNet.setElement(i,j,applyFunction(mNet.getElement(i,j)));
                }
            }

            return mNet;
        }

        public Matrix getDerivativeNet(Matrix mInput){
            Matrix mNet=Matrix.multiply(mWeights,mInput);
            Dimension dimNet=mNet.getDimensions();
            for (int i=1;i<=dimNet.width;++i){
                for(int j=1;j<=dimNet.height;++j){
                    mNet.setElement(i,j,applyDerivative(mNet.getElement(i,j)));
                }
            }

            return mNet;
        }

        public Matrix getWeights(){
            return mWeights;
        }

        public void setWeights(Matrix mWeights){
            this.mWeights=mWeights;
        }

        public int getNoNodes(){
            return noNodes;
        }


        public Layer(int layerDepth,int noNodes,Matrix mWeights){
            this.layerDepth=layerDepth;
            this.noNodes=noNodes;
            this.mWeights=mWeights;
        }

        int layerDepth;
        int noNodes;
        Matrix mWeights;
    }

    public int getNoNodes(int index){
        if (index<NO_LAYERS)
            return layers[index].getNoNodes();
        else
            return -1;
    }

    public void display(JTextArea status){
        status.append("PARAMETERS USED:-\n");
        status.append("-----------------------------------------\n");
        status.append("LEARNING RATE :"+this.LEARNING_RATE+"\n");
        status.append("\u03BB :"+this.LAMBDA+"\n");
        if (this.ACT_FUNC_TYPE==0)
            status.append("FUNCTION TYPE :BIPOLAR\n");
        else if(this.ACT_FUNC_TYPE==1)
            status.append("FUNCTION TYPE :UNIPOLAR\n");
        status.append("SCALING FACTOR :"+this.SCALING_FACTOR+"\n");
        status.append("MOMENTUM FACTOR :"+this.MOMENTUM_FACTOR+"\n");
        status.append("NUMBER OF INPUTS :"+this.NO_INPUTS+"\n");
        status.append("NUMBER OF LAYERS :"+this.NO_LAYERS+"\n");
        status.append("MAX ERROR :"+this.MAX_ERROR+"\n");
        status.append("NUMBER OF ITERATIONS :"+this.MAX_NO_ITERATIONS+"\n");
        status.append("MIN INITIAL WEIGHT :"+this.MIN_INITIAL_WEIGHT+"\n");
        status.append("MAX INITIAL WEIGHT :"+this.MAX_INITIAL_WEIGHT+"\n");
        status.append("WEIGHT STEP :"+this.INITIAL_WEIGHT_STEP+"\n");

        status.append("-----------------------------------------\n");

        for (int i=0;i<NO_LAYERS;++i){
            status.append(String.format("LAYER ="+(i+1)+", NO OF NODES ="+layers[i].getNoNodes())+"\n");
            Matrix m=layers[i].getWeights();
            m.display(status);
        }
        status.append("RMS ERROR :"+error+"\n");

    }

    public Matrix[] getOutput(Matrix mInput){
        Matrix[] layerOutputs=new Matrix[NO_LAYERS];
        layerOutputs[0]=layers[0].getOutput(mInput.addElement(-1.0));
        for (int i=1;i<NO_LAYERS;++i){
                layerOutputs[i]=layers[i].getOutput(layerOutputs[i-1].addElement(-1.0));
        }
        return layerOutputs;
    }

    public boolean simulate(Matrix[] testCases,int noCases,Matrix[] desiredOutputs,boolean isVerbose,JTextArea status){
        int iterations=0;
        error=0;
        double[] temp;
        int p=0;
        Matrix[] layerOutputs;
        Matrix[] prevWeightChange=new Matrix[this.NO_LAYERS];

        while (iterations<MAX_NO_ITERATIONS){
            while (p<noCases){
                layerOutputs=new Matrix[NO_LAYERS];
                layerOutputs[0]=layers[0].getOutput(testCases[p].addElement(-1.0));
                for (int i=1;i<NO_LAYERS;++i){
                    layerOutputs[i]=layers[i].getOutput(layerOutputs[i-1].addElement(-1.0));
                }


                Matrix errorSignal=Matrix.add(desiredOutputs[p],layerOutputs[NO_LAYERS-1],true);
                for (int i=1;i<=layers[NO_LAYERS-1].getNoNodes();++i){
                    error+=Math.pow(errorSignal.getElement(i,1),2);
                }

                Matrix mNet;
                Matrix weightChange;
                Matrix layerInput;
                for (int i=NO_LAYERS-1;i>=0;--i){
                    if (i>0)
                        layerInput=layerOutputs[i-1].addElement(-1.0);
                    else
                        layerInput=testCases[p].addElement(-1.0);

                    mNet=layers[i].getDerivativeNet(layerInput);

                    for (int j=1;j<=layers[i].getNoNodes();++j)
                        errorSignal.setElement(j,1,errorSignal.getElement(j,1)*mNet.getElement(j,1));

                    if (p==0&&iterations==0)
                        weightChange=Matrix.scalarMultiply(Matrix.multiply(errorSignal,Matrix.transpose(layerInput)),LEARNING_RATE);
                    else
                        weightChange=Matrix.add(Matrix.scalarMultiply(Matrix.multiply(errorSignal,Matrix.transpose(layerInput)),LEARNING_RATE),Matrix.scalarMultiply(prevWeightChange[i],MOMENTUM_FACTOR),false);
                    errorSignal=Matrix.multiply(Matrix.transpose(layers[i].getWeights()),errorSignal).removeElement();
                    layers[i].setWeights(Matrix.add(layers[i].getWeights(),weightChange,false));
                    prevWeightChange[i]=new Matrix(weightChange.toArray(),weightChange.getDimensions().width,weightChange.getDimensions().height);
                }
                ++p;
            }

            error=Math.sqrt(error)/(noCases*layers[NO_LAYERS-1].getNoNodes());

            if (error<MAX_ERROR){
                break;
            }
            else{
                p=0;
                if (iterations<MAX_NO_ITERATIONS-1)
                    error=0;
            }

            ++iterations;

        }
        if (isVerbose)
            status.append("After "+iterations+" Iterations, RMS Error ="+error+"\n");
        if (iterations<MAX_NO_ITERATIONS)
            return true;
        else
            return false;


    }





    public Neural(double maxInitialWeight){
        layers=new Layer[NO_LAYERS];
        int[] layerNodes=new int[NO_LAYERS];
        String[] nodes=NO_NODES.split(",");
        for (int i=0;i<NO_LAYERS;++i){
            layerNodes[i]=Integer.parseInt(nodes[i].trim());
        }
        layers[0]=new Layer(1,layerNodes[0],new Matrix(maxInitialWeight,layerNodes[0],NO_INPUTS+1));
        for (int i=1;i<NO_LAYERS;++i){
            layers[i]=new Layer(i+1,layerNodes[i],new Matrix(maxInitialWeight,layerNodes[i],layerNodes[i-1]+1));
        }

    }

    static double LAMBDA=1;
    static int ACT_FUNC_TYPE=0;
    static double MAX_ERROR=0.01;
    static double LEARNING_RATE=1;
    static int MAX_NO_ITERATIONS=2000;
    static double MIN_INITIAL_WEIGHT=0.1;
    static double MAX_INITIAL_WEIGHT=10;
    static double INITIAL_WEIGHT_STEP=0.1;
    static String NO_NODES="2,1";
    static double SCALING_FACTOR=1;
    static double MOMENTUM_FACTOR=0;

    static int NO_INPUTS=2;
    static int NO_LAYERS=2;
    Layer[] layers;
    double error;


}


class NeuralFrame extends JFrame{
    JPanel panel;
    JPanel parameters;
    JPanel bottomPanel;
    JTextArea status;
    Container container;
    JButton start;
    JButton stop;
    JButton clear;
    JButton fileButton;
    JButton tryMe;
    JCheckBox verbose;
    JProgressBar progress;
    JTextField inputTextField;
    Thread runThread;


    static int FRAMEWIDTH=650;
    static int FRAMEHEIGHT=380;
    static int SIDEPANELWIDTH=150;
    static int SIDEPANELCOMPONENTHEIGHT=20;
    static int SIDEPANELCOMPONENTBORDER=4;
    static int PANELBORDER=3;
    static int BOTTOMPANELHEIGHT=62;
    private boolean complete=false;
    private String input;
    private Neural n;
    private boolean isRunning=false;


    private File file;
    private Matrix[] testCases,desiredOutputs;
    private int noCases=0;


    private void update(){
        Component[] components =parameters.getComponents();
        for (int i=0;i<components.length;++i){
            try{
                if (i==0){
                        Neural.LEARNING_RATE=Double.parseDouble(((JTextField)(((Container)components[i]).getComponent(1))).getText());
                }
                if (i==1){
                        Neural.LAMBDA=Double.parseDouble(((JTextField)(((Container)components[i]).getComponent(1))).getText());
                }
                if (i==2){
                    int index=((JComboBox)((Container)components[i]).getComponent(1)).getSelectedIndex();
                    if (index==0)
                        Neural.ACT_FUNC_TYPE=0;
                    else if(index==1)
                        Neural.ACT_FUNC_TYPE=1;
                }
                if (i==3){
                        Neural.SCALING_FACTOR=Double.parseDouble(((JTextField)(((Container)components[i]).getComponent(1))).getText());
                }
                if (i==4){
                        Neural.MOMENTUM_FACTOR=Double.parseDouble(((JTextField)(((Container)components[i]).getComponent(1))).getText());
                }
                if (i==5)
                    Neural.NO_INPUTS=Integer.parseInt(((JTextField)(((Container)components[i]).getComponent(1))).getText());
                if (i==6)
                    Neural.NO_LAYERS=Integer.parseInt(((JTextField)(((Container)components[i]).getComponent(1))).getText());
                if (i==7)
                    Neural.NO_NODES=((JTextField)(((Container)components[i]).getComponent(1))).getText();
                if (i==8)
                    Neural.MAX_ERROR=Double.parseDouble(((JTextField)(((Container)components[i]).getComponent(1))).getText());
                if (i==9)
                    Neural.MAX_NO_ITERATIONS=Integer.parseInt(((JTextField)(((Container)components[i]).getComponent(1))).getText());

                if (i==10)
                    Neural.MIN_INITIAL_WEIGHT=Double.parseDouble(((JTextField)(((Container)components[i]).getComponent(1))).getText());

                if (i==11)
                    Neural.MAX_INITIAL_WEIGHT=Double.parseDouble(((JTextField)(((Container)components[i]).getComponent(1))).getText());
                if (i==12)
                    Neural.INITIAL_WEIGHT_STEP=Double.parseDouble(((JTextField)(((Container)components[i]).getComponent(1))).getText());
            }catch (NumberFormatException e){

                status.append("Error Parsing Value\n");
                if (i==0)
                        Neural.LEARNING_RATE=0;
                if (i==1)
                        Neural.LAMBDA=0;
                if (i==3)
                    Neural.SCALING_FACTOR=0;
                if (i==4)
                    Neural.MOMENTUM_FACTOR=0;
                if (i==5)
                    Neural.NO_INPUTS=0;
                if (i==6)
                    Neural.NO_LAYERS=0;
                if (i==7)
                    Neural.NO_NODES="";
                if (i==8)
                    Neural.MAX_ERROR=0;
                if (i==9)
                    Neural.MAX_NO_ITERATIONS=0;

                if (i==10)
                    Neural.MIN_INITIAL_WEIGHT=0;
                if (i==11)
                    Neural.MAX_INITIAL_WEIGHT=Neural.MIN_INITIAL_WEIGHT-1;
                if (i==12)
                    Neural.INITIAL_WEIGHT_STEP=0;
            }
            progress.setMinimum((int)Neural.MIN_INITIAL_WEIGHT);
            progress.setMaximum((int)Neural.MAX_INITIAL_WEIGHT);


        }
    }

    private void fileChooser(){
		JFileChooser chooser = new JFileChooser(System.getProperty("user.dir"));
		chooser.setDialogTitle("Choose Training Samples");
   		FileNameExtensionFilter filter = new FileNameExtensionFilter("Text Files", "txt");
    		chooser.setFileFilter(filter);
   		int returnVal = chooser.showOpenDialog(null);
  		if(returnVal == JFileChooser.APPROVE_OPTION) {
      			file=chooser.getSelectedFile();
   			if (file.getName().indexOf(".txt")==-1){
				status.append("Please select a .txt file\n");
				file=null;
			}
		}
	}

    class startThread implements Runnable{
        public void run(){
            isRunning=true;
            boolean isSuccess=false;
            complete=false;
            status.append("Training Process Started...\n");
            long time=System.currentTimeMillis();
            Matrix testCasesTemp[]=new Matrix[noCases];
            Matrix desiredOutputsTemp[]=new Matrix[noCases];
            for (int i=0;i<noCases;++i){
                testCasesTemp[i]=Matrix.scalarMultiply(testCases[i],Neural.SCALING_FACTOR);
                desiredOutputsTemp[i]=Matrix.scalarMultiply(desiredOutputs[i],Neural.SCALING_FACTOR);
            }
            boolean isVerbose=false;
            for (double i=Neural.MIN_INITIAL_WEIGHT;i<=Neural.MAX_INITIAL_WEIGHT;i+=Neural.INITIAL_WEIGHT_STEP){
                if(verbose.isSelected())
                    isVerbose=true;
                else
                    isVerbose=false;
                n=new Neural(i);
                progress.setValue((int)i);
                if (isVerbose)
                    status.append("Using Initial Weights Between "+Neural.MIN_INITIAL_WEIGHT+" - "+i+"\n");
                isSuccess=n.simulate(testCasesTemp,noCases,desiredOutputsTemp,isVerbose,status);
                if (isSuccess){
                    status.append("Training Process Completed in "+((System.currentTimeMillis()-time))+" ms...\n");
                    n.display(status);
                    complete=true;
                    break;
                }
                if (!isRunning)
                    break;
            }

            if (!isSuccess)
                status.append("Training Process Unsuccessful...\n");
            isRunning=false;
            disableControls(true);
            progress.setValue((int)Neural.MIN_INITIAL_WEIGHT);
        }
	}

	public void disableControls(boolean enable){
        fileButton.setEnabled(enable);
        tryMe.setEnabled(enable);
        start.setEnabled(enable);
        Component[] components=parameters.getComponents();
        for (int i=0;i<components.length;++i){
            ((JComponent)(((Container)components[i]).getComponent(1))).setEnabled(enable);
        }
	}

	public void start(){
        if (Neural.LEARNING_RATE<=0)
            status.append("Invalid Learning Rate Parameter\n");
        else if (Neural.LAMBDA<=0)
            status.append("Invalid \u03BB Parameter\n");
        else if (Neural.SCALING_FACTOR<=0)
            status.append("Invalid Scaling Factor\n");
        else if (Neural.NO_INPUTS<=0)
            status.append("Invalid Number of Inputs\n");
        else if (Neural.NO_LAYERS<=0)
            status.append("Invalid Number of Layers\n");
        else if (Neural.NO_NODES==""||Neural.NO_NODES.split(",").length!=Neural.NO_LAYERS)
            status.append("Invalid Number of Nodes\n");
        else if (Neural.MAX_ERROR<0)
            status.append("Invalid Maximum Error\n");
        else if (Neural.MAX_NO_ITERATIONS<=0)
            status.append("Invalid Number Of Iterations\n");
        else if (Neural.MIN_INITIAL_WEIGHT>Neural.MAX_INITIAL_WEIGHT)
            status.append("Invalid Initial Weights\n");
        else if (Neural.INITIAL_WEIGHT_STEP<=0)
            status.append("Invalid Initial Weight Step\n");
        else if (file==null)
            status.append("No Training Samples Loaded\n");
        else{
            boolean nodePass=true;
            String[] nodes=Neural.NO_NODES.split(",");
            for (int i=0;i<Neural.NO_LAYERS;++i){
                if(Integer.parseInt(nodes[i].trim())<=0){
                    nodePass=false;
                    break;
                }

            }

            if (!nodePass)
                status.append("Invalid Node found\n");
            else{

                boolean inputPass=true;
                int currentInputLength=Neural.NO_INPUTS;

                for (int i=0;i<testCases.length;++i){
                    if (currentInputLength!=testCases[i].getDimensions().width){
                        inputPass=false;
                        break;
                    }


                }

                if (!inputPass)
                    status.append("Invalid Test Case found\n");
                else{
                    boolean outputPass=true;
                    int currentOutputLength=Integer.parseInt(Neural.NO_NODES.split(",")[Neural.NO_LAYERS-1].trim());
                    int maxValue=1;
                    int minValue;
                    if (Neural.ACT_FUNC_TYPE==0)
                        minValue=-1;
                    else
                        minValue=0;
                    for (int i=0;i<desiredOutputs.length;++i){
                        if (currentOutputLength!=desiredOutputs[i].getDimensions().width){
                            outputPass=false;
                            break;
                        }

                        double[] value=desiredOutputs[i].toArray();
                        for(int j=0;j<value.length;++j){
                            if ((value[j]*Neural.SCALING_FACTOR)>maxValue||(value[j]*Neural.SCALING_FACTOR)<minValue){
                                outputPass=false;
                                break;
                            }
                        }

                    }
                    if (!outputPass)
                        status.append("Invalid Desired Output found\n");
                    else{
                        Thread runThread=new Thread(new startThread());
                        disableControls(false);
                        runThread.start();

                    }
                }
            }


        }
	}







    public JPanel createField(String text,String value){
        JPanel textPanel=new JPanel();
        textPanel.setLayout(new BoxLayout(textPanel,BoxLayout.X_AXIS));
        JLabel label=new JLabel(text+" : ");
        JTextField field=new JTextField(value);
        textPanel.add(label);
        textPanel.add(field);
        textPanel.setAlignmentX(0);
        textPanel.setMaximumSize(new Dimension(SIDEPANELWIDTH,SIDEPANELCOMPONENTHEIGHT+2*SIDEPANELCOMPONENTBORDER));
        field.getDocument().addDocumentListener(new DocumentListener(){
										public void insertUpdate(DocumentEvent event){
											update();
										}
										public void removeUpdate(DocumentEvent event){
											update();
										}
										public void changedUpdate(DocumentEvent event){}
									});
        return textPanel;
    }

    public void initialize(){
        panel=new JPanel();
        panel.setLayout(new BorderLayout());
        parameters=new JPanel();
        parameters.setLayout(new BoxLayout(parameters,BoxLayout.Y_AXIS));
        parameters.setPreferredSize(new Dimension(SIDEPANELWIDTH,FRAMEHEIGHT-BOTTOMPANELHEIGHT-4*PANELBORDER));
        parameters.add(createField("Learning Rate(\u03B7)",String.valueOf(Neural.LEARNING_RATE)));
        parameters.add(createField("\u03BB",String.valueOf(Neural.LAMBDA)));
        JPanel comboPanel=new JPanel();
        comboPanel.setLayout(new BoxLayout(comboPanel,BoxLayout.X_AXIS));
        JComboBox combo=new JComboBox(new String[]{"BiPolar","Unipolar"});
        combo.addActionListener(new ActionListener(){
										public void actionPerformed(ActionEvent event){
											update();
										}
									});
        comboPanel.add(new JLabel("Type : "));
        comboPanel.add(combo);
        comboPanel.setAlignmentX(0);
        comboPanel.setMaximumSize(new Dimension(SIDEPANELWIDTH,SIDEPANELCOMPONENTHEIGHT+2*SIDEPANELCOMPONENTBORDER));
        parameters.add(comboPanel);
        parameters.add(createField("Scaling Factor",String.valueOf(Neural.SCALING_FACTOR)));
        parameters.add(createField("Momentum Factor",String.valueOf(Neural.MOMENTUM_FACTOR)));
        parameters.add(createField("No of Inputs",String.valueOf(Neural.NO_INPUTS)));
        parameters.add(createField("No of Layers",String.valueOf(Neural.NO_LAYERS)));
        parameters.add(createField("No of Nodes",Neural.NO_NODES));
        parameters.add(createField("Max RMS Error",String.valueOf(Neural.MAX_ERROR)));
        parameters.add(createField("Max Iterations",String.valueOf(Neural.MAX_NO_ITERATIONS)));
        parameters.add(createField("Min Initial Wgt",String.valueOf(Neural.MIN_INITIAL_WEIGHT)));
        parameters.add(createField("Max Initial Wgt",String.valueOf(Neural.MAX_INITIAL_WEIGHT)));
        parameters.add(createField("Initial Wgt Step",String.valueOf(Neural.INITIAL_WEIGHT_STEP)));

        parameters.setBorder(new EmptyBorder(PANELBORDER,PANELBORDER,PANELBORDER,PANELBORDER));

        status=new JTextArea();
        //status.setLineWrap(true);
        //status.setWrapStyleWord(true);
        status.setFont(new Font("Monospaced",Font.PLAIN,12));
        status.setEditable(false);
        status.setBackground(Color.BLACK);
        status.setForeground(Color.GREEN);
        status.append("\t-------------------------------------------------\n");
        status.append("\tArtificial Neural Networks Simulator version 1.0b\n");
        status.append("\t     using Error Back Propogation Method\n");
        status.append("\t                 Created By :-\n");
        status.append("\t                 NIJITH JACOB\n");
        status.append("\t                   \u00A9 2009\n");
        status.append("\t-------------------------------------------------\n");


        JScrollPane scrollPane=new JScrollPane(status);
        scrollPane.setPreferredSize(new Dimension(FRAMEWIDTH-SIDEPANELWIDTH-4*PANELBORDER,FRAMEHEIGHT-BOTTOMPANELHEIGHT-4*PANELBORDER));
        scrollPane.setBorder(new EmptyBorder(PANELBORDER,PANELBORDER,PANELBORDER,PANELBORDER));


        bottomPanel=new JPanel();
        bottomPanel.setBorder(new EmptyBorder(PANELBORDER,PANELBORDER,PANELBORDER,PANELBORDER));
        bottomPanel.setMaximumSize(new Dimension(FRAMEWIDTH-2*PANELBORDER,BOTTOMPANELHEIGHT));
        bottomPanel.setLayout(new BoxLayout(bottomPanel,BoxLayout.Y_AXIS));

        JPanel bottomPanel1=new JPanel();
        bottomPanel1.setMaximumSize(new Dimension(FRAMEWIDTH-2*PANELBORDER,BOTTOMPANELHEIGHT/2));
        start=new JButton("Start Training");
        verbose=new JCheckBox("Verbose",false);
        stop=new JButton("Stop Training");
        fileButton=new JButton("Training Samples");
        clear=new JButton("Clear");
        progress=new JProgressBar((int)Neural.MIN_INITIAL_WEIGHT,(int)Neural.MAX_INITIAL_WEIGHT);

        start.addActionListener(new ActionListener(){
										public void actionPerformed(ActionEvent event){
											start();
										}
									});
        stop.addActionListener(new ActionListener(){
										public void actionPerformed(ActionEvent event){
                                            disableControls(true);
                                            if (isRunning){
                                                status.append("Training Process Stopped\n");
                                                progress.setValue((int)Neural.MIN_INITIAL_WEIGHT);
                                                isRunning=false;
                                            }
                                            else
                                                status.append("Training Process Not Started\n");


										}
									});

        clear.addActionListener(new ActionListener(){
										public void actionPerformed(ActionEvent event){
											status.setText("");
										}
									});
        fileButton.addActionListener(new ActionListener(){
								public void actionPerformed (ActionEvent e){
									fileChooser();
									if (file!=null){
										try{
                                            noCases=0;
											Scanner in=new Scanner(new FileInputStream(file));
											ArrayList<Matrix> cases=new ArrayList<Matrix>();
											ArrayList<Matrix> desiredOutput=new ArrayList<Matrix>();

											double[] casesTemp;
											double[] desiredOutputTemp;
											String input;
											String[] split1;
											String[] split2,split3;
											Matrix temp;

											while(in.hasNextLine()){
                                                ++noCases;
                                                input=in.nextLine().trim();
                                                split1=input.split(":");
                                                split2=split1[0].split(",");
                                                split3=split1[1].split(",");
                                                casesTemp=new double[split2.length];
                                                desiredOutputTemp=new double[split3.length];
                                                for (int i=0;i<split2.length;++i)
                                                    casesTemp[i]=Double.parseDouble(split2[i].trim());
                                                for (int i=0;i<split3.length;++i)
                                                    desiredOutputTemp[i]=Double.parseDouble(split3[i].trim());

                                                cases.add(temp=new Matrix(casesTemp,split2.length,1));
                                                status.append("Added Input Sample "+noCases+"\n");
                                                temp.display(status);

                                                desiredOutput.add(temp=new Matrix(desiredOutputTemp,split3.length,1));
                                                status.append("Added Desired Output Sample "+noCases+"\n");
                                                temp.display(status);
											}
											testCases=new Matrix[noCases];
											desiredOutputs=new Matrix[noCases];
											for (int i=0;i<noCases;++i){
                                                testCases[i]=(Matrix)cases.get(i);
                                                desiredOutputs[i]=(Matrix)desiredOutput.get(i);
											}
                                            status.append(noCases+" Training Samples Added\n");
										}catch(Exception e1){
                                            status.append("Error In Parsing File\n");
                                            noCases=0;
                                            file=null;
                                            testCases=null;
                                            desiredOutputs=null;
										}
									}

								}
							});
        bottomPanel1.add(fileButton);
        bottomPanel1.add(start);
        bottomPanel1.add(stop);
        bottomPanel1.add(clear);
        bottomPanel1.add(verbose);

        JPanel bottomPanel2=new JPanel();
        bottomPanel2.setMaximumSize(new Dimension(FRAMEWIDTH-2*PANELBORDER,BOTTOMPANELHEIGHT/2));


        inputTextField=new JTextField(30);
        tryMe=new JButton("Try it!");

        inputTextField.getDocument().addDocumentListener(new DocumentListener(){
										public void insertUpdate(DocumentEvent event){
											input=inputTextField.getText();
										}
										public void removeUpdate(DocumentEvent event){
											input=inputTextField.getText();
										}
										public void changedUpdate(DocumentEvent event){}
									});

        tryMe.addActionListener(new ActionListener(){
										public void actionPerformed(ActionEvent event){
                                            try{
                                                if (complete){
                                                    if (input!=""&&input!=null){
                                                        String[] s=input.split(",");
                                                        if (s.length!=Neural.NO_INPUTS)
                                                            status.append("Invalid Input\n");
                                                        else{
                                                            Matrix mInput;
                                                            double[] inputFields=new double[s.length];
                                                            for (int i=0;i<s.length;++i){
                                                                inputFields[i]=Double.parseDouble(s[i].trim());
                                                            }
                                                            Matrix layerOutputs[]=n.getOutput(mInput=new Matrix(inputFields,s.length,1));
                                                            status.append("GIVEN INPUT:-\n");
                                                            status.append("-----------------------------------------\n");
                                                            mInput.display(status);
                                                            status.append("OUTPUT:-\n");
                                                            status.append("-----------------------------------------\n");
                                                            mInput=Matrix.scalarMultiply(mInput,Neural.SCALING_FACTOR);

                                                            for (int i=0;i<layerOutputs.length;++i){
                                                                status.append("LAYER ="+(i+1)+", NO OF NODES ="+n.getNoNodes(i)+"\n");
                                                                Matrix.scalarMultiply(layerOutputs[i],1/Neural.SCALING_FACTOR).display(status);
                                                            }



                                                        }
                                                    }
                                                    else
                                                        status.append("No Input Specified\n");
                                                }
                                                else
                                                    status.append("Neural Training Not Complete\n");


                                            }catch (NumberFormatException e){
                                                status.append("Error Parsing Input\n");
                                            }
										}


									});

        bottomPanel2.add(inputTextField);
        bottomPanel2.add(tryMe);
        bottomPanel2.add(progress);


        bottomPanel.add(bottomPanel1);
        bottomPanel.add(bottomPanel2);





        panel.add(parameters,BorderLayout.EAST);
        panel.add(scrollPane,BorderLayout.WEST);
        panel.add(bottomPanel,BorderLayout.SOUTH);
        container.add(panel);

    }

    public NeuralFrame(){
        this.setSize(FRAMEWIDTH,FRAMEHEIGHT);
        this.setResizable(false);
        this.setTitle("Artificial Neural Networks Simulator v1.0b");
        this.setDefaultCloseOperation(EXIT_ON_CLOSE);
        container=this.getContentPane();
        initialize();
    }
}


class Main{


    public static void main(String[] args){

        NeuralFrame frame=new NeuralFrame();
        frame.setVisible(true);


/*
        for (double i=Neural.MIN_INITIAL_WEIGHT;i<=Neural.MAX_INITIAL_WEIGHT;i+=Neural.INITIAL_WEIGHT_STEP){
            n=new Neural(2,2,new int[] {2,1},i);
            isSuccess=n.simulate(testCases,4,desiredOutputs,false);
            if (isSuccess){
                n.display();
                break;
            }
        }
        */


	}

}

