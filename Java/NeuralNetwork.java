import java.util.*;
import java.lang.reflect.Array;
import java.util.stream.DoubleStream;

abstract class ActivationFunc{
    abstract double eval(double x);
    double[] eval (double[] X){
        double[] y = new double[X.length];
        for (int i = 0; i < X.length; i++){
            y[i] = eval(X[i]);
        }
        return y;
    }

    abstract double prime(double x);
    double[] prime (double[] X){
        double[] y = new double[X.length];
        for (int i = 0; i < X.length; i++){
            y[i] = prime(X[i]);
        }
        return y;
    }
}

class None extends ActivationFunc{
    public double eval(double x){
        return x;
    }

    public double prime(double x) {
        return 1;
    }
}

class Sigmoid extends ActivationFunc{
    public double eval(double x){
        return 1/(1+Math.exp(-x));
    }

    public double prime(double x) {
        double result = eval(x);
        return result * (1-result);
    }
}

class ReLU extends ActivationFunc{
    public double eval(double x){
        return Math.max(0,x);
    }

    public double prime(double x) {
        return (x>0) ? 1 : 0;
    }
}

class LeckyReLU extends ActivationFunc{
    double alpha;
    LeckyReLU(double alpha){
        this.alpha = alpha;
    }

    public double eval(double x){
        return Math.max(alpha * x, x);
    }

    public double prime(double x) {
        return (x>0) ? 1 : alpha;
    }
}

class ELU extends ActivationFunc{
    double alpha;
    ELU(double alpha){
        this.alpha = alpha;
    }

    public double eval(double x){
        return Math.max(0,alpha * (Math.exp(x) - 1));
    }

    public double prime(double x) {
        return (x>0) ? 1 : alpha * Math.exp(x);
    }
}

interface LossFunc{
    double eval(double[] y, double[] yHat);
    double[] prime(double[] y, double[] yHat);
}

class MSE implements LossFunc{
    public double eval(double[] y, double[] yHat) {
        double result = 0;
        for (int i = 0; i < y.length; i++){
            result += Math.pow(y[i] - yHat[i], 2);
        }
        return result / (2 * y.length);
    }

    public double[] prime(double[] y, double[] yHat) {
        double[] result = new double[y.length];
        for (int i = 0; i < y.length; i++){
            result[i] = -(y[i] - yHat[i]) / y.length;
        }
        return result;
    }
}

class Layer{
    private final int size;
    private Integer inputSize;
    private final ActivationFunc activation;
    private double[][] weight;
    private double[] bias;

    Layer(int size){
        this.size = size;
        inputSize = null;
        this.activation = new None();
        bias = new Random().doubles(size, -1, 1).toArray();
        weight = new double[size][];
    }
    Layer(int size, ActivationFunc activation){
        this.size = size;
        inputSize = null;
        this.activation = activation;
        bias = new Random().doubles(size, -1, 1).toArray();
        weight = new double[size][];
    }

    public int getSize(){
        return size;
    }

    public double[] eval(double[] X){
        if (inputSize == null){ // 최초 연결
            inputSize = X.length;
            for (int i = 0; i < size; i++){
                weight[i] = new Random().doubles(inputSize).toArray();
            }
        }
        double[] Z = new double[size];
        for (int i = 0; i < size; i++){
            Z[i] = bias[i];
            for (int j = 0; j < inputSize; j++)
            {
                Z[i] += weight[i][j] * X[j];
            }
        }
        return activation.eval(Z);
    }

    public double[] feedback(double[] X, double[] dEdY, double l){
        double[] Z = new double[size];
        for (int i = 0; i < size; i++){ // Z = W * X + b
            Z[i] = bias[i];
            for (int j = 0; j < inputSize; j++)
            {
                Z[i] += weight[i][j] * X[j];
            }
        }

        double[] dYdZ = activation.prime(Z);
        double[] dZdW = X;

        double[] dEdX = new double[inputSize];
        for (int i = 0; i < size; i++){
            for (int j = 0; j < inputSize; j++){
                dEdX[j] += dEdY[i] * dYdZ[i] * weight[i][j];
            }
        }
        for (int i = 0; i < size; i++){
            bias[i] -= l * dEdY[i] * dYdZ[i];
            for (int j = 0; j < inputSize; j++)
            {
                weight[i][j] -= l * dEdY[i] * dYdZ[i] * dZdW[j];
            }
        }
        return dEdX;
    }
}

abstract class NeuralNetwork {
    protected Layer[] layers;
    private LossFunc loss;
    private double learningRate;

    public double[] eval(double[] x){
        for (Layer layer: layers){
            x = layer.eval(x);
        }
        return x;
    }

    public void compile(LossFunc loss, double learningRate){
        this.loss = loss;
        this.learningRate = learningRate;
    }

    public void fit(double[][] x_train, double[][] y_train){
        for (int dataIndex = 0; dataIndex < y_train.length; dataIndex++){
            feedback(x_train[dataIndex], y_train[dataIndex]);
        }
    }
    public void feedback(double[] x_train, double[] y_train){
        double[][] Xs = new double[layers.length][];
        double[] YHat = new double[layers[layers.length-1].getSize()];
        Xs[0] = x_train;
        for (int i = 0; i < layers.length; i++){
            if (i == layers.length - 1){
                YHat = layers[i].eval(Xs[i]);
            } else {
                Xs[i+1] = layers[i].eval(Xs[i]);
            }
        }

        System.out.println(loss.eval(y_train, YHat));
        double[] dEdY = loss.prime(y_train, YHat);
        for (int i = layers.length - 1; i > -1; i--){
            dEdY = layers[i].feedback(Xs[i], dEdY, learningRate);
        }
    }
}

class Model extends NeuralNetwork {
    Model(){
        layers = new Layer[]{new Layer(20, new ReLU()), new Layer(20, new ReLU()), new Layer(5, new Sigmoid())};
    }
}
