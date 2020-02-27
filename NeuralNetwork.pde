import java.util.*;
import java.io.*;
import java.io.Serializable.*;

class NN implements Serializable{
  int I, H, O, NoH, total;
  float lr;

  Matrix[] weights, biases;

  NN(int I_, int H_, int O_, int NoH_) {
    I = I_;
    O = O_;
    H = H_;
    NoH = NoH_;
    total = NoH + 2;
    lr = 0.1;

    weights = new Matrix[total - 1];
    biases = new Matrix[total - 1];

    weights[0] = new Matrix(H, I);
    weights[weights.length - 1] = new Matrix(O, H);

    for (int i = 1; i < weights.length - 1; i++) {
      weights[i] = new Matrix(H, H);
    }

    biases[0] = new Matrix(H, 1);
    biases[biases.length - 1] = new Matrix(O, 1);
    for (int i = 1; i < biases.length - 1; i++) {
      biases[i] = new Matrix(H, 1);
    }

    for (int i = 0; i < weights.length; i++) {
      weights[i].matRand();
    }

    for (int i = 0; i < biases.length; i++) {
      biases[i].matRand();
    }
  }

  NN(NN copy) {
    copy.I = I;
    copy.O = O;
    copy.H = H;
    copy.NoH = NoH;
    copy.total = total;
    copy.lr = lr;

    for (int i = 0; i < copy.weights.length - 1; i++) {
      copy.weights[i] = weights[i].copy();
      copy.biases[i] = biases[i].copy();
    }
  }

  float[] predict(float[] inputs) {
    Matrix ip = toMatrix(inputs);

    Matrix[] weightedSums = new Matrix[total];

    weightedSums[0] = ip;

    for (int i = 1; i < weightedSums.length; i++) {
      weightedSums[i] = weights[i - 1].matMult(weightedSums[i-1]);
      weightedSums[i].matAdd(biases[i - 1]);
      weightedSums[i].applySig();
    }

    Matrix op = weightedSums[weightedSums.length - 1];
    return op.toArray();
  }

  void train(float[] inputs, float[] target) {
    Matrix ip = toMatrix(inputs);

    Matrix[] weightedSums = new Matrix[total];

    weightedSums[0] = ip;

    for (int i = 1; i < weightedSums.length; i++) {
      weightedSums[i] = weights[i - 1].matMult(weightedSums[i-1]);
      weightedSums[i].matAdd(biases[i - 1]);
      weightedSums[i].applySig();
    }

    Matrix op = weightedSums[weightedSums.length - 1];

    Matrix expected = toMatrix(target);

    Matrix[] errors = new Matrix[total - 1];

    errors[errors.length-1] = expected.matSub(op);

    for (int i = errors.length - 2; i >= 0; i--) {
      Matrix tmpT = weights[i+1].matTrans();
      errors[i] = tmpT.matMult(errors[i+1]);
    }

    Matrix[] DSigWeightedSums = new Matrix[total];
    for (int i = 0; i < weightedSums.length; i++) {
      DSigWeightedSums[i] = weightedSums[i].copy();
      DSigWeightedSums[i].applyDSig();
    }

    Matrix[] weights_deltas = new Matrix[total - 1];
    Matrix[] biases_deltas = new Matrix[total - 1];

    for (int i = biases_deltas.length - 1; i >= 0; i--) {
      biases_deltas[i] = DSigWeightedSums[i+1].hadMult(errors[i]);
      biases_deltas[i].matMult(lr);
    }
    for (int i = weights_deltas.length - 1; i >= 0; i--) {
      Matrix trans = weightedSums[i].matTrans();
      weights_deltas[i] = biases_deltas[i].matMult(trans);
    }

    for (int i = 0; i < weights.length; i++) {
      weights[i].matAdd(weights_deltas[i]);
      biases[i].matAdd(biases_deltas[i]);
    }
  }
  NN copyNN() {
    return new NN(this);
  }

  void mutate(float mr) {
    for (int i = 0; i < weights.length; i++) {
      weights[i].mutateMat(mr);
      biases[i].mutateMat(mr);
    }
  }

  void saveData(File file) {
    try {
      FileOutputStream fos = new FileOutputStream(file);
      ObjectOutputStream oos = new ObjectOutputStream(fos);
      oos.writeObject(this);
      fos.close();
    }
    catch (IOException e) {
      e.printStackTrace();
    }
  }

  NN loadData(File file) {
    NN savedBrain = null;
    try {
      FileInputStream fis = new FileInputStream(file);
      ObjectInputStream ois = new ObjectInputStream(fis);
      savedBrain = (NN) ois.readObject();
      fis.close();
    }
    catch (IOException e) {
      e.printStackTrace();
    }
    catch (ClassNotFoundException e) {
      e.printStackTrace();
    }
    return savedBrain;
  }
}
