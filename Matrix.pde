float sigmoid(float x) {
  return 1/(1+exp(-x));
}
Matrix toMatrix(float[] inputs) {
  Matrix ip = new Matrix(inputs.length, 1);
  for (int i = 0; i < ip.rows; i++) {
    ip.mat[i][0] = inputs[i];
  }
  return ip;
}

float dsigmoid(float ysigmoided) {
  return ysigmoided * (1 - ysigmoided);
}
class Matrix implements Serializable {
  int rows, cols;
  float[][] mat;
  Matrix(int r_, int c_) {
    rows = r_;
    cols = c_;

    mat = new float[rows][cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        mat[i][j] = 0;
      }
    }
  }

  void applySig() {
    for (int i = 0; i < rows; i++) {
      for (int j =0; j < cols; j++) {
        mat[i][j] = sigmoid(mat[i][j]);
      }
    }
  }

  void applyDSig() {
    for (int i = 0; i < rows; i++) {
      for (int j =0; j < cols; j++) {
        mat[i][j] = dsigmoid(mat[i][j]);
      }
    }
  }

  float[] toArray() {
    int count = 0;
    float[] a = new float[rows * cols];
    for (int i = 0; i < rows; i++) {
      for (int j =0; j < cols; j++) {
        a[count] = mat[i][j];
        count++;
      }
    }
    return a;
  }

  void matMult(float n) {
    for (int i = 0; i < rows; i++) {
      for (int j =0; j < cols; j++) {
        mat[i][j] *= n;
      }
    }
  }

  Matrix hadMult(Matrix m) {
    Matrix res = new Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j =0; j < cols; j++) {
        res.mat[i][j] = mat[i][j] * m.mat[i][j];
      }
    }
    return res;
  }

  void matAdd(float n) {
    for (int i = 0; i < rows; i++) {
      for (int j =0; j < cols; j++) {
        mat[i][j] += n;
      }
    }
  }

  Matrix matSub(Matrix b) {
    Matrix tmp = new Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j =0; j < cols; j++) {
        tmp.mat[i][j] = mat[i][j] - b.mat[i][j];
      }
    }
    return tmp;
  }

  Matrix matMult(Matrix n) {
    if (n.rows == cols) {
      Matrix res = new Matrix(rows, n.cols);
      for (int i = 0; i < res.rows; i++) {
        for (int j = 0; j < res.cols; j++) {
          float sum = 0;
          for (int k = 0; k < n.rows; k++) {
            sum += mat[i][k] * n.mat[k][j];
          }
          res.mat[i][j] = sum;
        }
      }
      return res;
    } else {
      println("Cant multiply!!!");
      return this;
    }
  }

  Matrix matTrans() {
    Matrix result = new Matrix(cols, rows);
    for (int i = 0; i < rows; i++) {
      for (int j =0; j < cols; j++) {
        result.mat[j][i] = mat[i][j];
      }
    }
    return result;
  }

  void matAdd(Matrix n) {
    if (n.rows == rows && n.cols == cols) {
      for (int i = 0; i < rows; i++) {
        for (int j =0; j < cols; j++) {
          mat[i][j] += n.mat[i][j];
        }
      }
    }
  }

  void matRand() {
    for (int i = 0; i < rows; i++) {
      for (int j =0; j < cols; j++) {
        mat[i][j] = (float) Math.random() * 2 - 1;
      }
    }
  }

  void printMat() {
    for (int i = 0; i < rows; i++) {
      for (int j =0; j < cols; j++) {
        print(mat[i][j]);
        print(" ");
      }
      println();
    }
    println();
  }

  void mutateMat(float mr) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (random(1) > mr) {
          mat[i][j] += randomGaussian() * 0.1;
        }
      }
    }
  }
  Matrix copy()
  {
    Matrix result = new Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.mat[i][j] = mat[i][j];
      }
    }
    return result;
  }
}
