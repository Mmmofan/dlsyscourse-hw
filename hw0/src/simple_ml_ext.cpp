#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void naive_gemm(const float *X, const float *Y, float *Z,
                size_t hx, size_t hy, size_t M, size_t N, size_t K,
                bool transpose=false)
{
  /*
  X: pointer of X, stored in row major, start from hx, of size m*n
  Y: pointer of Y, stored in row major, start from hy, of size n*k
  Z: pointer of Z, result, of size m*k
  */
  for (size_t m=0; m < M; ++m)
  {
    for (size_t k=0; k < K; ++k)
    {
      size_t l = m * K + k;
      Z[l] = 0.0;
      for (size_t n=0; n < N; ++n)
      {
        size_t i, j;
        if (!transpose) { // row-major
          i = hx + m * N + n;
          j = hy + n * K + k;
        } else { // transpose X
          i = hx + m + n * M;
          j = hy + n * K + k;
        }
        Z[l] += X[i] * Y[j];
      }
    }
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    float *dw = new float[n * k]; // derivative of theta, of size [n, k] (same as theta)
    float *x_ = new float[batch * k]; // = input @ theta ([batch*n] @ [n*k] = [batch * k])
    float *y_ = new float[batch * k]; // one-hot label, of size [batch, k]
    float *probs = new float[batch * k]; // probabilities (logits), of size [batch, k]

    for (size_t i = 0; i < m; i+=batch) {
        // init x_, y_, probs, dw
        for (size_t j = 0; j < batch * k; ++j) {
          y_[j] = 0.f;
          x_[j] = 0.f;
          probs[j] = 0.f;
        }
        // offsets
        size_t offset_x = i * n;
        size_t offset_y = i;
        // compute X @ theta
        naive_gemm(X, theta, x_, offset_x, 0, batch, n, k);

        // compute one-hot label y_
        for (size_t b = 0; b < batch; ++b) {
          int cls = (int)y[offset_y + b];
          y_[b * k + cls] = 1.f; // as positive
        }

        // compute softmax score of output
        for (size_t b = 0; b < batch; ++b) {
          float sum = 0.f;
          for (size_t i = 0; i < k; ++i) {
            float cur = std::exp(x_[b * k + i]);
            probs[b * k + i] = cur;
            sum += cur;
          }
          for (size_t i = 0; i < k; ++i) {
            probs[b * k + i] /= sum;
          }
        }

        // compute y_ - probs, set to y_
        for (size_t i = 0; i < batch * k; ++i) {
          y_[i] -= probs[i];
        }
        // compute dw
        naive_gemm(X, y_, dw, offset_x, 0, n, batch, k, true);
        for (size_t i = 0; i < n * k; ++i) {
          theta[i] += (lr / batch * dw[i]);
        }
    }
    delete[] x_;
    delete[] y_;
    delete[] probs;
    delete[] dw;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
