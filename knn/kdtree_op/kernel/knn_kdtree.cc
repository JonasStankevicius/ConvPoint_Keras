#include "tensorflow/core/framework/op_kernel.h"

#include <string>

#include "KDTreeTableAdaptor.h"
#include "knn_.h"

using namespace tensorflow;

class KnnKdtreeOp : public OpKernel {
 public:
  explicit KnnKdtreeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& nsampleTensor = context->input(0);
    const Tensor& ptsTensor = context->input(1);
    const Tensor& queriesTensor = context->input(2);

    const float* ptsBuffer = (float*)ptsTensor.data();
    const float* queriesBuffer = (float*)queriesTensor.data();

    int32 K = nsampleTensor.scalar<int32>()();

    // Create an output tensor
    Tensor* indiciesTensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, { ptsTensor.dim_size(0), queriesTensor.dim_size(1), K },
                                                     &indiciesTensor));

    auto indiciesFlat = indiciesTensor->flat<int64>();
    indiciesFlat.setZero();

    int64* indiciesBuffer = (int64*)indiciesTensor->data();

    std::string errorMsg;
    cpp_knn_batch_omp(ptsBuffer, ptsTensor.dim_size(0), ptsTensor.dim_size(1), ptsTensor.dim_size(2),
                queriesBuffer, queriesTensor.dim_size(1), K, (long long*)indiciesBuffer, errorMsg);

    OP_REQUIRES(context, errorMsg.empty(), Status(error::INTERNAL, errorMsg));
  }
};

REGISTER_KERNEL_BUILDER(Name("KnnKdtree").Device(DEVICE_CPU), KnnKdtreeOp);

class KnnKdtreeSamplerOp : public OpKernel {
  public:
  explicit KnnKdtreeSamplerOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& nsampleTensor = context->input(0);
    const Tensor& ptsTensor = context->input(1);
    const Tensor& nqueriesTensor = context->input(2);

    const float* ptsBuffer = (float*)ptsTensor.data();
    int32 K = nsampleTensor.scalar<int32>()();
    int32 nqueries = nqueriesTensor.scalar<int32>()();

    int64 batch_size = ptsTensor.dim_size(0);
    int64 npts = ptsTensor.dim_size(1);
    int64 dim = ptsTensor.dim_size(2);

    Tensor* indiciesTensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, { batch_size, nqueries, K }, &indiciesTensor));
    indiciesTensor->flat<int64>().setZero();

    Tensor* queriesTensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, { batch_size, nqueries, dim }, &queriesTensor));
    queriesTensor->flat<float>().setZero();

    int64* indiciesBuffer = (int64*)indiciesTensor->data();
    float* queriesBuffer = (float*)queriesTensor->data();

    std::string errorMsg;
    cpp_knn_batch_distance_pick_omp(ptsBuffer, batch_size, npts, dim, queriesBuffer, nqueries, K, (long long*)indiciesBuffer, errorMsg);
    
    OP_REQUIRES(context, errorMsg.empty(), Status(error::INTERNAL, errorMsg));
  }
};

REGISTER_KERNEL_BUILDER(Name("KnnKdtreeSampler").Device(DEVICE_CPU), KnnKdtreeSamplerOp);
