#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("KnnKdtree")
    .Input("nsample: int32")
    .Input("xyz: float32")
    .Input("new_xyz: float32")
    .Output("indices: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto batch_size = c->Dim(c->input(1), 0);
      auto queriesShape = c->Dim(c->input(2), 1);
      auto K = c->UnknownDim();
      c->set_output(0, c->MakeShape({ batch_size, queriesShape, K }));
      return Status::OK();
    });

REGISTER_OP("KnnKdtreeSampler")
    .Input("nsample: int32")
    .Input("xyz: float32")
    .Input("npts: int32")
    .Output("indices: int64")
    .Output("queries: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto batch_size = c->Dim(c->input(1), 0);
      auto dim = c->Dim(c->input(1), 2);
      c->set_output(0, c->MakeShape({ batch_size, c->UnknownDim(), c->UnknownDim() }));
      c->set_output(1, c->MakeShape({ batch_size, c->UnknownDim(), dim }));
      return Status::OK();
    });
