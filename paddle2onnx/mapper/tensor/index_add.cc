// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <limits>
#include "paddle2onnx/mapper/tensor/index_add.h"

namespace paddle2onnx {
REGISTER_MAPPER(index_add, IndexAddMapper)

int32_t IndexAddMapper::GetMinOpset(bool verbose) {
  Logger(verbose, 16) << RequireOpset(16) << std::endl;
  return 16;
}

void IndexAddMapper::Opset16() {
  auto x_info = GetInput("X");
  auto index_info = GetInput("Index");
  auto value_info = GetInput("AddValue");
  auto out_info = GetOutput("Out");

  auto dim1 = index_info[0].shape[0];
  auto dim2 = value_info[0].shape[axis_];

  if(dim1<dim2){
    std::string zeros_node = helper_->Constant({dim2-dim1}, GetOnnxDtype(index_info[0].dtype),static_cast<float>(0));
    helper_->Concat({index_info[0].name, zeros_node}, {index_info[0].name}, 0);
  }

  if(dim2<dim1){
    helper_->Slice({index_info[0].name}, {index_info[0].name},{0},{0},{dim2});
  }

  std::vector<int64_t> perm = Arange(0, x_info[0].Rank());
  perm[0] = axis_;
  perm[axis_] = 0;

  std::string x_node=helper_->Transpose(x_info[0].name, perm);
  std::string value_node=helper_->Transpose(value_info[0].name, perm);

  std::string index_node = helper_->Unsqueeze(index_info[0].name, {1});

  auto shape_node = helper_->MakeNode("Shape", {x_node});

  std::string zeros_like_node = helper_->ConstOfShape(
      shape_node->output(0), GetOnnxDtype(x_info[0].dtype),
      static_cast<float>(0));

  std::string input_ids_node = helper_->AutoCast(
      index_node, index_info[0].dtype, P2ODataType::INT64);

  auto scatter_nd_node = helper_->MakeNode(
      "ScatterND",
      {zeros_like_node, input_ids_node, value_node});
  AddAttribute(scatter_nd_node, "reduction", "add");
  
  auto add_node = helper_->MakeNode("Add", {x_node, scatter_nd_node->output(0)});

  helper_->Transpose(add_node->output(0), out_info[0].name, perm);
}

}  // namespace paddle2onnx
