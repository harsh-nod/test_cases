func.func @matmul(%lhs: tensor<128x1024xf16>, %rhs: tensor<1280x1024xf16>) -> tensor<128x1280xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x1280xf32>
  %inital_result = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x1280xf32>) -> tensor<128x1280xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<128x1024xf16>, tensor<1280x1024xf16>)
             outs(%inital_result: tensor<128x1280xf32>) -> tensor<128x1280xf32>
  return %result : tensor<128x1280xf32>
}
