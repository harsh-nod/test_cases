func.func @matmul(%lhs : tensor<64x32xf16>, %rhs : tensor<32x32xf16>) -> tensor<64x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<64x32xf32>
  %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<64x32xf32>) -> tensor<64x32xf32>
  %2 = linalg.matmul_transpose_b ins(%lhs, %rhs : tensor<64x32xf16>, tensor<32x32xf16>)
      outs(%1 : tensor<64x32xf32>) -> tensor<64x32xf32>
  return %2 : tensor<64x32xf32>
}
