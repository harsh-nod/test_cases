#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @matmul(%lhs : tensor<16x16xf16>, %rhs : tensor<16x16xf16>, %bias : tensor<16x16xf32>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %2 = linalg.matmul_transpose_b ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>)
      outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types=["parallel", "parallel"]}
        ins(%2, %bias : tensor<16x16xf32>, tensor<16x16xf32>) outs(%0 : tensor<16x16xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %10 = arith.subf %arg0, %arg1 : f32
            %11 = math.exp %10 : f32
            linalg.yield %11 : f32
       } -> tensor<16x16xf32>
  return %3 : tensor<16x16xf32>
}
