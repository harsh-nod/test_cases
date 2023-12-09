#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @matmul_reduction(%lhs : tensor<16x16xf16>, %rhs : tensor<16x16xf16>) -> tensor<16x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant -1.0e+04 : f32
  %acc = tensor.empty() : tensor<16xf32>
  %init = linalg.fill ins(%c1 : f32) outs(%acc : tensor<16xf32>) -> tensor<16xf32>
  %0 = tensor.empty() : tensor<16x16xf32>
  %1 = linalg.fill ins(%c0 : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %2 = linalg.matmul_transpose_b ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>)
      outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
        ins(%2 : tensor<16x16xf32>) outs(%init : tensor<16xf32>) {
        ^bb0(%in: f32, %out: f32):
          %20 = arith.maximumf %in, %out : f32
          linalg.yield %20 : f32
        } -> tensor<16xf32>
  %8 = linalg.generic {indexing_maps = [#map1, #map], iterator_types=["parallel", "parallel"]}
        ins(%6 : tensor<16xf32>) outs(%0 : tensor<16x16xf32>) {
        ^bb0(%in: f32,  %out: f32):
          linalg.yield %in : f32
        } -> tensor<16x16xf32>
  return %8 : tensor<16x16xf32>
}
