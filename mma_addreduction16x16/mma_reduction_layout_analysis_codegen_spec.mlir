// RUN: iree-opt %s

#layout = #iree_gpu.mfma_layout<F16_16x16x16_F32>

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__mma_main(
      %variant_op: !transform.any_op {transform.consumed}) {
    // Step 1. Find the fill and matmul ops
    // ===========================================================================
    %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match ops{["linalg.generic"]}
                attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]}
                in %variant_op : (!transform.any_op) -> !transform.any_op
    %reduce = transform.structured.match ops{["linalg.generic"]}
                attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]}
                in %variant_op : (!transform.any_op) -> !transform.any_op
    %broadcast = transform.structured.match ops{["linalg.generic"]}
                attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]}
                in %variant_op : (!transform.any_op) -> !transform.any_op

    // Step 2. Tile the matmul and fuse the fill
    // ===========================================================================
    %grid_reduction, %forall_grid =
    transform.structured.tile_using_forall %broadcast tile_sizes [16] ( mapping = [#gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()
    transform.structured.fuse_into_containing_op %reduce into %forall_grid : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %matmul into %forall_grid : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %fill into %forall_grid : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // // Promote operands in order to test loading from shared memory.
    // %matmul_2 = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    // %promoted_matmul, %alloc_0, %alloc_1 =
    //   transform.iree.promote_operands %matmul_2 [0, 1]
    //     : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)


    // Step 3. Vectorize
    // ===========================================================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op

    // Step 4. Bufferize
    // ===========================================================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.linalg.erase_unnecessary_inputs
    } : !transform.any_op
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_2 {
      transform.apply_patterns.iree.fold_arith_ext_into_contraction
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7
        workgroup_dims = [64, 1, 1] : (!transform.any_op) -> ()

    %contract = transform.structured.match ops{["vector.contract"]} in %variant_op_3 :  (!transform.any_op) -> !transform.any_op

    // Step 7. Do layout analysis and lower to mma
    %layout16x16x16 = transform.param.constant #layout -> !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract, %layout16x16x16 : !transform.any_op, !transform.any_param

    %distribute_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op

    transform.print %distribute_func : !transform.any_op

    transform.iree.amdgpu_distribute_vectors %distribute_func : !transform.any_op

    %distributed_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %distributed_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %distributed_func : !transform.any_op

    transform.yield
  }

  transform.named_sequence @custom_mma(%mma: !transform.any_op {transform.readonly}) {
    %variant_op = transform.get_parent_op %mma {op_name = "hal.executable.variant"} : (!transform.any_op) -> !transform.any_op
    %exports = transform.structured.match ops{["hal.executable.export"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %attn = transform.param.constant #iree_codegen.translation_info<TransformDialectCodegen
                                                                               codegen_spec = @__mma_main> -> !transform.any_param
    transform.annotate %exports "translation_info" = %attn : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_mma(%mma: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %mma ["linalg.matmul_transpose_b"] : !transform.any_op
    transform.yield %mma : !transform.any_op
  }

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        @match_mma -> @custom_mma
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} // module
