Instructions

Compile:
```
iree-build/tools/iree-compile mma_reduction_layout_analysis.mlir --iree-hal-target-backends=rocm --iree-rocm-target-chip=gfx90a --iree-rocm-link-bc=true --iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode --iree-flow-dispatch-use-transform-dialect=mma_reduction_layout_analysis_dispatch_spec.mlir --iree-codegen-transform-dialect-library=mma_reduction_layout_analysis_codegen_spec.mlir -o mm.vmfb
```

Generate reference outputs (needs numpy):
```
python generate_matrices.py
```

Run:
```
iree-build/tools/iree-run-module --module=mm.vmfb --function=matmul_reduction --device=rocm --input=@matrix_a.npy --input=@matrix_b.npy --output=@output.npy
```

Evaluate error:
```
python evaluate.py
```

Max error should be:
```
Max error =  0.0018558502
```
