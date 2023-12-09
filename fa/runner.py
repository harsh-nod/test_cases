#!/usr/bin/env python3.9
import os
import subprocess
import argparse
import numpy as np

def create_mlir(args):
    ir = ""
    ir += f"func.func @attention(%query: tensor<{args.shape}>, %key: tensor<{args.shape}>, %value: tensor<{args.shape}>) -> tensor<{args.shape}> {{\n"
    ir += f"  %0 = tensor.empty() : tensor<{args.shape}>\n"
    ir += f"  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<{args.shape}>, tensor<{args.shape}>, tensor<{args.shape}>) outs(%0 : tensor<{args.shape}>) -> tensor<{args.shape}>\n"
    ir += f"  return %1 : tensor<{args.shape}>\n"
    ir += "}\n"

    filename = "attention_" + args.shape + ".mlir"
    with open(filename, 'w') as f:
        f.write(ir)
    return filename

def compile(args):
    flags = [
      '/home/harsh/iree-build/tools/iree-compile',\
      '--iree-vm-bytecode-module-output-format=flatbuffer-binary',\
      '--iree-hal-target-backends=rocm',\
      f'--iree-rocm-target-chip=gfx90a',
       '--iree-rocm-link-bc=true',
       '--iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode',
      '--iree-codegen-llvmgpu-enable-transform-dialect-jit=false',\
      '--iree-codegen-use-transform-dialect-strategy=codegen',\
      f'--iree-codegen-transform-dialect-library={args.spec_file}'
    ]
    if args.dump:
        flags += [
            '--mlir-disable-threading',\
            '--mlir-print-ir-after-all',\
            '--iree-hal-dump-executable-binaries-to=/home/harsh/iree/tmp',\
            '--iree-hal-dump-executable-intermediates-to=/home/harsh/iree/tmp'
        ]
    flags += ['-iree-hal-benchmark-dispatch-repeat-count=100']
    flags += [
      f'{args.input_file}',\
      '-o',\
      '/home/harsh/iree/attn.vmfb'
    ]
    cmd = ' '.join(flags)
    print(cmd)
    with open('dump.txt', 'w') as output_file:
        p = subprocess.Popen(flags, stderr=output_file)
        p.wait()

def validate(args):
    flags = [
      '/home/harsh/iree-build/tools/iree-run-module',\
      '--module=/home/harsh/iree/attn.vmfb',\
      '--function=attention',\
      f'--input="@query_{args.shape}.npy"',\
      f'--input="@key_{args.shape}.npy"',\
      f'--input="@value_{args.shape}.npy"',\
      '--device=rocm',\
      '--output=@actual_output.npy'
    ]
    cmd = ' '.join(flags)
    print(cmd)
    os.system(cmd)

    golden = np.load(f'output_{args.shape}.npy')
    computed = np.load('actual_output.npy')
    print(computed[0, 0], computed.shape)
    print(np.max(np.abs(golden - computed)))
    error = np.max(np.abs(golden - computed))
    tol = 1e-1
    if error < tol:
        print(">>>>>>>>>>> Success")
    else:
        print(f">>>>>>>>>>> Failure! Got {error} > {tol}")

def benchmark(args):
    output_file = f'attention_{args.shape}'
    flags = [
      '/home/harsh/iree-build/tools/iree-benchmark-module',\
      '--module=/home/harsh/iree/attn.vmfb',\
      '--function=attention',\
      f'--input="{args.shape}"',\
      f'--input="{args.shape}"',\
      f'--input="{args.shape}"',\
      '--device=rocm',
      '--batch_size=100'
    ]
    cmd = ' '.join(flags)
    print(cmd)
    with open('nsys_dump.txt', 'w') as output_file:
        p = subprocess.Popen(flags, stderr=output_file)
        p.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-compile_only", action=argparse.BooleanOptionalAction)
    parser.add_argument("-benchmark_only", action=argparse.BooleanOptionalAction)
    parser.add_argument("-dump", action=argparse.BooleanOptionalAction)
    parser.add_argument("-shape", type=str, default='1x4x4xf16')
    parser.add_argument("-spec_file", type=str, default='flashy4.mlir')
    args = parser.parse_args()

    do_compile = True
    do_validate = True
    do_benchmark = True
    if args.compile_only:
        do_benchmark = False
    if args.benchmark_only:
        do_compile = False

    if args.dump is None:
        args.dump = False

    args.input_file = create_mlir(args)

    if do_compile:
        compile(args)
    if do_validate: validate(args)
    if do_benchmark: benchmark(args)
