<div align=center>
<img src="https://ort.pyke.io/assets/trend-banner.png" width="350px">
</div>
<div align=center>
<a href="https://github.com/pykeio/ort-artifacts/actions/workflows/build.yml"><img alt="Build All Targets" src="https://img.shields.io/github/actions/workflow/status/pykeio/ort-artifacts/build.yml?style=for-the-badge&label=build"></a> <img alt="ONNX Runtime" src="https://img.shields.io/badge/onnxruntime-v1.30.0-blue?style=for-the-badge&logo=cplusplus"> <img alt="Builds" src="https://img.shields.io/badge/builds-17-informational?style=for-the-badge"> <a href="https://github.com/pykeio/ort-artifacts/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green?style=for-the-badge"></a>
</div>
<hr /><br />

`ort-artifacts` builds the prebuilt [ONNX Runtime](https://onnxruntime.ai/) binaries that [`ort`](https://github.com/pykeio/ort) downloads through its `download-binaries` feature.

Each build clones an ONNX Runtime release, applies the patches in [`src/patches`](src/patches/all), and links everything into one static library per target, so that `ort` users get a working ONNX Runtime with hardware acceleration without compiling it themselves.

## 📦 Supported builds
The [**Build All Targets**](.github/workflows/build.yml) workflow produces **17 builds** across **9 targets**, covering Linux, Windows, macOS, iOS and Android.

### Desktop (CPU)
| Target | Execution providers | Runner |
|:--|:--|:--|
| `x86_64-unknown-linux-gnu` | CPU | `ubuntu-24.04` |
| `aarch64-unknown-linux-gnu` | CPU | `ubuntu-24.04` (cross-compiled) |
| `x86_64-pc-windows-msvc` | DirectML | `windows-2022` |
| `aarch64-pc-windows-msvc` | DirectML | `windows-2022` (cross-compiled) |
| `aarch64-apple-darwin` | CoreML | `macos-15` |

### NVIDIA (CUDA 13.2)
| Target | Execution providers | Runner |
|:--|:--|:--|
| `x86_64-unknown-linux-gnu` | CUDA, TensorRT, TensorRT RTX | `ubuntu-24.04` |
| `aarch64-unknown-linux-gnu` | CUDA, TensorRT, TensorRT RTX | `ubuntu-24.04-arm` |
| `x86_64-pc-windows-msvc` | CUDA, TensorRT, TensorRT RTX, DirectML | `windows-2022` |
| `x86_64-unknown-linux-gnu` | TensorRT RTX | `ubuntu-24.04` |
| `x86_64-pc-windows-msvc` | TensorRT RTX, DirectML | `windows-2022` |

CUDA kernels are compiled for these GPU architectures:

| Platform | Architectures |
|:--|:--|
| x86_64 | `sm_75` Turing, `sm_80` Ampere, `sm_90` Hopper, `sm_120` Blackwell |
| aarch64 | `sm_87` Jetson Orin, `sm_90` Grace Hopper, `sm_100` Grace Blackwell, `sm_110` Jetson Thor, `sm_121` DGX Spark |

### WebGPU
| Target | Execution providers | Runner |
|:--|:--|:--|
| `x86_64-unknown-linux-gnu` | WebGPU | `ubuntu-24.04` |
| `x86_64-pc-windows-msvc` | WebGPU | `windows-2025` |
| `aarch64-apple-darwin` | CoreML, WebGPU | `macos-15` |

### Mobile
| Target | Execution providers | Runner |
|:--|:--|:--|
| `aarch64-apple-ios` | CoreML | `macos-15` |
| `aarch64-apple-ios-sim` | CoreML | `macos-15` |
| `aarch64-linux-android` | NNAPI | `ubuntu-24.04` |
| `x86_64-linux-android` | NNAPI | `ubuntu-24.04` |

Every build is uploaded as `<target>+<feature-set>.tar.lzma2` (for example `x86_64-unknown-linux-gnu+cuda13,tensorrt,nvrtx.tar.lzma2`, or just `<target>.tar.lzma2` for plain CPU builds) and gets a [build provenance attestation](https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations).

## ⚙️ Running the builds
Both workflows are started by hand from the **Actions** tab.

- **Build All Targets** takes an ONNX Runtime version (like `1.30.0`) and builds the whole matrix above.
- **Build Runner** builds a single target, which is handy for testing a change. It takes the version, the Rust target triple, the builder arguments, the feature set name, and the runner to use, plus a CUDA version for NVIDIA builds.

With the GitHub CLI:
```shell
gh workflow run build.yml -f onnxruntime-version=1.30.0

gh workflow run build-runner.yml \
	-f onnxruntime-version=1.30.0 \
	-f target=x86_64-unknown-linux-gnu \
	-f args="--webgpu -N" \
	-f feature-set=webgpu \
	-f runs-on=ubuntu-24.04
```

## 🛠️ Building locally
The builder is a [Deno](https://deno.com/) script. You'll need Deno 2, Git, CMake, Ninja, and a C++ toolchain (Clang 21 on Linux, Visual Studio on Windows, Xcode on macOS).

```shell
deno run -A src/build.ts -v 1.30.0 -s -N
```

This clones the `rel-1.30.0` branch of ONNX Runtime into `onnxruntime/`, applies the patches, builds it, and writes the result to `artifact.tar.lzma2`.

| Option | Description |
|:--|:--|
| `-v, --upstream-version <version>` | ONNX Runtime version to build (required) |
| `-s, --static` | Build a single static library (used for all published builds) |
| `-N, --ninja` | Build with Ninja |
| `-A, --arch <arch>` | Target architecture (`x86_64` or `aarch64`), for cross-compiling |
| `-t, --training` | Enable the training API |
| `--cuda 13` | Enable the CUDA EP |
| `--trt` | Enable the TensorRT EP (needs `--cuda`) |
| `--nvrtx` | Enable the NVIDIA TensorRT RTX EP |
| `--directml` | Enable the DirectML EP |
| `--coreml` | Enable the CoreML EP |
| `--webgpu` | Enable the WebGPU EP |
| `--nnapi` | Enable the NNAPI EP |
| `--xnnpack` | Enable the XNNPACK EP |
| `--openvino` | Enable the OpenVINO EP |
| `--dnnl` | Enable the oneDNN EP |
| `--iphoneos`, `--iphonesimulator` | Build for iOS or the iOS simulator |
| `--android` | Build for Android (needs `ANDROID_NDK_HOME`) |
| `--vs2026` | Use the Visual Studio 2026 generator |
| `--debug` | Build the Debug configuration instead of Release |

CUDA builds for aarch64 Linux have to run on an arm64 host.

To use your own build with `ort`, unpack it and point `ORT_LIB_PATH` at the folder containing the library. See [Linking](https://ort.pyke.io/setup/linking) in the `ort` guide for details.

## 🩹 Patches
Every patch in [`src/patches/all`](src/patches/all) is applied to every build.

| Patch | What it does |
|:--|:--|
| `0001-no-soname` | Removes the `SONAME` from the shared library |
| `0002-ignore-cpuinfo-arm64-patch` | Only applies ONNX Runtime's cpuinfo patch on ARM64EC, not ARM64 |
| `0003-leak-logger-mutex` | Leaks the default logger mutex so it's still alive during shutdown |
| `0004-faulty-kernel-registry-release` | Keeps the CUDA and TensorRT kernel registries alive at shutdown |
| `0006-clang-nvcc-fixes` | Fixes for compiling with Clang and NVCC together |
| `0007-fuck-copilot-redux` | Fixes compile problems in some CUDA attention kernels |
| `0008-disable-deep-gemm` | Disables the DeepGEMM kernels |
| `0009-silence-nvcc-gsl-warnings` | Silences GSL warnings under NVCC |
| `0010-providers-core-static-library` | Always builds the CPU provider as a static library |
| `0011-nv-tensorrt-rtx-own-library-vars` | Links the TensorRT RTX EP against TensorRT RTX rather than TensorRT |
| `0012-sm121-uses-sm120-kernels` | Builds the `sm_120` grouped GEMM kernels for `sm_121` too |
| `0013-cuda-heavy-kernel-job-pool` | Limits how many memory-heavy CUDA kernels compile at once |

## 📁 Layout
| Path | Contents |
|:--|:--|
| [`src/build.ts`](src/build.ts) | The builder script |
| [`src/static-build`](src/static-build) | CMake project that bundles ONNX Runtime and its dependencies into one static library |
| [`src/patches/all`](src/patches/all) | Patches applied to ONNX Runtime before building |
| [`src/compressor`](src/compressor) | LZMA2 compressor used to pack the artifacts |
| [`toolchains`](toolchains) | CMake toolchain files for cross-compiling |
| [`.github/workflows`](.github/workflows) | The build matrix and the single-target build runner |

## 📄 License
Licensed under the [Apache License 2.0](LICENSE).
