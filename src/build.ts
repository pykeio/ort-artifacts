import { exists } from '@std/fs';
import { join } from '@std/path';
import { TarStream, TarStreamInput } from '@std/tar';

import { cpus, arch as getArch, platform as getPlatform } from 'node:os';

import { Command, EnumType, ValidationError } from '@cliffy/command';
import $ from '@david/dax';

import { Compressor } from './compressor/index.ts';

const arch = getArch() as 'x64' | 'arm64';
const platform = getPlatform() as 'win32' | 'darwin' | 'linux';

const TARGET_ARCHITECTURE_TYPE = new EnumType([ 'x86_64', 'aarch64' ]);

class CompressorStream extends TransformStream<Uint8Array<ArrayBuffer>, Uint8Array<ArrayBuffer>> {
	#compressor = new Compressor();

	constructor() {
		super({
			transform: (chunk, controller) => {
				const res = this.#compressor.push(chunk);
				if (res.byteLength) {
					controller.enqueue(res);
				}
			},
			flush: controller => {
				const res = this.#compressor.flush();
				if (res.byteLength) {
					controller.enqueue(res);
				}
			}
		});
	}
}

const CUDA_ARCHIVES: Record<number, Record<'win32' | 'linux', Record<'cudnn' | 'trt', string>>> = {
	12: {
		linux: {
			cudnn: 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn_jit/linux-x86_64/cudnn_jit-linux-x86_64-9.19.0.56_cuda12-archive.tar.xz',
			trt: 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.15.1/tars/TensorRT-10.15.1.29.Linux.x86_64-gnu.cuda-12.9.tar.gz'
		},
		win32: {
			cudnn: 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.19.0.56_cuda12-archive.zip',
			trt: 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.15.1/zip/TensorRT-10.15.1.29.Windows.amd64.cuda-12.9.zip'
		}
	},
	13: {
		linux: {
			cudnn: 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn_jit/linux-x86_64/cudnn_jit-linux-x86_64-9.19.0.56_cuda13-archive.tar.xz',
			trt: 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.15.1/tars/TensorRT-10.15.1.29.Linux.x86_64-gnu.cuda-13.1.tar.gz'
		},
		win32: {
			cudnn: 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.19.0.56_cuda13-archive.zip',
			trt: 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.15.1/zip/TensorRT-10.15.1.29.Windows.amd64.cuda-13.1.zip'
		}
	}
};
const NVRTX_ARCHIVES: Record<number, Record<'win32' | 'linux', string>> = {
	12: {
		linux: 'https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.3/TensorRT-RTX-1.3.0.35-Linux-x86_64-cuda-12.9-Release-external.tar.gz',
		win32: 'https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.3/TensorRT-RTX-1.3.0.35-win10-amd64-cuda-12.9-Release-external.zip'
	},
	13: {
		linux: 'https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.3/TensorRT-RTX-1.3.0.35-Linux-x86_64-cuda-13.1-Release-external.tar.gz',
		win32: 'https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.3/TensorRT-RTX-1.3.0.35-win10-amd64-cuda-13.1-Release-external.zip'
	}
};

async function *makeTarInput(folder: string): AsyncGenerator<TarStreamInput> {
	for await (const entry of Deno.readDir(folder)) {
		if (!entry.isFile) {
			continue;
		}

		const path = join(folder, entry.name);
		const { size } = await Deno.stat(path);
		yield {
			type: 'file',
			path: entry.name,
			size,
			readable: (await Deno.open(path, { read: true })).readable
		};
	}
}

await new Command()
	.name('ort-artifact')
	.version('0.1.0')
	.type('target-arch', TARGET_ARCHITECTURE_TYPE)
	.option('-v, --upstream-version <version:string>', 'Exact version of upstream package', { required: true })
	.option('-t, --training', 'Enable Training API')
	.option('-s, --static', 'Build static library')
	.option('--iphoneos', 'Target iOS / iPadOS')
	.option('--iphonesimulator', 'Target iOS / iPadOS simulator')
	.option('--android', 'Target Android')
	.option('--cuda <version:integer>', 'Enable CUDA EP', {
		value(value: number) {
			if (value !== 12 && value !== 13) {
				throw new ValidationError('--cuda must be either 12 or 13');
			}
			return value;
		}
	})
	.option('--trt', 'Enable TensorRT EP', { depends: [ 'cuda' ] })
	.option('--nvrtx <cuda_version:integer>', 'Enable NV TensorRT RTX EP', {
		value(value: number) {
			if (value !== 12 && value !== 13) {
				throw new ValidationError('--nvrtx must be either 12 or 13');
			}
			return value;
		}
	})
	.option('--directml', 'Enable DirectML EP')
	.option('--coreml', 'Enable CoreML EP')
	.option('--dnnl', 'Enable DNNL EP')
	.option('--xnnpack', 'Enable XNNPACK EP')
	.option('--webgpu', 'Enable WebGPU EP')
	.option('--openvino', 'Enable OpenVINO EP')
	.option('--nnapi', 'Enable NNAPI EP')
	.option('-N, --ninja', 'build with ninja')
	.option('--vs2026', 'Use Visual Studio 2026 generator')
	.option('-A, --arch <arch:target-arch>', 'Configure target architecture for cross-compile', { default: 'x86_64' })
	.action(async (options, ..._) => {
		const root = Deno.cwd();

		const onnxruntimeRoot = join(root, 'onnxruntime');
		const isExists = await exists(onnxruntimeRoot)
		let isBranchCorrect = false;
		if (isExists) {
			$.cd(onnxruntimeRoot);
			const currentBranch = (await $`git branch --show-current`.stdout("piped")).stdout.trim()
			isBranchCorrect = currentBranch === `rel-${options.upstreamVersion}`;
			$.cd(root);
			
			if (!isBranchCorrect) {
				console.log(`Removing onnxruntime directory because branch is incorrect: ${onnxruntimeRoot}, current branch: ${currentBranch}, expected branch: rel-${options.upstreamVersion}`);
				await Deno.remove(onnxruntimeRoot, { recursive: true });
			}
		}
		if (!isExists || !isBranchCorrect) {
			await $`git clone https://github.com/microsoft/onnxruntime --recursive --single-branch --depth 1 --branch rel-${options.upstreamVersion}`;
		}

		$.cd(onnxruntimeRoot);

		await $`git reset --hard HEAD`;
		await $`git clean -fdx`;

		const patchDir = join(root, 'src', 'patches', 'all');
		for await (const patchFile of Deno.readDir(patchDir)) {
			if (!patchFile.isFile) {
				continue;
			}

			await $`git apply ${join(patchDir, patchFile.name)} --ignore-whitespace --recount --verbose`;
			console.log(`applied ${patchFile.name}`);
		}

		const env = { ...Deno.env.toObject() };
		const args = [];
		const compilerFlags = [];
		const cudaFlags: string[] = [];

		const cudaArchives = options.cuda ? CUDA_ARCHIVES[options.cuda][platform as 'win32' | 'linux'] : null;

		if (platform === 'linux' && !options.android) {
			env.CC = 'clang-18';
			env.CXX = 'clang++-18';
			if (options.cuda) {
				cudaFlags.push('-ccbin', 'clang++-18');
			}
		} else if (platform === 'win32') {
			args.push('-G', options.vs2026 ? 'Visual Studio 18 2026' : 'Visual Studio 17 2022');
			if (options.arch === 'x86_64') {
				args.push('-A', 'x64');
			}
		}

		// Build for iOS on macOS.
		if (platform === 'darwin' && (options.iphoneos || options.iphonesimulator)) {
			args.push(`-DCMAKE_OSX_DEPLOYMENT_TARGET=${Deno.env.get("IPHONEOS_DEPLOYMENT_TARGET")}`)
			args.push('-DCMAKE_TOOLCHAIN_FILE=../cmake/onnxruntime_ios.toolchain.cmake');
			if(options.iphoneos) {
				args.push('-DCMAKE_OSX_SYSROOT=iphoneos');
			} else {
				args.push('-DCMAKE_OSX_SYSROOT=iphonesimulator');
			}
		}

		// Build for Android on Linux.
		if (platform === 'linux' && options.android) {
			// ANDROID_NDK_HOME and ANDROID_SDK_ROOT are expected to be set in the environment.
			args.push(`-DANDROID_PLATFORM=android-${Deno.env.get("ANDROID_API")}`);
			args.push('-DANDROID_ABI=arm64-v8a');
			args.push('-DANDROID_USE_LEGACY_TOOLCHAIN_FILE=false');
			args.push(`-DCMAKE_TOOLCHAIN_FILE=${join(Deno.env.get('ANDROID_NDK_HOME')!, 'build', 'cmake', 'android.toolchain.cmake')}`);
		}

		if (options.cuda) {
			args.push('-Donnxruntime_USE_CUDA=ON');
			// https://github.com/microsoft/onnxruntime/pull/20768
			args.push('-Donnxruntime_NVCC_THREADS=1');

			const cudnnOutPath = join(root, 'cudnn');
			let should_skip = await exists(cudnnOutPath);
			if (should_skip) {
				// Check dir whether is empty
				const files = await Array.fromAsync(Deno.readDir(cudnnOutPath));
				if (files.length === 0) {
					await $`rm -rf ${cudnnOutPath}`;
					should_skip = false;
				}
			}

			if (!should_skip) {
				const cudnnArchiveStream = await fetch(cudaArchives!.cudnn).then(c => c.body!);
				await Deno.mkdir(cudnnOutPath);
				await $`tar xvJC ${cudnnOutPath} --strip-components=1 -f -`.stdin(cudnnArchiveStream);
			}
			
			args.push(`-Donnxruntime_CUDNN_HOME=${cudnnOutPath}`);

			if (platform === 'win32') {
				// nvcc < 12.4 throws an error with VS 17.10
				cudaFlags.push('-allow-unsupported-compiler');
			}
		}

		if (options.cuda || options.trt || options.nvrtx) {
			args.push('-Donnxruntime_USE_FPA_INTB_GEMM=OFF');
			args.push('-DCMAKE_CUDA_ARCHITECTURES=75;80;90');
			cudaFlags.push('-compress-mode=size');
		}

		if (options.trt) {
			args.push('-Donnxruntime_USE_TENSORRT=ON');
			args.push('-Donnxruntime_USE_TENSORRT_BUILTIN_PARSER=ON');
		}
		if (options.nvrtx) {
			args.push('-Donnxruntime_USE_NV=ON');
			args.push('-Donnxruntime_USE_TENSORRT_BUILTIN_PARSER=ON');
		}

		if (options.trt) {
			const trtArchiveStream = await fetch(cudaArchives!.trt).then(c => c.body!);
			const trtOutPath = join(root, 'tensorrt');
			await Deno.mkdir(trtOutPath);
			await $`tar xvzC ${trtOutPath} --strip-components=1 -f -`.stdin(trtArchiveStream);
			args.push(`-Donnxruntime_TENSORRT_HOME=${trtOutPath}`);
		}
		if (options.nvrtx) {
			const trtxArchiveStream = await fetch(NVRTX_ARCHIVES[options.nvrtx!][platform as 'linux' | 'win32']).then(c => c.body!);
			const trtxOutPath = join(root, 'nvrtx');
			await Deno.mkdir(trtxOutPath);
			await $`tar xvzC ${trtxOutPath} --strip-components=1 -f -`.stdin(trtxArchiveStream);
			args.push(`-Donnxruntime_TENSORRT_RTX_HOME=${trtxOutPath}`);
		}

		if (platform === 'win32' && options.directml) {
			args.push('-Donnxruntime_USE_DML=ON');
		}
		if (platform === 'darwin' && options.coreml) {
			args.push('-Donnxruntime_USE_COREML=ON');
		}
		if (options.webgpu) {
			args.push('-Donnxruntime_USE_WEBGPU=ON');
			args.push('-Donnxruntime_BUILD_WEBGPU_EP_STATIC_LIB=ON');
			args.push('-Donnxruntime_ENABLE_DELAY_LOADING_WIN_DLLS=OFF');
			args.push('-Donnxruntime_USE_EXTERNAL_DAWN=OFF');
			args.push('-Donnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON');
			args.push('-Donnxruntime_WGSL_TEMPLATE=static');
		}
		if (options.dnnl) {
			args.push('-Donnxruntime_USE_DNNL=ON');
		}
		if (options.xnnpack) {
			args.push('-Donnxruntime_USE_XNNPACK=ON');
		}
		if (options.openvino) {
			args.push('-Donnxruntime_DISABLE_RTTI=OFF');
			args.push('-Donnxruntime_USE_OPENVINO=ON');
			args.push('-Donnxruntime_USE_OPENVINO_CPU=ON');
			args.push('-Donnxruntime_USE_OPENVINO_GPU=ON');
			args.push('-Donnxruntime_USE_OPENVINO_NPU=ON');
			// args.push('-Donnxruntime_USE_OPENVINO_INTERFACE=ON');
		}
		if(options.nnapi) {
			args.push('-Donnxruntime_USE_NNAPI_BUILTIN=ON');
		}

		if (platform === 'darwin') {
			if (options.arch === 'aarch64') {
				args.push('-DCMAKE_OSX_ARCHITECTURES=arm64');
			} else {
				args.push('-DCMAKE_OSX_ARCHITECTURES=x86_64');
			}
		} else {
			if (options.arch === 'aarch64' && arch !== 'arm64') {
				args.push('-Donnxruntime_CROSS_COMPILING=ON');
				switch (platform) {
					case 'win32':
						args.push('-A', 'ARM64');
						compilerFlags.push('-D_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS');
						break;
					case 'linux':
						if (!options.android) {
							args.push(`-DCMAKE_TOOLCHAIN_FILE=${join(root, 'toolchains', 'aarch64-unknown-linux-gnu.cmake')}`);
						}
						break;
				}
			}
		}

		if (options.training) {
			args.push('-Donnxruntime_ENABLE_TRAINING=ON');
			args.push('-Donnxruntime_ENABLE_LAZY_TENSOR=OFF');
		}

		if (options.training) {
			args.push('-Donnxruntime_DISABLE_RTTI=OFF');
		}

		if (platform === 'win32' && !options.static) {
			args.push('-DONNX_USE_MSVC_STATIC_RUNTIME=OFF');
			args.push('-Dprotobuf_MSVC_STATIC_RUNTIME=OFF');
			args.push('-Dgtest_force_shared_crt=ON');
		}

		if (!options.static) {
			args.push('-Donnxruntime_BUILD_SHARED_LIB=ON');
		} else {
			if (platform === 'win32') {
				args.push('-DONNX_USE_MSVC_STATIC_RUNTIME=OFF');
				args.push('-Dprotobuf_MSVC_STATIC_RUNTIME=OFF');
				args.push('-Dgtest_force_shared_crt=ON');
				args.push('-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL');
			}
		}

		// https://github.com/microsoft/onnxruntime/pull/21005
		if (platform === 'win32') {
			compilerFlags.push('-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR');
		}

		args.push('-Donnxruntime_BUILD_UNIT_TESTS=OFF');
		args.push(`-Donnxruntime_USE_KLEIDIAI=${options.arch === 'aarch64' ? 'ON' : 'OFF'}`);
		args.push('-Donnxruntime_CLIENT_PACKAGE_BUILD=ON');

		if (options.arch === 'x86_64') {
			switch (platform) {
				case 'linux':
					compilerFlags.push('-march=x86-64-v3');
					break;
				case 'win32':
					compilerFlags.push('/arch:AVX2');
					break;
			}
		}

		if (compilerFlags.length > 0) {
			const allFlags = compilerFlags.join(' ');
			args.push(`-DCMAKE_C_FLAGS=${allFlags}`);
			args.push(`-DCMAKE_CXX_FLAGS=${allFlags}`);
		}

		if (options.ninja && !(platform === 'win32' && options.arch === 'aarch64')) {
			args.push('-G', 'Ninja');
		}

		if (cudaFlags.length) {
			args.push(`-DCMAKE_CUDA_FLAGS_INIT=${cudaFlags.join(' ')}`);
		}

		const sourceDir = options.static ? join(root, 'src', 'static-build') : 'cmake';
		const artifactOutDir = join(root, 'artifact', 'onnxruntime');

		await $`cmake -S ${sourceDir} -B build -D CMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES=Release -DCMAKE_INSTALL_PREFIX=${artifactOutDir} -DONNXRUNTIME_SOURCE_DIR=${onnxruntimeRoot} --compile-no-warning-as-error ${args}`
			.env(env);
		await $`cmake --build build --config Release --parallel ${cpus().length}`;
		await $`cmake --install build`;

		const artifactOut = await Deno.open(join(root, 'artifact.tar.lzma2'), { create: true, write: true });
		await ReadableStream.from(makeTarInput(join(artifactOutDir, 'lib')))
			.pipeThrough(new TarStream())
			.pipeThrough(new CompressorStream())
			.pipeTo(artifactOut.writable);
	})
	.parse(Deno.args);
