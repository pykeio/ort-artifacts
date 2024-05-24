import { exists } from 'https://deno.land/std@0.224.0/fs/mod.ts';
import { join } from 'https://deno.land/std@0.224.0/path/mod.ts';

import { arch, cpus, platform } from 'node:os'; // is there really no deno-native function for this?

import { Command, EnumType } from '@cliffy/command';
import $ from '@david/dax';

const TARGET_ARCHITECTURE_TYPE = new EnumType([ 'x86_64', 'aarch64' ]);

await new Command()
	.name('ort-artifact')
	.version('0.1.0')
	.type('target-arch', TARGET_ARCHITECTURE_TYPE)
	.option('-v, --upstream-version <version:string>', 'Exact version of upstream package', { required: true })
	.option('-t, --training', 'Enable Training API')
	.option('-s, --static', 'Build static library')
	.option('--cuda', 'Enable CUDA EP')
	.option('--trt', 'Enable TensorRT EP', { depends: [ 'cuda' ] })
	.option('--directml', 'Enable DirectML EP')
	.option('--coreml', 'Enable CoreML EP')
	.option('--xnnpack', 'Enable XNNPACK EP')
	.option('-A, --arch <arch:target-arch>', 'Configure target architecture for cross-compile', { default: 'x86_64' })
	.option('-W, --wasm', 'Compile for WebAssembly (with patches)')
	.action(async (options, ..._) => {
		const root = Deno.cwd();

		const onnxruntimeRoot = join(root, 'onnxruntime');
		if (!await exists(onnxruntimeRoot)) {
			await $`git clone https://github.com/microsoft/onnxruntime --recursive --single-branch --depth 1 --branch v${options.upstreamVersion}`;
		}

		$.cd(onnxruntimeRoot);

		const args = [];
		if (options.cuda) {
			args.push('-Donnxruntime_USE_CUDA=ON');
			// https://github.com/microsoft/onnxruntime/pull/20768
			args.push('-Donnxruntime_NVCC_THREADS=1');
			if (options.trt) {
				args.push('-Donnxruntime_USE_TENSORRT=ON');
				args.push('-Donnxruntime_USE_TENSORRT_BUILTIN_PARSER=ON');
			}

			switch (platform()) {
				case 'linux': {
					const cudnnArchiveStream = await fetch(Deno.env.get('CUDNN_URL')!).then(c => c.body!);
					const cudnnOutPath = join(root, 'cudnn');
					await Deno.mkdir(cudnnOutPath);
					await $`tar xvJC ${cudnnOutPath} --strip-components=1 -f -`.stdin(cudnnArchiveStream);
					args.push(`-Donnxruntime_CUDNN_HOME=${cudnnOutPath}`);
					
					if (options.trt) {
						const trtArchiveStream = await fetch(Deno.env.get('TENSORRT_URL')!).then(c => c.body!);
						const trtOutPath = join(root, 'tensorrt');
						await Deno.mkdir(trtOutPath);
						await $`tar xvzC ${trtOutPath} --strip-components=1 -f -`.stdin(trtArchiveStream);
						args.push(`-Donnxruntime_TENSORRT_HOME=${trtOutPath}`);
					}

					break;
				}
				case 'win32': {
					// windows should ship with bsdtar which supports extracting .zips
					const cudnnArchiveStream = await fetch(Deno.env.get('CUDNN_URL')!).then(c => c.body!);
					const cudnnOutPath = join(root, 'cudnn');
					await Deno.mkdir(cudnnOutPath);
					await $`tar xvC ${cudnnOutPath} --strip-components=1 -f -`.stdin(cudnnArchiveStream);
					args.push(`-Donnxruntime_CUDNN_HOME=${cudnnOutPath}`);
					
					if (options.trt) {
						const trtArchiveStream = await fetch(Deno.env.get('TENSORRT_URL')!).then(c => c.body!);
						const trtOutPath = join(root, 'tensorrt');
						await Deno.mkdir(trtOutPath);
						await $`tar xvC ${trtOutPath} --strip-components=1 -f -`.stdin(trtArchiveStream);
						args.push(`-Donnxruntime_TENSORRT_HOME=${trtOutPath}`);
					}

					break;
				}
			}
		}

		if (platform() === 'win32' && options.directml) {
			args.push('-Donnxruntime_USE_DIRECTML=ON');
		}
		if (platform() === 'darwin' && options.coreml) {
			args.push('-Donnxruntime_USE_COREML=ON');
		}
		if (options.xnnpack) {
			args.push('-Donnxruntime_USE_XNNPACK=ON');
		}

		if (platform() === 'darwin') {
			if (options.arch === 'aarch64') {
				args.push('-DCMAKE_OSX_ARCHITECTURES=arm64');
			} else {
				args.push('-DCMAKE_OSX_ARCHITECTURES=x86_64');
			}
		} else {
			if (options.arch === 'aarch64' && arch() !== 'arm64') {
				args.push('-Donnxruntime_CROSS_COMPILING=ON');
				switch (platform()) {
					case 'win32':
						args.push('-A', 'ARM64');
						args.push('-DCMAKE_CXX_FLAGS=-D_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS');
						break;
					case 'linux':
						args.push(`-DCMAKE_TOOLCHAIN_FILE=${join(root, 'toolchains', 'aarch64-unknown-linux-gnu.cmake')}`);
						break;
				}
			}
		}

		if (options.training) {
			args.push('-Donnxruntime_ENABLE_TRAINING=ON');
			args.push('-Donnxruntime_ENABLE_LAZY_TENSOR=OFF');
			args.push('-Donnxruntime_DISABLE_RTTI=OFF');
		}

		if (platform() === 'win32' && !options.static) {
			args.push('-DONNX_USE_MSVC_STATIC_RUNTIME=OFF');
			args.push('-Dprotobuf_MSVC_STATIC_RUNTIME=OFF');
			args.push('-Dgtest_force_shared_crt=OFF');
		}

		args.push('-Donnxruntime_BUILD_UNIT_TESTS=OFF');

		const sourceDir = options.static ? join(root, 'src', 'static-build') : 'cmake';

		await $`cmake -S ${sourceDir} -B build -D CMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES=Release -DCMAKE_INSTALL_PREFIX=${join(root, 'output')} -DONNXRUNTIME_SOURCE_DIR=${onnxruntimeRoot} --compile-no-warning-as-error ${args}`;
		await $`cmake --build build --config Release --parallel ${cpus().length}`;
		await $`cmake --install build --config Release`;
	})
	.parse(Deno.args);
