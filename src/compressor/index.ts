// @generated file from wasmbuild -- do not edit
// @ts-nocheck: generated
// deno-lint-ignore-file
// deno-fmt-ignore-file
// @ts-self-types="./compressor.d.ts"

// source-hash: d6bc595daee69ff84e006612e819043d2bda824f
import * as wasm from "./compressor.wasm";
export * from "./compressor.internal.js";
import { __wbg_set_wasm } from "./compressor.internal.js";
__wbg_set_wasm(wasm);
wasm.__wbindgen_start();

export declare class Compressor {
  free(): void;
  [Symbol.dispose](): void;
  constructor();
  push(data: Uint8Array): Uint8Array<ArrayBuffer>;
  flush(): Uint8Array<ArrayBuffer>;
}
