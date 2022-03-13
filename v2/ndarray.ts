import { NDArrayLike, Scalar } from "./types.ts";

export interface NDArray {
  get shape(): number[];

  add(other: NDArrayLike | NDArray | Scalar): this;
  sub(other: NDArrayLike | NDArray | Scalar): this;
  mul(other: NDArrayLike | NDArray | Scalar): this;
  div(other: NDArrayLike | NDArray | Scalar): this;

  dot?(other: NDArrayLike | NDArray): NDArray | Scalar;
  transpose?(): NDArray;
  reshape?(shape: number[]): NDArray;
  convolve?(kernel: NDArrayLike | NDArray): NDArray;

  clone(): NDArray;
  rand(min: Scalar, max: Scalar): this;
  all(value: Scalar): this;
}
