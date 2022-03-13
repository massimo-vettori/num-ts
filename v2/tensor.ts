import { Scalar, TensorLike } from "./types.ts";
import { NDArray } from "./ndarray.ts";
import { Matrix } from "./matrix.ts";

export class Tensor extends Array<Matrix> implements NDArray {
  get depth() {
    return this.length;
  }
  get rows() {
    return this[0].rows;
  }
  get cols() {
    return this[0].cols;
  }

  get shape() {
    return [this.rows, this.cols, this.depth];
  }

  constructor(rows: number, cols: number, depth: number) {
    super(depth);
    for (let i = 0; i < depth; ++i) this[i] = new Matrix(rows, cols);
  }

  add(other: TensorLike | Tensor | Scalar): this {
    if (other instanceof Array) {
      for (let i = 0; i < this.depth; ++i) this[i].add(other[i]);
    } else {
      for (let i = 0; i < this.depth; ++i) this[i].add(other);
    }

    return this;
  }

  sub(other: TensorLike | Tensor | Scalar): this {
    if (other instanceof Array) {
      for (let i = 0; i < this.depth; ++i) this[i].sub(other[i]);
    } else {
      for (let i = 0; i < this.depth; ++i) this[i].sub(other);
    }

    return this;
  }

  mul(other: TensorLike | Tensor | Scalar): this {
    if (other instanceof Array) {
      for (let i = 0; i < this.depth; ++i) this[i].mul(other[i]);
    } else {
      for (let i = 0; i < this.depth; ++i) this[i].mul(other);
    }

    return this;
  }

  div(other: TensorLike | Tensor | Scalar): this {
    if (other instanceof Array) {
      for (let i = 0; i < this.depth; ++i) this[i].div(other[i]);
    } else {
      for (let i = 0; i < this.depth; ++i) this[i].div(other);
    }

    return this;
  }

  clone(): Tensor {
    const out = new Tensor(this.rows, this.cols, this.depth);
    for (let i = 0; i < this.depth; ++i) out[i] = this[i].clone();
    return out;
  }

  copy(other: Tensor | TensorLike): this {
    for (let i = 0; i < this.depth; ++i) this[i].copy(other[i]);
    return this;
  }

  all(value: Scalar): this {
    for (let i = 0; i < this.depth; ++i) this[i].all(value);
    return this;
  }

  rand(min = -0.1, max = 0.1): this {
    for (let i = 0; i < this.depth; ++i) this[i].rand(min, max);
    return this;
  }

  convolve(kernel: Tensor | TensorLike): Tensor {
    const out = new Tensor(this.rows, this.cols, this.depth);
    for (let i = 0; i < this.depth; ++i) out[i] = this[i].convolve(kernel[i]);
    return out;
  }

  ////////////////////////////////////////////////////////////////////////////////

  static zeros(rows: number, cols: number, depth: number): Tensor {
    return new Tensor(rows, cols, depth);
  }

  static ones(rows: number, cols: number, depth: number): Tensor {
    return Tensor.values(rows, cols, depth, 1);
  }

  static values(
    rows: number,
    cols: number,
    depth: number,
    value: Scalar,
  ): Tensor {
    return new Tensor(rows, cols, depth).all(value);
  }

  static rand(
    rows: number,
    cols: number,
    depth: number,
    min = -0.1,
    max = 0.1,
  ): Tensor {
    return new Tensor(rows, cols, depth).rand(min, max);
  }

  static from(raw: TensorLike | Tensor): Tensor {
    return new Tensor(raw[0].length, raw[0][0].length, raw.length).copy(raw);
  }
}
