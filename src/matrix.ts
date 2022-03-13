import { MatrixLike, Scalar } from "./types.ts";
import { NDArray } from "./ndarray.ts";
import { Vector } from "./vector.ts";

export class Matrix extends Array<Vector> implements NDArray {
  get rows(): number {
    return this.length;
  }

  get cols(): number {
    return this[0].length;
  }

  get shape(): number[] {
    return [this.rows, this.cols];
  }

  constructor(rows: number, cols: number) {
    super(rows);
    for (let i = 0; i < rows; i++) this[i] = new Vector(cols);
  }

  add(other: MatrixLike | Scalar): this {
    if (other instanceof Array) {
      for (let i = 0; i < this.rows; i++) this[i].add(other[i]);
    } else {
      for (let i = 0; i < this.rows; i++) this[i].add(other);
    }

    return this;
  }

  sub(other: MatrixLike | Scalar): this {
    if (other instanceof Array) {
      for (let i = 0; i < this.rows; i++) this[i].sub(other[i]);
    } else {
      for (let i = 0; i < this.rows; i++) this[i].sub(other);
    }

    return this;
  }

  mul(other: MatrixLike | Scalar): this {
    if (other instanceof Array) {
      for (let i = 0; i < this.rows; i++) this[i].mul(other[i]);
    } else {
      for (let i = 0; i < this.rows; i++) this[i].mul(other);
    }

    return this;
  }

  div(other: MatrixLike | Scalar): this {
    if (other instanceof Array) {
      for (let i = 0; i < this.rows; i++) this[i].div(other[i]);
    } else {
      for (let i = 0; i < this.rows; i++) this[i].div(other);
    }

    return this;
  }

  clone(): Matrix {
    const out = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) out[i] = this[i].clone();
    return out;
  }

  copy(other: MatrixLike): this {
    for (let i = 0; i < this.rows; i++) this[i].copy(other[i]);
    return this;
  }

  rand(min = -0.1, max = 0.1): this {
    for (let i = 0; i < this.rows; i++) this[i].rand(min, max);
    return this;
  }

  all(value: Scalar): this {
    for (let i = 0; i < this.rows; i++) this[i].all(value);
    return this;
  }

  row(index: number): Vector {
    return this[index].clone();
  }

  col(index: number): Vector {
    const out = new Vector(this.rows);
    for (let i = 0; i < this.rows; i++) out[i] = this[i][index] || 0;
    return out;
  }

  dot(other: MatrixLike): Matrix {
    return Matrix.dot(this, other);
  }

  convolve(kernel: MatrixLike): Matrix {
    const out = new Matrix(this.rows, this.cols);

    const kr = kernel.length;
    const kc = kernel[0].length;

    const ph = Math.floor(kr / 2);
    const pw = Math.floor(kc / 2);

    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        for (let r = 0; r < kr; r++) {
          for (let c = 0; c < kc; c++) {
            out[i][j] += (this[i + r - ph][j + c - pw] * kernel[r][c]) || 0;
          }
        }
      }
    }
    return out;
  }

  transpose(): Matrix {
    const out = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        out[j][i] = this[i][j];
      }
    }
    return out;
  }

  ///////////////////////////////////////////////////////////////////

  static from(raw: MatrixLike): Matrix {
    return new Matrix(raw.length, raw[0].length).copy(raw);
  }

  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  static ones(rows: number, cols: number): Matrix {
    return Matrix.values(rows, cols, 1);
  }

  static values(rows: number, cols: number, value: number): Matrix {
    return new Matrix(rows, cols).all(value);
  }

  static rand(rows: number, cols: number, min = -0.1, max = 0.1): Matrix {
    return new Matrix(rows, cols).rand(min, max);
  }

  static dot(a: MatrixLike, b: MatrixLike): Matrix {
    const out = new Matrix(a.length, b[0].length);

    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < b[0].length; j++) {
        for (let k = 0; k < a[0].length; k++) {
          out[i][j] += a[i][k] * b[k][j];
        }
      }
    }

    return out;
  }
}
