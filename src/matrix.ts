import { NDArray, Scalar } from "./ndarray.ts";
import { Vector, VectorLike } from "./vector.ts";

type MatrixLike = Vector[] | Matrix | number[][];

export class Matrix extends NDArray<Matrix, Vector> {
  constructor(rows: number, cols: number) {
    super(rows);
    for (let i = 0; i < rows; i++) {
      this[i] = new Vector(cols);
    }
  }

  public get shape(): number[] {
    return [this.rows, this.cols];
  }

  public get rows(): number {
    return this.length;
  }

  public get cols(): number {
    return this[0].length;
  }

  public add(arg: Matrix | Scalar): this {
    if (typeof arg === "number") {
      for (let i = 0; i < this.length; i++) this[i].add(arg);
    } else {
      for (let i = 0; i < this.length; i++) this[i].add(arg[i]);
    }

    return this;
  }

  public sub(arg: Matrix | Scalar): this {
    if (typeof arg === "number") {
      for (let i = 0; i < this.length; i++) this[i].sub(arg);
    } else {
      for (let i = 0; i < this.length; i++) this[i].sub(arg[i]);
    }

    return this;
  }

  public mul(arg: Matrix | Scalar): this {
    if (typeof arg === "number") {
      for (let i = 0; i < this.length; i++) this[i].mul(arg);
    } else {
      for (let i = 0; i < this.length; i++) this[i].mul(arg[i]);
    }

    return this;
  }

  public div(arg: Matrix | Scalar): this {
    if (typeof arg === "number") {
      for (let i = 0; i < this.length; i++) this[i].div(arg);
    } else {
      for (let i = 0; i < this.length; i++) this[i].div(arg[i]);
    }

    return this;
  }

  public rand(min = -0.1, max = 0.1): this {
    for (let i = 0; i < this.length; i++) this[i].rand(min, max);
    return this;
  }

  public copy(): Matrix {
    return Matrix.from(this);
  }

  public dot(arg: Matrix): Matrix {
    return Matrix.dot(this, arg);
  }

  public convolve(kernel: Matrix): Matrix {
    return Matrix.convolve(this, kernel);
  }

  ///// ///// ///// ///// /////

  public static from(raw: MatrixLike): Matrix {
    const rows = raw.length;
    const cols = raw[0].length;
    const out = new Matrix(rows, cols);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        out[i][j] = raw[i][j];
      }
    }

    return out;
  }

  public static createRow(raw: VectorLike): Matrix {
    const out = new Matrix(1, raw.length);
    for (let i = 0; i < raw.length; i++) {
      out[0][i] = raw[i];
    }
    return out;
  }

  public static createColumn(raw: VectorLike): Matrix {
    const out = new Matrix(raw.length, 1);
    for (let i = 0; i < raw.length; i++) {
      out[i][0] = raw[i];
    }
    return out;
  }

  public static dot(a: MatrixLike, b: MatrixLike): Matrix {
    if (a[0].length !== b.length) {
      throw new Error("Matrix dimensions do not match");
    }

    const out = new Matrix(a.length, b[0].length);

    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < b[0].length; j++) {
        out[i][j] = 0;
        for (let k = 0; k < a[0].length; k++) {
          out[i][j] += (a[i][k] * b[k][j]) || 0;
        }
      }
    }

    return out;
  }

  public static convolve(m: MatrixLike, kernel: MatrixLike): Matrix {
    const padh = Math.floor(kernel.length / 2);
    const padw = Math.floor(kernel[0].length / 2);
    const mrows = m.length;
    const mcols = m[0].length;
    const krows = kernel.length;
    const kcols = kernel[0].length;

    const out = new Matrix(m.length, m[0].length);

    for (let i = 0; i < mrows; i++) {
      for (let j = 0; j < mcols; j++) {
        for (let k = 0; k < krows; k++) {
          for (let l = 0; l < kcols; l++) {
            const x = i + k - padh;
            const y = j + l - padw;
            if (x >= 0 && x < mrows && y >= 0 && y < mcols) {
              out[i][j] += (m[x][y] * kernel[k][l]) || 0;
            }
          }
        }
      }
    }

    return out;
  }
}
