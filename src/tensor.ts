import { NDArray, Scalar } from "./ndarray.ts";
import { Matrix } from "./matrix.ts";
import { Vector } from "./vector.ts";

export type TensorLike = Tensor | number[][][] | Matrix[];

export class Tensor extends NDArray<Tensor, Matrix> {

  constructor(rows: number, cols: number, depth: number) {
    super(depth);
    for (let i=0; i<this.length; i++) {
      this[i] = new Matrix(rows, cols);
    }
  }

  public get shape(): number[] {
    return [this.rows, this.cols, this.depth];
  }

  public get rows(): number {
    return this[0].rows;
  }

  public get cols(): number {
    return this[0].cols;
  }

  public get depth(): number {
    return this.length;
  }

  public add(arg: Tensor | Scalar): this {
    if (typeof arg === "number") {
      for (let i = 0; i < this.length; i++) {
        this[i].add(arg);
      }
    } else {
      for (let i = 0; i < this.length; i++) {
        this[i].add(arg[i]);
      }
    }
    return this;
  }

  public sub(arg: Tensor | Scalar): this {
    if (typeof arg === "number") {
      for (let i = 0; i < this.length; i++) {
        this[i].sub(arg);
      }
    } else {
      for (let i = 0; i < this.length; i++) {
        this[i].sub(arg[i]);
      }
    }
    return this;
  }

  public mul(arg: Tensor | Scalar): this {
    if (typeof arg === "number") {
      for (let i = 0; i < this.length; i++) {
        this[i].mul(arg);
      }
    } else {
      for (let i = 0; i < this.length; i++) {
        this[i].mul(arg[i]);
      }
    }
    return this;
  }

  public div(arg: Tensor | Scalar): this {
    if (typeof arg === "number") {
      for (let i = 0; i < this.length; i++) {
        this[i].div(arg);
      }
    } else {
      for (let i = 0; i < this.length; i++) {
        this[i].div(arg[i]);
      }
    }
    return this;
  }

  public rand(min = -0.1, max = 0.1): this {
    for (let i = 0; i < this.length; i++) {
      this[i].rand(min, max);
    }
    return this;
  }

  public copy(): Tensor {
    return Tensor.from(this);
  }

  public convolve(kernel: TensorLike): Tensor {
    return Tensor.convolve(this, kernel);
  }

  ///// ///// ///// ///// /////

  public static from(raw: TensorLike): Tensor {
    const out = new Tensor(raw[0].length, raw[0][0].length, raw.length);

    for (let i = 0; i < raw.length; i++) {
      for (let j = 0; j < raw[i].length; j++) {
        for (let k = 0; k < raw[i][j].length; k++) {
          out[i][j][k] = raw[i][j][k];
        }
      }
    }

    return out;
  }

  public static convolve(m: TensorLike, kernel: TensorLike): Tensor {
    const padh = Math.floor(kernel[0].length / 2);
    const padw = Math.floor(kernel[0][0].length / 2);

    const out = new Tensor(m[0].length, m[0][0].length, m.length);

    for (let z=0; z<out.depth; ++z) {
      for (let y=0; y<out.rows; ++y) {
        for (let x=0; x<out.cols; ++x) {
          
          for (let kz=0; kz<kernel.length; ++kz) {
            for (let ky=0; ky<kernel[0].length; ++ky) {
              for (let kx=0; kx<kernel[0][0].length; ++kx) {
                const mx = x + kx - padw;
                const my = y + ky - padh;

                if (mx >= 0 && mx < m[0].length && my >= 0 && my < m[0][0].length) {
                  out[z][y][x] += m[z][my][mx] * kernel[kz][ky][kx];
                }
              }
            }
          }

        }
      }
    }

    return out;
  }

  public static values(rows: number, cols: number, depth: number, value: Scalar): Tensor {
    const out = new Tensor(rows, cols, depth);

    for (let i = 0; i < out.length; i++) {
      for (let j = 0; j < out[i].length; j++) {
        for (let k = 0; k < out[i][j].length; k++) {
          out[i][j][k] = value;
        }
      }
    }

    return out;
  }

  public static zeros(rows: number, cols: number, depth: number): Tensor {
    return new Tensor(rows, cols, depth);
  }

  public static ones(rows: number, cols: number, depth: number): Tensor {
    return Tensor.values(rows, cols, depth, 1);
  }
}
