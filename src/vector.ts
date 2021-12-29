import { NDArray, Scalar } from "./ndarray.ts";
import { Matrix } from "./matrix.ts";

export type VectorLike = Scalar[] | Vector;

export class Vector extends NDArray<Vector, Scalar> {
  constructor(size: number) {
    super(size);
    this.fill(0);
  }

  public get shape(): number[] {
    return [this.length];
  }

  public add(arg: Vector | Scalar): this {
    if (typeof arg === "number") {
      for (let i = 0; i < this.length; i++) this[i] += arg || 0;
    } else {
      for (let i = 0; i < this.length; i++) this[i] += arg[i] || 0;
    }
    return this;
  }

  public sub(arg: Vector | Scalar): this {
    if (typeof arg === "number") {
      for (let i = 0; i < this.length; i++) this[i] -= arg || 0;
    } else {
      for (let i = 0; i < this.length; i++) this[i] -= arg[i] || 0;
    }
    return this;
  }

  public mul(arg: Vector | Scalar): this {
    if (typeof arg === "number") {
      for (let i = 0; i < this.length; i++) this[i] *= arg || 0;
    } else {
      for (let i = 0; i < this.length; i++) this[i] *= arg[i] || 0;
    }
    return this;
  }

  public div(arg: Vector | Scalar): this {
    if (typeof arg === "number") {
      for (let i = 0; i < this.length; i++) this[i] /= arg || 0;
    } else {
      for (let i = 0; i < this.length; i++) this[i] /= arg[i] || 0;
    }
    return this;
  }

  public rand(min = -0.1, max = 0.1): this {
    for (let i = 0; i < this.length; i++) {
      this[i] = Math.random() * (max - min) + min;
    }
    return this;
  }

  public copy(): Vector {
    return Vector.from(this);
  }

  public dot(arg: Vector): Scalar {
    let out = 0;
    for (let i = 0; i < this.length; i++) out += this[i] * arg[i];
    return out;
  }

  public toRowMatrix(): Matrix {
    return Matrix.createRow(this);
  }

  public toColumnMatrix(): Matrix {
    return Matrix.createColumn(this);
  }

  ///// ///// ///// ///// /////

  public static from(raw: VectorLike): Vector {
    const out = new Vector(raw.length);
    for (let i = 0; i < raw.length; i++) out[i] = raw[i];
    return out;
  }

  public static zeros(size: number): Vector {
    return new Vector(size);
  }

  public static ones(size: number): Vector {
    return Vector.values(size, 1);
  }

  public static values(size: number, value: Scalar): Vector {
    return new Vector(size).fill(value);
  }

  public static rand(size: number, min = -0.1, max = 0.1): Vector {
    const out = new Vector(size);
    out.rand(min, max);
    return out;
  }
}
