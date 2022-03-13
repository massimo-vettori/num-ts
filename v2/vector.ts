import { Scalar, VectorLike } from "./types.ts";
import { NDArray } from "./ndarray.ts";

export class Vector extends Array<Scalar> implements NDArray {
  constructor(size: number) {
    super(size);
  }

  get shape(): number[] {
    return [this.length];
  }

  add(other: VectorLike | Vector | Scalar): this {
    if (other instanceof Array) {
      for (let i = 0; i < this.length; i++) this[i] += other[i] || 0;
    } else {
      for (let i = 0; i < this.length; i++) this[i] += other || 0;
    }

    return this;
  }

  sub(other: VectorLike | Vector | Scalar): this {
    if (other instanceof Array) {
      for (let i = 0; i < this.length; i++) this[i] -= other[i] || 0;
    } else {
      for (let i = 0; i < this.length; i++) this[i] -= other || 0;
    }

    return this;
  }

  mul(other: VectorLike | Vector | Scalar): this {
    if (other instanceof Array) {
      for (let i = 0; i < this.length; i++) this[i] *= other[i] || 0;
    } else {
      for (let i = 0; i < this.length; i++) this[i] *= other || 0;
    }

    return this;
  }

  div(other: VectorLike | Vector | Scalar): this {
    if (other instanceof Array) {
      for (let i = 0; i < this.length; i++) {
        this[i] /= other[i] || Number.EPSILON;
      }
    } else {
      for (let i = 0; i < this.length; i++) this[i] /= other || Number.EPSILON;
    }

    return this;
  }

  clone(): Vector {
    const out = new Vector(this.length);
    for (let i = 0; i < this.length; i++) out[i] = this[i];
    return out;
  }

  rand(min = -0.1, max = 0.1): this {
    for (let i = 0; i < this.length; i++) {
      this[i] = Math.random() * (max - min) + min;
    }
    return this;
  }

  all(value: Scalar): this {
    for (let i = 0; i < this.length; i++) this[i] = value;
    return this;
  }

  dot(other: VectorLike | Vector): Scalar {
    let out = 0;
    for (let i = 0; i < this.length; i++) {
      out += this[i] * (other[i] || 0);
    }
    return out;
  }

  ////////////////////////////////////////////////////////////////////////////////

  static zeros(size: number) {
    return new Vector(size);
  }

  static ones(size: number) {
    return Vector.values(size, 1);
  }

  static values(size: number, value: Scalar) {
    return new Vector(size).all(value);
  }

  static rand(size: number, min = -0.1, max = 0.1) {
    return new Vector(size).rand(min, max);
  }
}
