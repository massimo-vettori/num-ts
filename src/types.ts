export type Scalar = number;
export type VectorLike = Scalar[];
export type MatrixLike = VectorLike[];
export type TensorLike = MatrixLike[];
export type NDArrayLike = VectorLike | NDArrayLike[];
