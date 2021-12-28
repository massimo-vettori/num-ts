export type Scalar = number;

export abstract class NDArray<
  ContainerType extends NDArray<ContainerType, SubType>,
  SubType,
> extends Array<SubType> {
  constructor(size: number) {
    super(size);
  }

  public abstract get shape(): number[];

  public abstract add(other: ContainerType | Scalar): this;
  public abstract sub(other: ContainerType | Scalar): this;
  public abstract mul(other: ContainerType | Scalar): this;
  public abstract div(other: ContainerType | Scalar): this;
  public abstract rand(min: Scalar, max: Scalar): this;
  public abstract copy(): ContainerType;

  public dot?(arg: ContainerType | SubType): ContainerType | SubType;
  public convolve?(kernel: ContainerType): ContainerType;
}
