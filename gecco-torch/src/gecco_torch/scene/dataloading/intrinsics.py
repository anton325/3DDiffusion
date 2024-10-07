from dataclasses import dataclass


@dataclass
class Intrinsics:
    height: int
    width: int
    focal: float
    near: int
    far: int