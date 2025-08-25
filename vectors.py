from math import *
from typing import Union, List, Dict, Tuple

Number = Union[int, float]


class Vector2:
    """A 2D vector class with basic mathematical operations."""

    __slots__ = ["x", "y"]

    def __init__(self, x: Number, y: Number):
        """Initialize a 2D vector.

        Args:
            x: X component
            y: Y component
        """
        self.x, self.y = float(x), float(y)

    def __repr__(self):
        return f"Vector2({self.x}, {self.y})"

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: Number) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: Number) -> "Vector2":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> "Vector2":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vector2":
        return Vector2(-self.x, -self.y)

    def __eq__(self, other: "Vector2") -> bool:
        epsilon = 1e-10
        return abs(self.x - other.x) < epsilon and abs(self.y - other.y) < epsilon

    def __hash__(self):
        return hash((round(self.x, 10), round(self.y, 10)))

    def dot(self, other: "Vector2") -> float:
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y

    def magnitude(self) -> float:
        """Calculate the magnitude of the vector."""
        return sqrt(self.x * self.x + self.y * self.y)

    def magnitude_squared(self) -> float:
        """Calculate the squared magnitude (faster than magnitude)."""
        return self.x * self.x + self.y * self.y

    def normalize(self) -> "Vector2":
        """Return a normalized version of this vector."""
        mag_sq = self.x * self.x + self.y * self.y
        if mag_sq == 0:
            return Vector2(0, 0)
        inv_mag = 1.0 / sqrt(mag_sq)
        return Vector2(self.x * inv_mag, self.y * inv_mag)

    def distance_to(self, other: "Vector2") -> float:
        """Calculate distance to another vector."""
        dx = self.x - other.x
        dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)

    def angle(self) -> float:
        """Get the angle of this vector in radians."""
        return atan2(self.y, self.x)

    def rotate(self, angle: float) -> "Vector2":
        """Rotate the vector by the given angle in radians."""
        cos_a, sin_a = cos(angle), sin(angle)
        return Vector2(self.x * cos_a - self.y * sin_a, self.x * sin_a + self.y * cos_a)

    def is_zero(self) -> bool:
        """Check if the vector is zero (within epsilon)."""
        epsilon = 1e-10
        return abs(self.x) < epsilon and abs(self.y) < epsilon


class Vector3:
    """A 3D vector class with basic mathematical operations."""

    __slots__ = ["x", "y", "z"]

    def __init__(self, x: Number, y: Number, z: Number):
        """Initialize a 3D vector.

        Args:
            x: X component
            y: Y component
            z: Z component
        """
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: Number) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: Number) -> "Vector3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> "Vector3":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector3":
        return Vector3(-self.x, -self.y, -self.z)

    def __eq__(self, other: "Vector3") -> bool:
        epsilon = 1e-10
        return (
            abs(self.x - other.x) < epsilon
            and abs(self.y - other.y) < epsilon
            and abs(self.z - other.z) < epsilon
        )

    def __hash__(self):
        return hash((round(self.x, 10), round(self.y, 10), round(self.z, 10)))

    def dot(self, other: "Vector3") -> float:
        """Calculate dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        """Calculate cross product with another vector."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def magnitude(self) -> float:
        """Calculate the magnitude of the vector."""
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def magnitude_squared(self) -> float:
        """Calculate the squared magnitude (faster than magnitude)."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalize(self) -> "Vector3":
        """Return a normalized version of this vector."""
        mag_sq = self.x * self.x + self.y * self.y + self.z * self.z
        if mag_sq == 0:
            return Vector3(0, 0, 0)
        inv_mag = 1.0 / sqrt(mag_sq)
        return Vector3(self.x * inv_mag, self.y * inv_mag, self.z * inv_mag)

    def distance_to(self, other: "Vector3") -> float:
        """Calculate distance to another vector."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return sqrt(dx * dx + dy * dy + dz * dz)

    def is_zero(self) -> bool:
        """Check if the vector is zero (within epsilon)."""
        epsilon = 1e-10
        return abs(self.x) < epsilon and abs(self.y) < epsilon and abs(self.z) < epsilon

    def project_onto(self, other: "Vector3") -> "Vector3":
        """Project this vector onto another vector."""
        if other.is_zero():
            return Vector3(0, 0, 0)
        return other * (self.dot(other) / other.magnitude_squared())

    def reflect(self, normal: "Vector3") -> "Vector3":
        """Reflect this vector off a surface with the given normal."""
        return self - 2 * self.project_onto(normal)

    def rotate_point_around_axis(self, anchor, axis, angle):
        """Rotate a point around an arbitrary axis using Rodrigues' rotation formula.

        Args:
            point: The point to rotate
            anchor: The anchor point of the rotation axis
            axis: The axis of rotation (will be normalized)
            angle: Rotation angle in radians

        Returns:
            The rotated point
        """
        p = self - anchor
        k = axis.normalize()

        cos_a = cos(angle)
        sin_a = sin(angle)

        rotated = (
            p * cos_a
            + Vector3(
                k.y * p.z - k.z * p.y, k.z * p.x - k.x * p.z, k.x * p.y - k.y * p.x
            )
            * sin_a
            + k * (k.dot(p)) * (1 - cos_a)
        )

        return rotated + anchor


def zero2() -> Vector2:
    """Create a zero Vector2."""
    return Vector2(0, 0)


def zero3() -> Vector3:
    """Create a zero Vector3."""
    return Vector3(0, 0, 0)


def unit_x3() -> Vector3:
    """Create a unit vector in the X direction."""
    return Vector3(1, 0, 0)


def unit_y3() -> Vector3:
    """Create a unit vector in the Y direction."""
    return Vector3(0, 1, 0)


def unit_z3() -> Vector3:
    """Create a unit vector in the Z direction."""
    return Vector3(0, 0, 1)
