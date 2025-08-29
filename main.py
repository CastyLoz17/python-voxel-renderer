import turtle
from math import *
from typing import Union, List, Dict, Tuple, Optional
import time
import random
from dataclasses import dataclass

import keyboard

try:
    using_mouse = True
    import mouse
except (ImportError, OSError):
    using_mouse = False
    from pynput.mouse import Listener as MouseListener

Number = Union[int, float]


# Vectors
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


# World
@dataclass
class Block:
    """Represents a single block with texture information"""

    block_type: int
    top_texture: int = 0
    side_texture: int = 0
    bottom_texture: int = 0

    def __post_init__(self):
        if self.side_texture == 0:
            self.side_texture = self.top_texture
        if self.bottom_texture == 0:
            self.bottom_texture = self.top_texture


class TextureManager:
    def __init__(self):
        self.textures = {
            0: "#000000",  # Air/Empty (black)
            1: "#654321",  # Dirt (brown)
            2: "#228b22",  # Grass top (green)
            3: "#8b4513",  # Dirt/Grass side (brown)
            4: "#a0a0a0",  # Stone (gray)
            5: "#2c3e50",  # Bedrock (dark blue-gray)
            6: "#8b4513",  # Tree trunk (brown)
            7: "#228b22",  # Leaves (green)
        }

        self.textures = {
            k: tuple(int(v[i : i + 2], 16) for i in range(1, 7, 2))
            for k, v in self.textures.items()
        }

        self.block_names = {
            0: "Air",
            1: "Dirt",
            2: "Grass",
            3: "Stone",
            4: "Bedrock",
            5: "Tree Trunk",
            6: "Leaves",
        }

        self.block_definitions = {
            0: Block(0, 0, 0, 0),  # Air
            1: Block(1, 1, 1, 1),  # Dirt - all dirt texture
            2: Block(2, 2, 3, 1),  # Grass - grass top, dirt sides, dirt bottom
            3: Block(3, 4, 4, 4),  # Stone - all stone texture
            4: Block(4, 5, 5, 5),  # Bedrock - all bedrock texture
            5: Block(5, 6, 6, 6),  # Tree Trunk - all trunk texture
            6: Block(6, 7, 7, 7),  # Leaves - all leaves texture
        }
        self._lighting_cache = {}
        self._precompute_lighting()

    def _precompute_lighting(self):
        max_distance = 30
        min_brightness = 0.2
        face_aos = [1.0, 0.9, 0.8, 0.6]

        for texture_id in self.textures:
            for distance in range(0, int(max_distance) + 1):
                for ao in face_aos:
                    key = (texture_id, distance, ao)
                    base_color = self.textures[texture_id]
                    distance_factor = max(
                        min_brightness, 1.0 - (distance / max_distance)
                    )
                    lighting_factor = distance_factor * ao
                    r, g, b = base_color
                    self._lighting_cache[key] = (
                        int(r * lighting_factor),
                        int(g * lighting_factor),
                        int(b * lighting_factor),
                    )

    def get_color(
        self,
        texture_id: int,
        distance: float = 0,
        ambient_occlusion: float = 1.0,
        max_distance: float = 30,
        min_brightness: float = 0.2,
    ) -> Tuple[int, int, int]:
        cache_key = (texture_id, int(distance), ambient_occlusion)
        if cache_key in self._lighting_cache:
            return self._lighting_cache[cache_key]

        base_color = self.textures.get(texture_id, (255, 255, 255))
        distance_factor = max(min_brightness, 1.0 - (distance / max_distance))
        lighting_factor = distance_factor * ambient_occlusion
        r, g, b = base_color
        return (
            int(r * lighting_factor),
            int(g * lighting_factor),
            int(b * lighting_factor),
        )

    def get_block(self, block_type: int) -> Block:
        return self.block_definitions.get(block_type, self.block_definitions[0])

    def get_block_name(self, block_type: int) -> str:
        return self.block_names.get(block_type, "Unknown")


class NoiseGenerator:
    """Simple noise generator for terrain"""

    def __init__(self, seed: int = None):
        self.seed = seed
        if seed:
            random.seed(seed)
        self.permutation = list(range(256))
        random.shuffle(self.permutation)
        self.permutation += self.permutation

    def noise2d(self, x, y):
        X, Y = int(floor(x)) & 255, int(floor(y)) & 255
        x, y = x - floor(x), y - floor(y)
        u, v = x * x * x * (x * (x * 6 - 15) + 10), y * y * y * (y * (y * 6 - 15) + 10)

        A, B = self.permutation[X] + Y, self.permutation[X + 1] + Y
        AA, AB, BA, BB = (
            self.permutation[A],
            self.permutation[A + 1],
            self.permutation[B],
            self.permutation[B + 1],
        )

        def grad(h, x, y):
            h &= 15
            u = x if h < 8 else y
            v = y if h < 4 else (x if h == 12 or h == 14 else 0)
            return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

        return (1 - v) * (
            (1 - u) * grad(self.permutation[AA], x, y)
            + u * grad(self.permutation[BA], x - 1, y)
        ) + v * (
            (1 - u) * grad(self.permutation[AB], x, y - 1)
            + u * grad(self.permutation[BB], x - 1, y - 1)
        )

    def noise3d(self, x, y, z):
        """3D Perlin noise for better cave generation"""
        X = int(floor(x)) & 255
        Y = int(floor(y)) & 255
        Z = int(floor(z)) & 255

        x -= floor(x)
        y -= floor(y)
        z -= floor(z)

        u = x * x * x * (x * (x * 6 - 15) + 10)
        v = y * y * y * (y * (y * 6 - 15) + 10)
        w = z * z * z * (z * (z * 6 - 15) + 10)

        A = self.permutation[X] + Y
        AA = self.permutation[A] + Z
        AB = self.permutation[A + 1] + Z
        B = self.permutation[X + 1] + Y
        BA = self.permutation[B] + Z
        BB = self.permutation[B + 1] + Z

        def grad3d(hash_val, x, y, z):
            h = hash_val & 15
            u = x if h < 8 else y
            v = y if h < 4 else (x if h == 12 or h == 14 else z)
            return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

        return self._lerp(
            w,
            self._lerp(
                v,
                self._lerp(
                    u,
                    grad3d(self.permutation[AA], x, y, z),
                    grad3d(self.permutation[BA], x - 1, y, z),
                ),
                self._lerp(
                    u,
                    grad3d(self.permutation[AB], x, y - 1, z),
                    grad3d(self.permutation[BB], x - 1, y - 1, z),
                ),
            ),
            self._lerp(
                v,
                self._lerp(
                    u,
                    grad3d(self.permutation[AA + 1], x, y, z - 1),
                    grad3d(self.permutation[BA + 1], x - 1, y, z - 1),
                ),
                self._lerp(
                    u,
                    grad3d(self.permutation[AB + 1], x, y - 1, z - 1),
                    grad3d(self.permutation[BB + 1], x - 1, y - 1, z - 1),
                ),
            ),
        )

    def _lerp(self, t, a, b):
        return a + t * (b - a)

    def fbm(self, x, y, z, octaves=4, persistence=0.5, lacunarity=2.0):
        """Fractional Brownian Motion for more complex noise patterns"""
        total = 0.0
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0

        for i in range(octaves):
            total += (
                self.noise3d(x * frequency, y * frequency, z * frequency) * amplitude
            )
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        return total / max_value


class TreeGenerator:
    def __init__(self, noise_gen: NoiseGenerator):
        self.noise_gen = noise_gen

    def should_generate_tree(self, world_x: int, world_z: int) -> bool:
        spacing = 0.5
        tree_noise = self.noise_gen.noise2d(world_x * spacing, world_z * spacing)
        return tree_noise > 0.5

    def generate_tree(self, chunk, world_x: int, surface_y: int, world_z: int):
        if surface_y < 16 or surface_y > 25:
            return

        map_variation = 0.1
        raw_noise = self.noise_gen.noise2d(
            world_x * map_variation, world_z * map_variation
        )
        tree_height = 4 + int((raw_noise + 1) * 2)

        trunk_height = max(3, tree_height - 2)

        local_x = world_x - chunk.chunk_x * Chunk.CHUNK_SIZE
        local_z = world_z - chunk.chunk_z * Chunk.CHUNK_SIZE

        if not (0 <= local_x < Chunk.CHUNK_SIZE and 0 <= local_z < Chunk.CHUNK_SIZE):
            return

        for y in range(
            surface_y + 1, min(surface_y + trunk_height + 1, Chunk.CHUNK_HEIGHT)
        ):
            chunk.blocks[(local_x, y, local_z)] = 5  # Tree trunk block ID

        leaves_start = surface_y + max(2, trunk_height - 1)
        leaves_end = min(surface_y + trunk_height + 1, Chunk.CHUNK_HEIGHT - 1)

        for y in range(leaves_start, leaves_end + 1):
            if y == leaves_start:
                leaf_type = "square"
                radius = 1
            elif y == leaves_end:
                leaf_type = "circle"
                radius = 1
            else:
                leaf_type = "circle"
                radius = 1

            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx == 0 and dz == 0 and y < leaves_end:
                        continue

                    place_leaf = False
                    if leaf_type == "square":
                        place_leaf = True
                    elif leaf_type == "circle":
                        distance = dx * dx + dz * dz
                        place_leaf = distance <= radius * radius

                    if place_leaf:
                        leaf_x = local_x + dx
                        leaf_z = local_z + dz

                        if (
                            0 <= leaf_x < Chunk.CHUNK_SIZE
                            and 0 <= leaf_z < Chunk.CHUNK_SIZE
                            and 0 <= y < Chunk.CHUNK_HEIGHT
                        ):
                            existing_block = chunk.blocks.get((leaf_x, y, leaf_z), 0)
                            if existing_block == 0:
                                chunk.blocks[(leaf_x, y, leaf_z)] = 6  # Leaves block ID


class Chunk:
    CHUNK_SIZE = 8
    CHUNK_HEIGHT = 32

    def __init__(
        self,
        chunk_x: int,
        chunk_z: int,
        noise_gen: NoiseGenerator,
        texture_manager: TextureManager,
        chunk_manager=None,
    ):
        self.chunk_x = chunk_x
        self.chunk_z = chunk_z
        self.blocks = {}
        self.noise_gen = noise_gen
        self.texture_manager = texture_manager
        self.chunk_manager = chunk_manager
        self.tree_generator = TreeGenerator(noise_gen)
        self._visible_faces_cache = {}
        self._cache_dirty = True
        self._face_culling_computed = False
        self.generate_terrain()

    def _precompute_face_culling(self):
        if self._face_culling_computed and not self._cache_dirty:
            return

        face_offsets = {
            "front": (0, 0, -1),
            "back": (0, 0, 1),
            "left": (-1, 0, 0),
            "right": (1, 0, 0),
            "top": (0, 1, 0),
            "bottom": (0, -1, 0),
        }

        self._visible_faces_cache.clear()

        solid_blocks = set(self.blocks.items())

        for (x, y, z), block_type in solid_blocks:
            if block_type == 0:
                continue

            visible_faces = []
            for face_name, (dx, dy, dz) in face_offsets.items():
                adj_pos = (x + dx, y + dy, z + dz)

                if adj_pos in self.blocks:
                    if self.blocks[adj_pos] == 0:
                        visible_faces.append(face_name)
                else:
                    adjacent_block = self._get_adjacent_block_safe(
                        x + dx, y + dy, z + dz
                    )
                    if adjacent_block == 0:
                        visible_faces.append(face_name)

            if visible_faces:
                self._visible_faces_cache[(x, y, z)] = visible_faces

        self._face_culling_computed = True
        self._cache_dirty = False

    def _get_adjacent_block_safe(self, x: int, y: int, z: int) -> int:
        if not (0 <= y < self.CHUNK_HEIGHT):
            return 0

        if 0 <= x < self.CHUNK_SIZE and 0 <= z < self.CHUNK_SIZE:
            return self.blocks.get((x, y, z), 0)
        elif self.chunk_manager:
            world_x = self.chunk_x * self.CHUNK_SIZE + x
            world_z = self.chunk_z * self.CHUNK_SIZE + z
            return self.chunk_manager.get_block_world_safe(world_x, y, world_z)
        return 0

    def get_world_pos(self, local_x: int, local_z: int):
        return (
            self.chunk_x * self.CHUNK_SIZE + local_x,
            self.chunk_z * self.CHUNK_SIZE + local_z,
        )

    def _generate_complex_caves(self, world_x: int, world_y: int, world_z: int) -> bool:
        if world_y < 3 or world_y > 20:
            return False

        scale1 = 0.04
        cave_noise1 = self.noise_gen.noise3d(
            world_x * scale1, world_y * scale1 * 0.7, world_z * scale1
        )

        height_factor = max(0.2, 1.0 - (world_y / 22.0))
        base_threshold = -0.4

        return cave_noise1 > base_threshold and cave_noise1 < (
            base_threshold + 0.25 * height_factor
        )

    def generate_terrain(self):
        surface_heights = {}

        height_cache = {}
        for x in range(self.CHUNK_SIZE):
            for z in range(self.CHUNK_SIZE):
                world_x, world_z = self.get_world_pos(x, z)
                height = self.get_height(world_x, world_z)
                surface_heights[(x, z)] = height
                height_cache[(x, z)] = height

        blocks_to_generate = []
        for x in range(self.CHUNK_SIZE):
            for z in range(self.CHUNK_SIZE):
                world_x, world_z = self.get_world_pos(x, z)
                height = height_cache[(x, z)]

                for y in range(min(height + 1, self.CHUNK_HEIGHT)):
                    if y <= height:
                        block_type = self.get_block_type(world_x, y, world_z, height)
                        if block_type != 0:
                            blocks_to_generate.append(
                                ((x, y, z), (world_x, y, world_z), block_type)
                            )

        for (x, y, z), (world_x, world_y, world_z), block_type in blocks_to_generate:
            if not self._generate_complex_caves(world_x, world_y, world_z):
                self.blocks[(x, y, z)] = block_type

        for x in range(self.CHUNK_SIZE):
            for z in range(self.CHUNK_SIZE):
                world_x, world_z = self.get_world_pos(x, z)
                surface_y = surface_heights[(x, z)]

                if (
                    surface_y > 15
                    and self.blocks.get((x, surface_y, z)) == 2
                    and self.tree_generator.should_generate_tree(world_x, world_z)
                ):
                    self.tree_generator.generate_tree(self, world_x, surface_y, world_z)

    def get_height(self, world_x: int, world_z: int) -> int:
        factor = self.chunk_manager.dramaticness if self.chunk_manager else 1.0
        height_noise = self.noise_gen.noise2d(world_x * 0.05, world_z * 0.05)
        base_height = 16 + int(height_noise * 4 * factor)
        return max(1, min(base_height, self.CHUNK_HEIGHT - 1))

    def get_block_type(
        self, world_x: int, y: int, world_z: int, surface_height: int
    ) -> int:
        if y == surface_height and surface_height > 15:
            return 2  # Grass block
        elif y >= surface_height - 2 and surface_height > 15:
            return 1  # Dirt block
        elif y < 2:
            return 4  # Bedrock block
        elif y < 5:
            return 3  # Stone block
        else:
            return 3  # Stone block

    def get_block(self, x: int, y: int, z: int) -> int:
        if (
            0 <= x < self.CHUNK_SIZE
            and 0 <= y < self.CHUNK_HEIGHT
            and 0 <= z < self.CHUNK_SIZE
        ):
            return self.blocks.get((x, y, z), 0)
        return 0

    def get_block_with_neighbors(self, x: int, y: int, z: int) -> int:
        return self._get_adjacent_block_safe(x, y, z)

    def get_visible_faces(self, x: int, y: int, z: int) -> List[str]:
        if self._cache_dirty or not self._face_culling_computed:
            self._precompute_face_culling()
        return self._visible_faces_cache.get((x, y, z), [])

    def mark_dirty(self):
        self._cache_dirty = True
        self._face_culling_computed = False


class ChunkManager:
    def __init__(
        self,
        texture_manager: TextureManager,
        render_distance: int = 2,
        dramaticness: float = 1.0,
        seed: int = None,
    ):
        self.chunks = {}
        self.texture_manager = texture_manager
        self.noise_gen = NoiseGenerator(seed=seed)
        self.render_distance = render_distance
        self.dramaticness = dramaticness
        self.block_cache = {}
        self._last_camera_chunk = None
        self._cached_visible_chunks = []
        self._cache_distance = 0
        self._generating_chunks = set()

    def get_chunk_coords(self, world_x: float, world_z: float) -> Tuple[int, int]:
        return int(floor(world_x / Chunk.CHUNK_SIZE)), int(
            floor(world_z / Chunk.CHUNK_SIZE)
        )

    def load_chunk(self, chunk_x: int, chunk_z: int) -> Chunk:
        key = (chunk_x, chunk_z)
        if key not in self.chunks and key not in self._generating_chunks:
            self._generating_chunks.add(key)
            chunk = Chunk(chunk_x, chunk_z, self.noise_gen, self.texture_manager, self)
            self.chunks[key] = chunk
            self._generating_chunks.remove(key)
            return chunk
        return self.chunks.get(key)

    def get_block_world(self, world_x: int, world_y: int, world_z: int) -> int:
        cache_key = (world_x, world_y, world_z)
        if cache_key in self.block_cache:
            return self.block_cache[cache_key]

        chunk_x, chunk_z = self.get_chunk_coords(world_x, world_z)
        chunk_key = (chunk_x, chunk_z)

        if chunk_key in self.chunks:
            chunk = self.chunks[chunk_key]
            local_x = world_x - chunk_x * Chunk.CHUNK_SIZE
            local_z = world_z - chunk_z * Chunk.CHUNK_SIZE
            block = chunk.get_block(local_x, world_y, local_z)

            if len(self.block_cache) < 1000:
                self.block_cache[cache_key] = block

            return block

        return 0

    def get_block_world_safe(self, world_x: int, world_y: int, world_z: int) -> int:
        if not (0 <= world_y < Chunk.CHUNK_HEIGHT):
            return 0

        chunk_x, chunk_z = self.get_chunk_coords(world_x, world_z)
        chunk_key = (chunk_x, chunk_z)

        if chunk_key not in self.chunks:
            return 0

        chunk = self.chunks[chunk_key]
        local_x = world_x - chunk_x * Chunk.CHUNK_SIZE
        local_z = world_z - chunk_z * Chunk.CHUNK_SIZE

        return chunk.get_block(local_x, world_y, local_z)

    def remove_block(self, world_x: int, world_y: int, world_z: int) -> bool:
        """Remove a block from the world"""
        if not (0 <= world_y < Chunk.CHUNK_HEIGHT):
            return False

        chunk_x, chunk_z = self.get_chunk_coords(world_x, world_z)
        chunk_key = (chunk_x, chunk_z)

        if chunk_key in self.chunks:
            chunk = self.chunks[chunk_key]
            local_x = world_x - chunk_x * Chunk.CHUNK_SIZE
            local_z = world_z - chunk_z * Chunk.CHUNK_SIZE

            if (local_x, world_y, local_z) in chunk.blocks:
                del chunk.blocks[(local_x, world_y, local_z)]
                chunk.mark_dirty()
                self.refresh_chunk_faces(chunk_x, chunk_z)

                cache_key = (world_x, world_y, world_z)
                if cache_key in self.block_cache:
                    del self.block_cache[cache_key]

                return True

        return False

    def create_block(
        self, world_x: int, world_y: int, world_z: int, block_type: int
    ) -> bool:
        """Create a block in the world"""
        if not (0 <= world_y < Chunk.CHUNK_HEIGHT) or block_type <= 0:
            return False

        chunk_x, chunk_z = self.get_chunk_coords(world_x, world_z)
        chunk_key = (chunk_x, chunk_z)

        if chunk_key not in self.chunks:
            self.load_chunk(chunk_x, chunk_z)

        if chunk_key in self.chunks:
            chunk = self.chunks[chunk_key]
            local_x = world_x - chunk_x * Chunk.CHUNK_SIZE
            local_z = world_z - chunk_z * Chunk.CHUNK_SIZE

            if 0 <= local_x < Chunk.CHUNK_SIZE and 0 <= local_z < Chunk.CHUNK_SIZE:
                chunk.blocks[(local_x, world_y, local_z)] = block_type
                chunk.mark_dirty()
                self.refresh_chunk_faces(chunk_x, chunk_z)

                cache_key = (world_x, world_y, world_z)
                self.block_cache[cache_key] = block_type

                return True

        return False

    def _generate_circle_chunks(
        self, center_x: int, center_z: int, radius: int
    ) -> List[Tuple[int, int]]:
        chunks = []
        radius_sq = radius * radius

        for dx in range(-radius, radius + 1):
            max_dz_sq = radius_sq - dx * dx
            if max_dz_sq < 0:
                continue

            max_dz = int(sqrt(max_dz_sq))

            for dz in range(-max_dz, max_dz + 1):
                if dx * dx + dz * dz <= radius_sq:
                    chunks.append((center_x + dx, center_z + dz))

        return chunks

    def get_visible_chunks(self, camera_pos: Vector3) -> List[Chunk]:
        camera_chunk_x, camera_chunk_z = self.get_chunk_coords(
            camera_pos.x, camera_pos.z
        )

        current_camera_chunk = (camera_chunk_x, camera_chunk_z)
        if (
            self._last_camera_chunk == current_camera_chunk
            and self._cache_distance == self.render_distance
            and self._cached_visible_chunks
        ):
            return self._cached_visible_chunks

        chunk_coords = self._generate_circle_chunks(
            camera_chunk_x, camera_chunk_z, self.render_distance
        )

        visible_chunks = []
        for chunk_x, chunk_z in chunk_coords:
            chunk = self.load_chunk(chunk_x, chunk_z)
            if chunk is not None:
                visible_chunks.append(chunk)

        self._last_camera_chunk = current_camera_chunk
        self._cached_visible_chunks = visible_chunks
        self._cache_distance = self.render_distance

        return visible_chunks

    def unload_distant_chunks(self, camera_pos: Vector3, unload_distance: int = None):
        if unload_distance is None:
            unload_distance = self.render_distance + 2

        camera_chunk_x, camera_chunk_z = self.get_chunk_coords(
            camera_pos.x, camera_pos.z
        )

        chunks_to_unload = []
        unload_distance_sq = unload_distance * unload_distance

        for chunk_x, chunk_z in list(self.chunks.keys()):
            dx = chunk_x - camera_chunk_x
            dz = chunk_z - camera_chunk_z
            distance_sq = dx * dx + dz * dz

            if distance_sq > unload_distance_sq:
                chunks_to_unload.append((chunk_x, chunk_z))

        for chunk_key in chunks_to_unload:
            if chunk_key in self.chunks:
                del self.chunks[chunk_key]

        if chunks_to_unload:
            self._last_camera_chunk = None
            self._cached_visible_chunks = []
            self.block_cache.clear()

    def get_chunk_load_stats(self) -> Dict[str, int]:
        return {
            "loaded_chunks": len(self.chunks),
            "cached_blocks": len(self.block_cache),
            "cached_visible_chunks": len(self._cached_visible_chunks),
        }

    def refresh_chunk_faces(self, chunk_x: int, chunk_z: int):
        key = (chunk_x, chunk_z)
        if key in self.chunks:
            self.chunks[key].mark_dirty()
            for dx in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dz == 0:
                        continue
                    neighbor_key = (chunk_x + dx, chunk_z + dz)
                    if neighbor_key in self.chunks:
                        self.chunks[neighbor_key].mark_dirty()


# Player
@dataclass
class RenderFace:
    """Face to be rendered with depth sorting and lighting"""

    distance: float
    projected: List[Tuple[float, float]]
    color: Tuple[int, int, int]
    ambient_occlusion: float = 1.0
    highlight: bool = False


class Player:
    def __init__(
        self,
        pos,
        chunk_manager,
        height=1.8,
        width=0.6,
        gravity=2.0,
        move_coefficient=0.98,
        max_fall_speed=20.0,
        interaction_range=5.0,
    ):
        self.pos = pos
        self.chunk_manager = chunk_manager
        self.height = height
        self.width = width
        self.gravity = gravity
        self.move_coefficient = move_coefficient
        self.max_fall_speed = max_fall_speed
        self.interaction_range = interaction_range
        self.velocity = Vector3(0, 0, 0)
        self.on_ground = False
        self.ground_offset = 0.1

    def get_head_position(self):
        return Vector3(self.pos.x, self.pos.y + self.height, self.pos.z)

    def raycast_to_block(self, direction: Vector3, max_distance: float = 5.0):
        """Cast a ray from the player's head position to find the first block hit"""
        start_pos = self.get_head_position()
        step_size = 0.1
        steps = int(max_distance / step_size)

        for i in range(steps):
            current_pos = start_pos + direction * (i * step_size)
            block_x = int(floor(current_pos.x))
            block_y = int(floor(current_pos.y))
            block_z = int(floor(current_pos.z))

            block_type = self.chunk_manager.get_block_world_safe(
                block_x, block_y, block_z
            )
            if block_type != 0:
                return Vector3(block_x, block_y, block_z)

        return None

    def raycast_to_placement_position(
        self, direction: Vector3, max_distance: float = 5.0
    ):
        """Cast a ray to find the position where a block should be placed"""
        start_pos = self.get_head_position()
        step_size = 0.1
        steps = int(max_distance / step_size)
        last_empty_pos = None

        for i in range(steps):
            current_pos = start_pos + direction * (i * step_size)
            block_x = int(floor(current_pos.x))
            block_y = int(floor(current_pos.y))
            block_z = int(floor(current_pos.z))

            block_type = self.chunk_manager.get_block_world_safe(
                block_x, block_y, block_z
            )

            if block_type == 0:
                last_empty_pos = Vector3(block_x, block_y, block_z)
            else:
                if last_empty_pos is not None:
                    return last_empty_pos
                break

        return None

    def break_block_at_crosshair(self, direction: Vector3):
        """Break the block the player is looking at"""
        block_pos = self.raycast_to_block(direction, self.interaction_range)
        if block_pos:
            return self.chunk_manager.remove_block(
                int(block_pos.x), int(block_pos.y), int(block_pos.z)
            )
        return False

    def place_block_at_crosshair(self, direction: Vector3, block_type: int = 3):
        """Place a block at the face the player is looking at"""
        placement_pos = self.raycast_to_placement_position(
            direction, self.interaction_range
        )
        if placement_pos:
            world_x, world_y, world_z = (
                int(placement_pos.x),
                int(placement_pos.y),
                int(placement_pos.z),
            )

            player_collision_box = [
                (self.pos.x - self.width / 2, self.pos.y, self.pos.z - self.width / 2),
                (self.pos.x + self.width / 2, self.pos.y, self.pos.z - self.width / 2),
                (self.pos.x - self.width / 2, self.pos.y, self.pos.z + self.width / 2),
                (self.pos.x + self.width / 2, self.pos.y, self.pos.z + self.width / 2),
                (
                    self.pos.x - self.width / 2,
                    self.pos.y + self.height,
                    self.pos.z - self.width / 2,
                ),
                (
                    self.pos.x + self.width / 2,
                    self.pos.y + self.height,
                    self.pos.z - self.width / 2,
                ),
                (
                    self.pos.x - self.width / 2,
                    self.pos.y + self.height,
                    self.pos.z + self.width / 2,
                ),
                (
                    self.pos.x + self.width / 2,
                    self.pos.y + self.height,
                    self.pos.z + self.width / 2,
                ),
            ]

            for px, py, pz in player_collision_box:
                if (
                    int(floor(px)) == world_x
                    and int(floor(py)) == world_y
                    and int(floor(pz)) == world_z
                ):
                    return False

            return self.chunk_manager.create_block(
                world_x, world_y, world_z, block_type
            )
        return False

    def break_block(self, block_pos: Vector3):
        """Remove a block from the world"""
        if not block_pos:
            return False

        world_x, world_y, world_z = int(block_pos.x), int(block_pos.y), int(block_pos.z)

        chunk_x, chunk_z = self.chunk_manager.get_chunk_coords(world_x, world_z)
        chunk_key = (chunk_x, chunk_z)

        if chunk_key in self.chunk_manager.chunks:
            chunk = self.chunk_manager.chunks[chunk_key]
            local_x = world_x - chunk_x * Chunk.CHUNK_SIZE
            local_z = world_z - chunk_z * Chunk.CHUNK_SIZE

            if (local_x, world_y, local_z) in chunk.blocks:
                del chunk.blocks[(local_x, world_y, local_z)]
                chunk.mark_dirty()
                self.chunk_manager.refresh_chunk_faces(chunk_x, chunk_z)

                cache_key = (world_x, world_y, world_z)
                if cache_key in self.chunk_manager.block_cache:
                    del self.chunk_manager.block_cache[cache_key]

                return True

        return False

    def check_collision(self, new_pos):
        half_width = self.width / 2

        collision_points = [
            (new_pos.x - half_width, new_pos.y, new_pos.z - half_width),
            (new_pos.x + half_width, new_pos.y, new_pos.z - half_width),
            (new_pos.x - half_width, new_pos.y, new_pos.z + half_width),
            (new_pos.x + half_width, new_pos.y, new_pos.z + half_width),
            (new_pos.x - half_width, new_pos.y + self.height, new_pos.z - half_width),
            (new_pos.x + half_width, new_pos.y + self.height, new_pos.z - half_width),
            (new_pos.x - half_width, new_pos.y + self.height, new_pos.z + half_width),
            (new_pos.x + half_width, new_pos.y + self.height, new_pos.z + half_width),
        ]

        for x, y, z in collision_points:
            block = self.chunk_manager.get_block_world_safe(
                int(floor(x)), int(floor(y)), int(floor(z))
            )
            if block != 0:
                return True
        return False

    def check_collision_axis(self, new_pos, axis):
        test_pos = Vector3(self.pos.x, self.pos.y, self.pos.z)
        setattr(test_pos, axis, getattr(new_pos, axis))
        return self.check_collision(test_pos)

    def find_ground_height(self, x, z):
        for y in range(int(self.pos.y + self.height), -1, -1):
            block = self.chunk_manager.get_block_world_safe(
                int(floor(x)), int(floor(y)), int(floor(z))
            )
            if block != 0:
                return float(y + 1)
        return 0.0

    def update_physics(self, deltatime):
        if not self.on_ground:
            self.velocity.y -= self.gravity * deltatime
            if self.velocity.y < -self.max_fall_speed:
                self.velocity.y = -self.max_fall_speed

        new_pos = self.pos + self.velocity * deltatime

        half_width = self.width / 2
        ground_points = [
            (self.pos.x - half_width, self.pos.z - half_width),
            (self.pos.x + half_width, self.pos.z - half_width),
            (self.pos.x - half_width, self.pos.z + half_width),
            (self.pos.x + half_width, self.pos.z + half_width),
        ]

        ground_height = max(self.find_ground_height(x, z) for x, z in ground_points)

        if new_pos.y <= ground_height + self.ground_offset:
            new_pos.y = ground_height + self.ground_offset
            self.velocity.y = 0
            self.on_ground = True
        else:
            self.on_ground = False

        if not self.check_collision(new_pos):
            self.pos = new_pos
        else:
            final_pos = Vector3(self.pos.x, self.pos.y, self.pos.z)

            if not self.check_collision_axis(new_pos, "x"):
                final_pos.x = new_pos.x

            if not self.check_collision_axis(
                Vector3(final_pos.x, new_pos.y, final_pos.z), "y"
            ):
                final_pos.y = new_pos.y
            else:
                self.velocity.y = -1.0 * deltatime

            if not self.check_collision_axis(
                Vector3(final_pos.x, final_pos.y, new_pos.z), "z"
            ):
                final_pos.z = new_pos.z

            self.pos = final_pos

        damping_factor = pow(self.move_coefficient, deltatime)
        self.velocity.x *= damping_factor
        self.velocity.z *= damping_factor

    def move(self, steps, direction, deltatime):
        movement = direction * steps * deltatime
        self.velocity.x += movement.x
        self.velocity.z += movement.z

    def strafe(self, steps, right_vector, deltatime):
        movement = right_vector * steps * deltatime
        self.velocity.x += movement.x
        self.velocity.z += movement.z

    def move_axis(self, world, deltatime):
        self.velocity += world * deltatime

    def move_relative(
        self, movement, camera_direction, deltatime, horizontal_only=False
    ):
        if horizontal_only:
            forward = Vector3(camera_direction.x, 0, camera_direction.z)
            if forward.magnitude() > 0:
                forward = forward.normalize()
            else:
                forward = Vector3(0, 0, 1)
        else:
            forward = camera_direction

        up = Vector3(0, 1, 0)
        right = forward.cross(up)
        if right.magnitude() > 0:
            right = right.normalize()
        else:
            right = Vector3(1, 0, 0)

        if not horizontal_only:
            up = right.cross(forward).normalize()

        absolute_movement = Vector3(0, 0, 0)
        absolute_movement += right * movement.x
        if horizontal_only:
            absolute_movement.y += movement.y
        else:
            absolute_movement += up * movement.y
        absolute_movement += forward * movement.z

        self.velocity += absolute_movement * deltatime

    def jump(self, strength=6.0, _forced=False):
        if _forced or self.on_ground:
            self.velocity.y = strength
            self.on_ground = False


class Camera:
    __slots__ = (
        "pos",
        "direction",
        "pen",
        "canvas",
        "screen",
        "texture_manager",
        "chunk_manager",
        "face_normals",
        "face_ao",
        "cube_vertices_pattern",
        "faces_data",
        "_projection_cache",
        "_frustum_cache",
        "_render_stats",
        "_last_pos",
        "_last_dir",
        "_static_chunks_cache",
        "player",
        "highlighted_block",
    )

    def __init__(self, player, direction=None, screen_width=800, screen_height=600):
        self.player = player
        self.pos = player.get_head_position()
        self.direction = direction.normalize() if direction else Vector3(0, 0, 1)
        self.highlighted_block = None

        turtle.tracer(0, 0)
        turtle.setup(screen_width, screen_height)
        turtle.colormode(255)
        self.screen = turtle.getscreen()
        self.screen.cv._rootwindow.resizable(False, False)

        turtle.hideturtle()
        self.pen = turtle.Turtle()
        self.pen.speed(0)
        self.pen.hideturtle()
        self.pen.penup()
        self.canvas = self.pen.screen.getcanvas()
        self.texture_manager = TextureManager()
        self.chunk_manager = player.chunk_manager

        self.face_normals = {
            "front": Vector3(0, 0, -1),
            "back": Vector3(0, 0, 1),
            "left": Vector3(-1, 0, 0),
            "right": Vector3(1, 0, 0),
            "top": Vector3(0, 1, 0),
            "bottom": Vector3(0, -1, 0),
        }

        self.face_ao = {
            "top": 1.0,
            "front": 0.9,
            "back": 0.9,
            "left": 0.8,
            "right": 0.8,
            "bottom": 0.6,
        }

        self.cube_vertices_pattern = (
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(1, 1, 0),
            Vector3(0, 1, 0),
            Vector3(0, 0, 1),
            Vector3(1, 0, 1),
            Vector3(1, 1, 1),
            Vector3(0, 1, 1),
        )

        self.faces_data = {
            "front": ([0, 1, 2, 3], Vector3(0, 0, -0.5)),
            "back": ([5, 4, 7, 6], Vector3(0, 0, 0.5)),
            "left": ([4, 0, 3, 7], Vector3(-0.5, 0, 0)),
            "right": ([1, 5, 6, 2], Vector3(0.5, 0, 0)),
            "top": ([3, 2, 6, 7], Vector3(0, 0.5, 0)),
            "bottom": ([4, 5, 1, 0], Vector3(0, -0.5, 0)),
        }

        self._projection_cache = {}
        self._frustum_cache = {}
        self._render_stats = {"faces_rendered": 0, "faces_culled": 0}
        self._last_pos = None
        self._last_dir = None
        self._static_chunks_cache = {}

    def update_position(self):
        self.pos = self.player.get_head_position()
        self.highlighted_block = self.player.raycast_to_block(self.direction)
        self._clear_position_caches()

    def calculate_relative(self, pos, horizontal_only=False):
        if horizontal_only:
            forward = Vector3(self.direction.x, 0, self.direction.z)
            if forward.magnitude() > 0:
                forward = forward.normalize()
            else:
                forward = Vector3(0, 0, 1)
        else:
            forward = self.direction

        up = Vector3(0, 1, 0)
        right = forward.cross(up)
        if right.magnitude() > 0:
            right = right.normalize()
        else:
            right = Vector3(1, 0, 0)

        if not horizontal_only:
            up = right.cross(forward).normalize()

        absolute_movement = Vector3(0, 0, 0)
        absolute_movement += right * pos.x
        if horizontal_only:
            absolute_movement.y += pos.y
        else:
            absolute_movement += up * pos.y
        absolute_movement += forward * pos.z

        return absolute_movement

    def rotate(self, pitch_delta, yaw_delta):
        """Rotate camera with hard pitch clamping to prevent gimbal lock"""
        current_pitch = asin(max(-1, min(1, self.direction.y)))

        new_pitch = current_pitch + pitch_delta
        max_pitch = pi / 2 - 0.01
        new_pitch = max(-max_pitch, min(max_pitch, new_pitch))

        horizontal_length = sqrt(self.direction.x**2 + self.direction.z**2)
        if horizontal_length > 0:
            current_yaw = atan2(self.direction.x, self.direction.z)
        else:
            current_yaw = 0

        new_yaw = current_yaw - yaw_delta

        cos_pitch = cos(new_pitch)
        self.direction = Vector3(
            cos_pitch * sin(new_yaw), sin(new_pitch), cos_pitch * cos(new_yaw)
        )

        self._clear_position_caches()

    def _clear_position_caches(self):
        self._last_pos = None
        self._last_dir = None
        if len(self._projection_cache) > 200:
            self._projection_cache.clear()
        if len(self._frustum_cache) > 50:
            self._frustum_cache.clear()

    def is_face_visible(self, face_center, face_normal):
        view_vec = (face_center - self.pos).normalize()
        return face_normal.dot(view_vec) < 0

    def get_block_texture(self, block_type: int, face_name: str) -> int:
        block = self.texture_manager.get_block(block_type)
        if face_name == "top":
            return block.top_texture
        elif face_name == "bottom":
            return block.bottom_texture
        else:
            return block.side_texture

    def clip_polygon_near_plane(self, vertices_3d, near_plane=0.1):
        """Optimized polygon clipping with early exits and cached computations"""
        if not vertices_3d or len(vertices_3d) < 3:
            return []

        cam_vertices = []
        z_values = []
        for vertex in vertices_3d:
            cam_pos = vertex - self.pos
            z = cam_pos.dot(self.direction)
            cam_vertices.append(cam_pos)
            z_values.append(z)

        if all(z < near_plane for z in z_values):
            return []

        if all(z >= near_plane for z in z_values):
            return vertices_3d

        output = []
        num_vertices = len(vertices_3d)

        for i in range(num_vertices):
            current = vertices_3d[i]
            previous = vertices_3d[i - 1]

            current_z = z_values[i]
            previous_z = z_values[i - 1]

            current_inside = current_z >= near_plane
            previous_inside = previous_z >= near_plane

            if current_inside:
                if not previous_inside:
                    t = (near_plane - previous_z) / (current_z - previous_z)
                    intersection = previous + (current - previous) * t
                    output.append(intersection)
                output.append(current)
            elif previous_inside:
                t = (near_plane - previous_z) / (current_z - previous_z)
                intersection = previous + (current - previous) * t
                output.append(intersection)

        return output if len(output) >= 3 else []

    def project_point_batch(self, points, fov=90, screen_width=800, screen_height=600):
        """Batch project multiple points for better performance"""
        if not points:
            return []

        forward = self.direction
        up = Vector3(0, 1, 0)
        right = forward.cross(up)

        if right.magnitude() == 0:
            return [None] * len(points)

        right = right.normalize()
        actual_up = right.cross(forward)

        fov_rad = radians(fov)
        f = 1.0 / tan(fov_rad * 0.5)
        screen_scale_x = screen_width * 0.5
        screen_scale_y = screen_height * 0.5
        aspect_ratio = screen_height / screen_width

        results = []
        for point in points:
            rel = point - self.pos

            x = rel.dot(right)
            y = rel.dot(actual_up)
            z = rel.dot(forward)

            if z == 0:
                z = 0.001

            screen_x = (x * f / z) * aspect_ratio
            screen_y = y * f / z

            screen_x = screen_x * screen_scale_x + screen_scale_x
            screen_y = screen_y * screen_scale_y + screen_scale_y

            results.append((screen_x - screen_scale_x, screen_y - screen_scale_y, z))

        return results

    def is_face_in_frustum_fast(
        self, block_center: Vector3, max_distance: float = 25.0
    ) -> bool:
        """Optimized frustum culling with distance-based early exits"""
        rel = block_center - self.pos
        distance_sq = rel.x * rel.x + rel.y * rel.y + rel.z * rel.z

        max_distance_sq = max_distance * max_distance
        if distance_sq > max_distance_sq:
            return False

        if distance_sq < 25:
            return True

        forward_dot = rel.dot(self.direction)
        if forward_dot <= -1.0:
            return False

        if distance_sq > 100:
            right = self.direction.cross(Vector3(0, 1, 0))
            if right.magnitude() == 0:
                return True

            right = right.normalize()
            right_dot = abs(rel.dot(right))
            up = right.cross(self.direction).normalize()
            up_dot = abs(rel.dot(up))

            distance = sqrt(distance_sq)
            fov_factor = distance * 1.2

            return right_dot <= fov_factor and up_dot <= fov_factor

        return True

    def render_world(
        self,
        fov=90,
        screen_width=800,
        screen_height=600,
        max_render_distance=20,
        min_brightness=0.2,
        render_distance=2,
    ):
        """Optimized render_world with batched operations and reduced allocations"""
        self.pen.clear()

        if self.chunk_manager.render_distance != render_distance:
            self.chunk_manager.render_distance = render_distance

        visible_chunks = self.chunk_manager.get_visible_chunks(self.pos)
        render_faces = []
        faces_processed = 0
        faces_culled = 0

        camera_chunk_x, camera_chunk_z = self.chunk_manager.get_chunk_coords(
            self.pos.x, self.pos.z
        )

        max_render_distance_sq = max_render_distance * max_render_distance
        render_distance_sq = render_distance * render_distance

        for chunk in visible_chunks:
            chunk_distance_sq = (chunk.chunk_x - camera_chunk_x) ** 2 + (
                chunk.chunk_z - camera_chunk_z
            ) ** 2

            if chunk_distance_sq > render_distance_sq:
                continue

            blocks_to_render = []

            for (local_x, local_y, local_z), block_type in chunk.blocks.items():
                world_x = chunk.chunk_x * Chunk.CHUNK_SIZE + local_x
                world_y = local_y
                world_z = chunk.chunk_z * Chunk.CHUNK_SIZE + local_z

                block_center = Vector3(world_x + 0.5, world_y + 0.5, world_z + 0.5)

                rel = block_center - self.pos
                distance_sq = rel.x * rel.x + rel.y * rel.y + rel.z * rel.z

                if distance_sq > max_render_distance_sq:
                    faces_culled += 1
                    continue

                if not self.is_face_in_frustum_fast(block_center, max_render_distance):
                    faces_culled += 1
                    continue

                visible_faces = chunk.get_visible_faces(local_x, local_y, local_z)
                if not visible_faces:
                    faces_culled += 1
                    continue

                blocks_to_render.append(
                    (
                        (world_x, world_y, world_z),
                        block_type,
                        visible_faces,
                        sqrt(distance_sq),
                        block_center,
                    )
                )

            for (
                (world_x, world_y, world_z),
                block_type,
                visible_faces,
                distance,
                block_center,
            ) in blocks_to_render:
                vertices = [
                    Vector3(world_x, world_y, world_z) + vertex_offset
                    for vertex_offset in self.cube_vertices_pattern
                ]

                for face_name in visible_faces:
                    faces_processed += 1
                    indices, offset = self.faces_data[face_name]
                    face_center = block_center + offset
                    face_normal = self.face_normals[face_name]

                    view_vec = face_center - self.pos
                    if face_normal.dot(view_vec) >= 0:
                        faces_culled += 1
                        continue

                    face_vertices_3d = [vertices[i] for i in indices]

                    clipped_vertices_3d = self.clip_polygon_near_plane(
                        face_vertices_3d, 0.1
                    )

                    if len(clipped_vertices_3d) < 3:
                        faces_culled += 1
                        continue

                    projections = self.project_point_batch(
                        clipped_vertices_3d, fov, screen_width, screen_height
                    )

                    projected = []
                    for proj in projections:
                        if proj is not None:
                            projected.append((proj[0], proj[1]))

                    if len(projected) >= 3:
                        texture_id = self.get_block_texture(block_type, face_name)
                        ambient_occlusion = self.face_ao[face_name]
                        color = self.texture_manager.get_color(
                            texture_id,
                            distance,
                            ambient_occlusion,
                            max_render_distance,
                            min_brightness,
                        )

                        highlight = (
                            self.highlighted_block == Vector3(world_x, world_y, world_z)
                            if self.highlighted_block
                            else False
                        )

                        render_faces.append(
                            RenderFace(
                                distance, projected, color, ambient_occlusion, highlight
                            )
                        )
                    else:
                        faces_culled += 1

        self._render_stats = {
            "faces_rendered": len(render_faces),
            "faces_culled": faces_culled,
            "faces_processed": faces_processed,
        }

        render_faces.sort(key=lambda x: x.distance, reverse=True)

        self.pen.color((0, 0, 0))
        self.pen.width(1)

        face_limit = 400
        for face in render_faces[-face_limit:]:
            if len(face.projected) >= 3:
                self.pen.fillcolor(face.color)
                self.pen.goto(*face.projected[0])
                if face.highlight:
                    self.pen.pendown()
                self.pen.begin_fill()
                for point in face.projected[1:]:
                    self.pen.goto(*point)
                self.pen.goto(*face.projected[0])
                self.pen.end_fill()
                if face.highlight:
                    self.pen.penup()

        self.canvas.create_line(0, 4, 0, -5, width=1, capstyle="butt", fill="#000")
        self.canvas.create_line(4, 0, -5, -0, width=1, capstyle="butt", fill="#000")


def mouse_init():
    global mouse_delta_x, mouse_delta_y, last_mouse_x, last_mouse_y, mouse_initialized, mouse_listener, current_mouse_x, current_mouse_y
    mouse_delta_x = 0
    mouse_delta_y = 0
    last_mouse_x = 0
    last_mouse_y = 0
    current_mouse_x = 400
    current_mouse_y = 300
    mouse_initialized = False

    if not using_mouse:

        def on_move(x, y):
            global current_mouse_x, current_mouse_y
            current_mouse_x = x
            current_mouse_y = y

        mouse_listener = MouseListener(on_move=on_move)
        mouse_listener.start()


def handle_movement(
    speed=6.0,
    sensitivity=0.01,
    player: Player = None,
    jump_strength: float = 10.0,
    cyote_time: int = 2,
):
    global mouse_initialized, cyote, current_mouse_x, current_mouse_y, last_mouse_x, last_mouse_y

    camera_movement = zero3()
    camera_angle = zero2()
    break_block = False
    place_block = False

    try:
        keys = {
            "w": keyboard.is_pressed("w"),
            "s": keyboard.is_pressed("s"),
            "a": keyboard.is_pressed("a"),
            "d": keyboard.is_pressed("d"),
            "ctrl": keyboard.is_pressed("ctrl"),
            "space": keyboard.is_pressed("space"),
            "left_click": keyboard.is_pressed("q"),
            "right_click": keyboard.is_pressed("e"),
        }

        if keys["w"]:
            camera_movement.z += speed
        if keys["s"]:
            camera_movement.z -= speed
        if keys["a"]:
            camera_movement.x -= speed
        if keys["d"]:
            camera_movement.x += speed

        if keys["left_click"]:
            break_block = True
        if keys["right_click"]:
            place_block = True

        if player is None:
            if keys["ctrl"]:
                camera_movement.y = -speed
            if keys["space"]:
                camera_movement.y = speed

        else:
            if player.on_ground:
                cyote = 0
            else:
                cyote += 1

            if keys["space"] and cyote <= cyote_time:
                cyote = cyote_time + 1
                player.jump(jump_strength, _forced=True)

    except Exception:
        pass

    if mouse_locked:
        try:
            if using_mouse:
                x, y = mouse.get_position()
                if mouse_initialized:
                    camera_angle.x -= (y - 600) * sensitivity
                    camera_angle.y += (x - 1200) * sensitivity
                    mouse.move(1200, 600)
                else:
                    mouse_initialized = True
                    mouse.move(1200, 600)
            else:
                if mouse_initialized:
                    camera_angle.x -= (current_mouse_y - last_mouse_y) * sensitivity
                    camera_angle.y += (current_mouse_x - last_mouse_x) * sensitivity
                else:
                    mouse_initialized = True

                last_mouse_x = current_mouse_x
                last_mouse_y = current_mouse_y
        except Exception:
            pass

    return camera_movement, camera_angle, break_block, place_block


# Performance monitoring
frame = 0
samples = 10
fps_data = []


def accurate_sleep(seconds: int | float):
    """Function to sleep accurately"""
    if seconds == 0:
        return
    elif seconds < 0.05:
        target = time.perf_counter() + seconds
        while time.perf_counter() < target:
            pass
    else:
        time.sleep(seconds)


def normalize_framerate(target):
    def decorator(func):
        def wrapped(*args, **kwargs):
            global frame, fps_data
            frame += 1
            start_time = time.time()
            deltatime = 1.0 / target

            result = func(deltatime, *args, **kwargs)

            camera.pen.color("#000000")
            uncapped = 1 / (time.time() - start_time)
            fps_data.append(uncapped)
            if len(fps_data) > samples:
                fps_data.pop(0)

            camera.pen.goto(-390, 260)
            camera.pen.write(
                f"POS: {camera.pos.x:.2f}, {camera.pos.y:.2f}, {camera.pos.z:.2f}"
            )
            camera.pen.goto(-390, 240)
            camera.pen.write(
                f"VEL: {player.velocity.x:.2f}, {player.velocity.y:.2f}, {player.velocity.z:.2f} (SPD: {player.velocity.magnitude():.2f}>{Vector3(player.velocity.x, 0 ,player.velocity.z).magnitude():.2f})"
            )
            camera.pen.goto(-390, 220)
            camera.pen.write(
                f"MEM: {process.memory_info().rss / (1024 * 1024):.2f}MB/{process.memory_info().vms / (1024 * 1024):.2f}MB"
            )
            camera.pen.goto(-390, 200)
            camera.pen.write("OTHER DATA:")
            camera.pen.goto(-370, 180)
            camera.pen.write(
                f"SEED: {chunk_manager.noise_gen.seed} (DRAMATICNESS: {chunk_manager.dramaticness})"
            )
            camera.pen.goto(-370, 160)
            waiya = player.raycast_to_block(camera.direction)
            if waiya:
                camera.pen.write(
                    f"WAIYA: {chunk_manager.texture_manager.get_block_name(chunk_manager.get_block_world(*waiya))} ({chunk_manager.get_block_world(*waiya)})"
                )
            else:
                camera.pen.write(f"WAIYA: None")
            camera.pen.goto(-370, 140)
            camera.pen.write(
                f"Holding: {chunk_manager.texture_manager.get_block_name(hold_id)} ({hold_id})"
            )
            camera.pen.goto(-390, 120)
            camera.pen.write(
                "WASD to move, Mouse to look, Q to break, E to place, scroll to change item selection"
            )

            time_to_sleep = max(0, (1 / target) - (time.time() - start_time))
            time.sleep(time_to_sleep)

            camera.pen.goto(-390, 280)
            camera.pen.write(
                f"FPS: {sum(fps_data)/len(fps_data):.1f} | {1/(time.time() - start_time):.3f} ({uncapped:.3f} | {target})"
            )

            turtle.update()

            return result

        return wrapped

    return decorator


if __name__ == "__main__":
    mouse_initialized = False
    mouse_locked = True
    cyote = 0
    mouse_listener = None
    current_mouse_x = 400
    current_mouse_y = 300
    last_mouse_x = 400
    last_mouse_y = 300
    last_break_time = 0
    last_place_time = 0

    turtle.bgcolor("#85C8FF")

    chunk_manager = ChunkManager(
        TextureManager(), render_distance=2, dramaticness=2.3, seed=28168261
    )
    player = Player(
        Vector3(-155, 50, 105),
        chunk_manager,
        height=1.2,
        width=0.5,
        move_coefficient=0.000001,
        gravity=38.0,
        max_fall_speed=float("inf"),
    )
    camera = Camera(player, Vector3(0, 0, 1), screen_width=800, screen_height=600)
    scroll = 3
    hold_id = 3

    mouse_init()

    def on_scroll(event):
        global scroll, hold_id
        if isinstance(event, mouse.WheelEvent):
            scroll += int(event.delta)
            # Filter out air block (ID 0) from selectable blocks
            selectable_blocks = [
                k
                for k in chunk_manager.texture_manager.block_definitions.keys()
                if k != 0
            ]
            hold_id = selectable_blocks[scroll % len(selectable_blocks)]

    mouse.hook(on_scroll)

    @normalize_framerate(20)
    def main(deltatime):
        global mouse_locked, last_break_time, last_place_time

        movement, angle, break_block, place_block = handle_movement(
            speed=60.0,
            sensitivity=0.005,
            player=player,
            jump_strength=10.0,
            cyote_time=3,
        )

        current_time = time.time()

        if break_block and current_time - last_break_time > 0.2:
            player.break_block_at_crosshair(camera.direction)
            last_break_time = current_time

        if place_block and current_time - last_place_time > 0.2:
            player.place_block_at_crosshair(
                camera.direction,
                block_type=hold_id,
            )
            last_place_time = current_time

        player.move_relative(
            movement, camera.direction, deltatime, horizontal_only=True
        )

        if angle.x != 0 or angle.y != 0:
            camera.rotate(angle.x, angle.y)

        player.update_physics(deltatime)
        camera.update_position()

        camera.render_world(
            fov=103,
            screen_width=800,
            screen_height=600,
            max_render_distance=10,
            min_brightness=0.2,
            render_distance=2,
        )

        chunk_manager.unload_distant_chunks(camera.pos)

        try:
            if keyboard.is_pressed("p"):
                mouse_locked = not mouse_locked
                time.sleep(0.2)

            if keyboard.is_pressed("esc"):
                mouse_locked = False
        except:
            pass

    import psutil, os

    process = psutil.Process(os.getpid())

    while True:
        main()
