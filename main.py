import turtle
from math import *
from typing import Union, List, Dict, Tuple, Optional
import time
import random
from dataclasses import dataclass

from vectors import *

import mouse
import keyboard

Number = Union[int, float]


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
            0: (44, 62, 80),
            1: (139, 69, 19),
            2: (34, 139, 34),
            3: (101, 67, 33),
            4: (160, 160, 160),
            5: (218, 165, 32),
            6: (143, 188, 143),
            7: (139, 69, 19),
            8: (70, 130, 180),
            9: (255, 99, 71),
            10: (112, 128, 144),
            11: (101, 67, 33),
            12: (34, 139, 34),
        }
        self.block_definitions = {
            0: Block(0, 0, 0, 0),
            1: Block(1, 1, 1, 1),
            2: Block(2, 2, 3, 1),
            3: Block(3, 4, 4, 4),
            4: Block(4, 5, 5, 5),
            5: Block(5, 6, 6, 6),
            6: Block(6, 7, 7, 7),
            7: Block(7, 8, 8, 8),
            8: Block(8, 9, 9, 9),
            9: Block(9, 10, 10, 10),
            10: Block(10, 11, 11, 11),
            11: Block(11, 12, 12, 12),
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


class NoiseGenerator:
    """Simple noise generator for terrain"""

    def __init__(self, seed: int = 12345):
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

        tree_height = 4 + int(
            abs(self.noise_gen.noise2d(world_x * 0.3, world_z * 0.3)) * 3
        )
        trunk_height = max(4, tree_height - 2)

        local_x = world_x - chunk.chunk_x * Chunk.CHUNK_SIZE
        local_z = world_z - chunk.chunk_z * Chunk.CHUNK_SIZE

        if not (0 <= local_x < Chunk.CHUNK_SIZE and 0 <= local_z < Chunk.CHUNK_SIZE):
            return

        for y in range(
            surface_y + 1, min(surface_y + trunk_height + 1, Chunk.CHUNK_HEIGHT)
        ):
            chunk.blocks[(local_x, y, local_z)] = 10

        # mm vegetation (leaf)

        leaves_start = surface_y + max(2, trunk_height - 1)  # first layer of leaves
        leaves_end = min(
            surface_y + trunk_height + 1, Chunk.CHUNK_HEIGHT - 1
        )  # last layer of leaves

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
                                chunk.blocks[(leaf_x, y, leaf_z)] = 11


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

        self._visible_faces_cache = {}

        for (x, y, z), block_type in self.blocks.items():
            if block_type == 0:
                continue

            visible_faces = []
            for face_name, (dx, dy, dz) in face_offsets.items():
                adjacent_block = self._get_adjacent_block_safe(x + dx, y + dy, z + dz)
                if adjacent_block == 0:
                    visible_faces.append(face_name)

            if visible_faces:
                self._visible_faces_cache[(x, y, z)] = visible_faces

        self._face_culling_computed = True
        self._cache_dirty = False

    def _get_adjacent_block_safe(self, x: int, y: int, z: int) -> int:
        if 0 <= y < self.CHUNK_HEIGHT:
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

    def generate_terrain(self):
        surface_heights = {}

        for x in range(self.CHUNK_SIZE):
            for z in range(self.CHUNK_SIZE):
                world_x, world_z = self.get_world_pos(x, z)
                height = self.get_height(world_x, world_z)
                surface_heights[(x, z)] = height

                for y in range(min(height + 1, self.CHUNK_HEIGHT)):
                    if y <= height:
                        block_type = self.get_block_type(world_x, y, world_z, height)
                        if block_type != 0:
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
        height_noise = self.noise_gen.noise2d(world_x * 0.05, world_z * 0.05)
        base_height = 16 + int(height_noise * 4)
        return max(1, min(base_height, self.CHUNK_HEIGHT - 1))

    def get_block_type(
        self, world_x: int, y: int, world_z: int, surface_height: int
    ) -> int:
        if y == surface_height and surface_height > 15:
            return 2
        elif y >= surface_height - 2 and surface_height > 15:
            return 1
        elif y < 2:
            return 8
        elif y < 5:
            return 3
        else:
            return 3

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
    def __init__(self, texture_manager: TextureManager, render_distance: int = 2):
        self.chunks = {}
        self.texture_manager = texture_manager
        self.noise_gen = NoiseGenerator()
        self.render_distance = render_distance
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
            self.chunks[key] = Chunk(
                chunk_x, chunk_z, self.noise_gen, self.texture_manager, self
            )
            self._generating_chunks.remove(key)
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

            self.block_cache[cache_key] = block
            if len(self.block_cache) > 2000:
                keys_to_remove = list(self.block_cache.keys())[
                    : len(self.block_cache) // 2
                ]
                for key in keys_to_remove:
                    del self.block_cache[key]

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

        for chunk in visible_chunks:
            chunk.mark_dirty()
            chunk._precompute_face_culling()

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

        for chunk_x, chunk_z in self.chunks.keys():
            dx = chunk_x - camera_chunk_x
            dz = chunk_z - camera_chunk_z
            distance_sq = dx * dx + dz * dz

            if distance_sq > unload_distance_sq:
                chunks_to_unload.append((chunk_x, chunk_z))

        for chunk_key in chunks_to_unload:
            del self.chunks[chunk_key]

        if chunks_to_unload:
            self._last_camera_chunk = None
            self._cached_visible_chunks = []

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


@dataclass
class RenderFace:
    """Face to be rendered with depth sorting and lighting"""

    distance: float
    projected: List[Tuple[float, float]]
    color: Tuple[int, int, int]
    ambient_occlusion: float = 1.0


class Camera:
    def __init__(self, pos, direction=None, screen_width=800, screen_height=600):
        self.pos = pos
        self.direction = direction.normalize() if direction else Vector3(0, 0, 1)

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
        self.texture_manager = TextureManager()
        self.chunk_manager = ChunkManager(self.texture_manager, render_distance=2)

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

        self.cube_vertices_pattern = [
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            Vector3(1, 1, 0),
            Vector3(0, 1, 0),
            Vector3(0, 0, 1),
            Vector3(1, 0, 1),
            Vector3(1, 1, 1),
            Vector3(0, 1, 1),
        ]

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

    def calculate_relative(self, pos, horizontal_only=False):
        """Convert relative movement vector to absolute world coordinates"""
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

    def move_axis(self, world):
        self.pos += world
        self._clear_position_caches()

    def move(self, steps, horizontal_only=False):
        if horizontal_only:
            forward = Vector3(self.direction.x, 0, self.direction.z)
            if forward.magnitude() > 0:
                forward = forward.normalize()
            else:
                forward = Vector3(0, 0, 1)
        else:
            forward = self.direction
        self.pos += forward * steps
        self._clear_position_caches()
        return self.pos

    def strafe(self, steps):
        up = Vector3(0, 1, 0)
        right = self.direction.cross(up)
        if right.magnitude() > 0:
            right = right.normalize()
        else:
            right = Vector3(1, 0, 0)
        self.pos += right * steps
        self._clear_position_caches()
        return self.pos

    def move_relative(self, pos, horizontal_only=False):
        absolute_movement = self.calculate_relative(pos, horizontal_only)
        self.pos += absolute_movement
        self._clear_position_caches()

    def rotate(self, pitch_delta, yaw_delta):
        up = Vector3(0, 1, 0)
        if yaw_delta != 0:
            cos_yaw, sin_yaw = cos(yaw_delta), sin(yaw_delta)
            new_x = self.direction.x * cos_yaw - self.direction.z * sin_yaw
            new_z = self.direction.x * sin_yaw + self.direction.z * cos_yaw
            self.direction = Vector3(new_x, self.direction.y, new_z).normalize()
        if pitch_delta != 0:
            cos_pitch, sin_pitch = cos(pitch_delta), sin(pitch_delta)
            new_direction = self.direction * cos_pitch + up * sin_pitch
            self.direction = new_direction.normalize()
        self._clear_position_caches()

    def _clear_position_caches(self):
        self._projection_cache.clear()
        self._frustum_cache.clear()

    def project_point(self, point, fov=90, screen_width=800, screen_height=600):
        cache_key = (point.x, point.y, point.z, self.pos.x, self.pos.y, self.pos.z)
        if cache_key in self._projection_cache:
            return self._projection_cache[cache_key]

        rel = point - self.pos
        forward = self.direction
        up = Vector3(0, 1, 0)
        right = forward.cross(up)

        if right.magnitude() == 0:
            return None
        right = right.normalize()
        actual_up = right.cross(forward)

        x = rel.dot(right)
        y = rel.dot(actual_up)
        z = rel.dot(forward)

        if z <= 0.1:
            return None

        fov_rad = radians(fov)
        f = 1.0 / tan(fov_rad * 0.5)

        screen_x = (x * f / z) * (screen_height / screen_width)
        screen_y = y * f / z

        screen_x = screen_x * (screen_width * 0.5) + screen_width * 0.5
        screen_y = screen_y * (screen_height * 0.5) + screen_height * 0.5

        result = (screen_x - screen_width * 0.5, screen_y - screen_height * 0.5)
        self._projection_cache[cache_key] = result
        return result

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

    def is_face_in_frustum(
        self, face_vertices: List[Vector3], fov=90, screen_width=800, screen_height=600
    ) -> bool:
        forward = self.direction
        right = forward.cross(Vector3(0, 1, 0)).normalize()
        up = right.cross(forward)

        fov_rad = radians(fov)
        half_fov = fov_rad * 0.5
        aspect_ratio = screen_width / screen_height

        vertices_in_front = 0
        vertices_in_horizontal = 0
        vertices_in_vertical = 0

        for vertex in face_vertices:
            rel = vertex - self.pos
            forward_dot = rel.dot(forward)
            if forward_dot > 0.1:
                vertices_in_front += 1
                right_dot = rel.dot(right)
                horizontal_bound = forward_dot * tan(half_fov) * aspect_ratio
                if abs(right_dot) <= horizontal_bound:
                    vertices_in_horizontal += 1
                up_dot = rel.dot(up)
                vertical_bound = forward_dot * tan(half_fov)
                if abs(up_dot) <= vertical_bound:
                    vertices_in_vertical += 1

        return (
            vertices_in_front > 0
            and vertices_in_horizontal > 0
            and vertices_in_vertical > 0
        )

    def render_world(
        self,
        fov=90,
        screen_width=800,
        screen_height=600,
        max_render_distance=25,
        min_brightness=0.2,
        render_distance=2,
    ):
        self.pen.clear()

        if self.chunk_manager.render_distance != render_distance:
            self.chunk_manager.render_distance = render_distance

        visible_chunks = self.chunk_manager.get_visible_chunks(self.pos)
        render_faces = []

        for chunk in visible_chunks:
            for (local_x, local_y, local_z), block_type in chunk.blocks.items():
                world_x = chunk.chunk_x * Chunk.CHUNK_SIZE + local_x
                world_y = local_y
                world_z = chunk.chunk_z * Chunk.CHUNK_SIZE + local_z

                block_center = Vector3(world_x + 0.5, world_y + 0.5, world_z + 0.5)
                distance_to_camera = (block_center - self.pos).magnitude()

                if distance_to_camera > max_render_distance:
                    continue

                visible_faces = chunk.get_visible_faces(local_x, local_y, local_z)
                if not visible_faces:
                    continue

                vertices = [
                    Vector3(world_x, world_y, world_z) + vertex_offset
                    for vertex_offset in self.cube_vertices_pattern
                ]

                for face_name in visible_faces:
                    indices, offset = self.faces_data[face_name]
                    face_center = block_center + offset
                    face_normal = self.face_normals[face_name]

                    if not self.is_face_visible(face_center, face_normal):
                        continue

                    face_vertices = [vertices[i] for i in indices]

                    if not self.is_face_in_frustum(
                        face_vertices, fov, screen_width, screen_height
                    ):
                        continue

                    projected = [
                        self.project_point(v, fov, screen_width, screen_height)
                        for v in face_vertices
                    ]

                    if all(p is not None for p in projected):
                        texture_id = self.get_block_texture(block_type, face_name)
                        ambient_occlusion = self.face_ao[face_name]
                        color = self.texture_manager.get_color(
                            texture_id,
                            distance_to_camera,
                            ambient_occlusion,
                            max_render_distance,
                            min_brightness,
                        )

                        render_faces.append(
                            RenderFace(
                                distance_to_camera, projected, color, ambient_occlusion
                            )
                        )

        render_faces.sort(key=lambda x: x.distance, reverse=True)

        # self.pen.color("#00FFFF")w

        for face in render_faces:
            if len(face.projected) >= 3:
                self.pen.color(face.color)
                self.pen.goto(*face.projected[0])
                # self.pen.pendown()
                self.pen.begin_fill()
                for point in face.projected[1:]:
                    self.pen.goto(*point)
                self.pen.goto(*face.projected[0])
                # self.pen.penup()
                self.pen.end_fill()


def mouse_init():
    """Initiates the mouse"""
    global mouse_delta_x, mouse_delta_y, last_mouse_x, last_mouse_y, mouse_initialized
    mouse_delta_x = 0
    mouse_delta_y = 0
    last_mouse_x = 0
    last_mouse_y = 0
    mouse_initialized = False


def handle_movement(speed=0.3, sensitivity=0.01):
    """Handles camera movement with optimizations"""
    global mouse_initialized

    camera_movement = zero3()
    camera_angle = zero2()

    try:
        keys = {
            "w": keyboard.is_pressed("w"),
            "s": keyboard.is_pressed("s"),
            "a": keyboard.is_pressed("a"),
            "d": keyboard.is_pressed("d"),
            "ctrl": keyboard.is_pressed("ctrl"),
            "space": keyboard.is_pressed("space"),
        }

        if keys["w"]:
            camera_movement.z = speed
        if keys["s"]:
            camera_movement.z = -speed
        if keys["a"]:
            camera_movement.x = -speed
        if keys["d"]:
            camera_movement.x = speed
        if keys["ctrl"]:
            camera_movement.y = -speed
        if keys["space"]:
            camera_movement.y = speed
    except Exception:
        pass

    if mouse_locked:
        try:
            x, y = mouse.get_position()
            if mouse_initialized:
                camera_angle.x -= (y - 300) * sensitivity
                camera_angle.y += (x - 400) * sensitivity
                mouse.move(400, 300)
            else:
                mouse_initialized = True
                mouse.move(400, 300)
        except Exception:
            pass

    return camera_movement, camera_angle


# Performance monitoring
total = []
frame = 0
sample_freq = 10
fps_data = []
frames = []


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
    """Decorator function to normalize a function's runtime"""

    def decorator(func):
        def wrapped(*args, **kwargs):
            global total, frame, fps_data, frames
            frame += 1
            start_time = time.time()
            result = func(*args, **kwargs)

            camera.pen.goto(-390, 280)
            camera.pen.color("#000000")
            camera.pen.write(f"FPS: {round(1/(time.time() - start_time), 3)}")
            camera.pen.goto(-390, 260)
            camera.pen.write(
                f"POS: {round(camera.pos.x,2), round(camera.pos.y,2), round(camera.pos.z,2)}"
            )

            time_to_sleep = max(0, (1 / target) - (time.time() - start_time))
            time.sleep(time_to_sleep)

            camera.pen.goto(-390, 240)
            camera.pen.write(f"Capped FPS: {round(1/(time.time() - start_time), 3)}")

            turtle.update()

            return result

        return wrapped

    return decorator


if __name__ == "__main__":
    random.seed(28168261)
    mouse_initialized = False
    mouse_locked = True

    turtle.bgcolor("#85C8FF")
    camera = Camera(
        Vector3(0, 20, -5), Vector3(0, 0, 1), screen_width=800, screen_height=600
    )

    mouse_init()
    momentum = Vector3(0, 0, 0)

    @normalize_framerate(40)
    def main():
        global mouse_locked, momentum

        movement, angle = handle_movement(speed=0.4, sensitivity=0.005)

        momentum += camera.calculate_relative(movement, horizontal_only=True)
        momentum *= 0.8

        camera.move_axis(momentum)

        if angle.x != 0 or angle.y != 0:
            camera.rotate(angle.x, angle.y)

        camera.render_world(
            fov=103,
            screen_width=800,
            screen_height=600,
            max_render_distance=20,
            min_brightness=0.2,
            render_distance=2,
        )

        try:
            if keyboard.is_pressed("p"):
                mouse_locked = not mouse_locked
                time.sleep(0.2)

            if keyboard.is_pressed("esc"):
                mouse_locked = False
        except:
            pass

    while True:
        main()

    # import cProfile

    # main()

    # cProfile.run(
    #     """camera.render_world(
    #         fov=103,
    #         screen_width=800,
    #         screen_height=600,
    #         max_render_distance=20,
    #         min_brightness=0.2,
    #         render_distance=2,
    #     )""",
    #     "mz.txt",
    # )
