from math import hypot


def _point_tuple(point):
    x, y, frame_index = point[:3]
    label = int(point[3]) if len(point) >= 4 else 1
    return (float(x), float(y), int(frame_index), label)


def copy_frame_points(points_by_frame):
    return [[_point_tuple(point) for point in frame_points] for frame_points in (points_by_frame or [])]


def empty_frame_points(frame_count):
    return [[] for _ in range(max(0, int(frame_count)))]


def ensure_frame_points(points_by_frame, frame_count):
    copied = copy_frame_points(points_by_frame)
    target_count = max(0, int(frame_count))
    while len(copied) < target_count:
        copied.append([])
    return copied[:target_count]


def _clamp_frame_index(frame_index, frame_count):
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    return min(max(int(frame_index), 0), frame_count - 1)


def append_refinement_point(points_by_frame, frame_index, x, y, label):
    updated = copy_frame_points(points_by_frame)
    frame_index = _clamp_frame_index(frame_index, len(updated))
    updated[frame_index].append((float(x), float(y), frame_index, int(label)))
    return updated


def remove_nearest_refinement_point(points_by_frame, frame_index, x, y, max_distance=12.0):
    updated = copy_frame_points(points_by_frame)
    frame_index = _clamp_frame_index(frame_index, len(updated))
    frame_points = updated[frame_index]
    if not frame_points:
        return updated, False

    distances = [hypot(float(point[0]) - float(x), float(point[1]) - float(y)) for point in frame_points]
    nearest_index = min(range(len(distances)), key=distances.__getitem__)
    if distances[nearest_index] > float(max_distance):
        return updated, False

    del frame_points[nearest_index]
    return updated, True


def pop_refinement_point(points_by_frame, frame_index):
    updated = copy_frame_points(points_by_frame)
    frame_index = _clamp_frame_index(frame_index, len(updated))
    if not updated[frame_index]:
        return updated, False
    updated[frame_index].pop()
    return updated, True


def clear_frame_refinement_points(points_by_frame, frame_index):
    updated = copy_frame_points(points_by_frame)
    frame_index = _clamp_frame_index(frame_index, len(updated))
    updated[frame_index] = []
    return updated


def clear_all_refinement_points(points_by_frame):
    return empty_frame_points(len(points_by_frame or []))


def merge_frame_point_lists(base_points, refinement_points):
    base = copy_frame_points(base_points)
    refinements = copy_frame_points(refinement_points)
    frame_count = max(len(base), len(refinements))
    base = ensure_frame_points(base, frame_count)
    refinements = ensure_frame_points(refinements, frame_count)
    return [base_frame + refinement_frame for base_frame, refinement_frame in zip(base, refinements)]


def count_frame_points(points_by_frame):
    return sum(len(frame_points) for frame_points in (points_by_frame or []))
