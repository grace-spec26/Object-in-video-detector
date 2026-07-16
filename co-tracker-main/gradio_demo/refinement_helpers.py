from math import hypot


def _point_tuple(point):
    x, y, frame_index = point[:3]
    label = int(point[3]) if len(point) >= 4 else 1
    return (float(x), float(y), int(frame_index), label)


def copy_frame_points(points_by_frame):
    return [[_point_tuple(point) for point in frame_points] for frame_points in (points_by_frame or [])]


def copy_frame_values(values_by_frame):
    return [list(frame_values) for frame_values in (values_by_frame or [])]


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


def remove_nearest_frame_point(points_by_frame, values_by_frame, frame_index, x, y, max_distance=12.0):
    updated_points = copy_frame_points(points_by_frame)
    updated_values = copy_frame_values(values_by_frame)
    frame_index = _clamp_frame_index(frame_index, len(updated_points))
    while len(updated_values) < len(updated_points):
        updated_values.append([])
    updated_values = updated_values[:len(updated_points)]

    frame_points = updated_points[frame_index]
    if not frame_points:
        return updated_points, updated_values, False

    distances = [hypot(float(point[0]) - float(x), float(point[1]) - float(y)) for point in frame_points]
    nearest_index = min(range(len(distances)), key=distances.__getitem__)
    if distances[nearest_index] > float(max_distance):
        return updated_points, updated_values, False

    del frame_points[nearest_index]
    if nearest_index < len(updated_values[frame_index]):
        del updated_values[frame_index][nearest_index]
    return updated_points, updated_values, True


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


def flatten_prompt_sources(base_points, refinement_points=None):
    base = copy_frame_points(base_points)
    refinements = copy_frame_points(refinement_points)
    frame_count = max(len(base), len(refinements))
    base = ensure_frame_points(base, frame_count)
    refinements = ensure_frame_points(refinements, frame_count)

    sources = []
    for frame_index in range(frame_count):
        sources.extend(("base", frame_index, point_index) for point_index in range(len(base[frame_index])))
        sources.extend(
            ("refinement", frame_index, point_index)
            for point_index in range(len(refinements[frame_index]))
        )
    return sources


def remove_prompt_by_source(base_points, base_colors, refinement_points, source):
    base = copy_frame_points(base_points)
    colors = copy_frame_values(base_colors)
    refinements = copy_frame_points(refinement_points)
    frame_count = max(len(base), len(colors), len(refinements))
    base = ensure_frame_points(base, frame_count)
    refinements = ensure_frame_points(refinements, frame_count)
    while len(colors) < frame_count:
        colors.append([])
    colors = colors[:frame_count]

    kind, frame_index, point_index = source
    frame_index = int(frame_index)
    point_index = int(point_index)
    if frame_index < 0 or frame_index >= frame_count or point_index < 0:
        return base, colors, refinements, False

    if str(kind) == "base":
        if point_index >= len(base[frame_index]):
            return base, colors, refinements, False
        del base[frame_index][point_index]
        if point_index < len(colors[frame_index]):
            del colors[frame_index][point_index]
        return base, colors, refinements, True

    if str(kind) == "refinement":
        if point_index >= len(refinements[frame_index]):
            return base, colors, refinements, False
        del refinements[frame_index][point_index]
        return base, colors, refinements, True

    return base, colors, refinements, False


def drop_prompt_source(prompt_sources, removed_index):
    sources = [tuple(source) for source in (prompt_sources or [])]
    removed_index = int(removed_index)
    if removed_index < 0 or removed_index >= len(sources):
        return sources

    removed_kind, removed_frame, removed_point_index = sources[removed_index]
    removed_kind = str(removed_kind)
    removed_frame = int(removed_frame)
    removed_point_index = int(removed_point_index)
    updated = []
    for index, source in enumerate(sources):
        if index == removed_index:
            continue

        kind, frame_index, point_index = source
        kind = str(kind)
        frame_index = int(frame_index)
        point_index = int(point_index)
        if (
            kind == removed_kind
            and frame_index == removed_frame
            and point_index > removed_point_index
        ):
            point_index -= 1
        updated.append((kind, frame_index, point_index))
    return updated


def pending_refinement_points(refinement_points, tracked_prompt_sources=None, frame_count=None):
    refinements = copy_frame_points(refinement_points)
    target_count = len(refinements) if frame_count is None else max(0, int(frame_count))
    refinements = ensure_frame_points(refinements, target_count)

    tracked_refinements = set()
    for source in tracked_prompt_sources or []:
        if len(source) < 3:
            continue
        kind, frame_index, point_index = source[:3]
        if str(kind) == "refinement":
            tracked_refinements.add((int(frame_index), int(point_index)))

    return [
        [
            point
            for point_index, point in enumerate(frame_points)
            if (frame_index, point_index) not in tracked_refinements
        ]
        for frame_index, frame_points in enumerate(refinements)
    ]


def count_frame_points(points_by_frame):
    return sum(len(frame_points) for frame_points in (points_by_frame or []))
