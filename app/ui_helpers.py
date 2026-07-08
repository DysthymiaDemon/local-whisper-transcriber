from __future__ import annotations


def meter_segment_count(width: int, bar_width: int = 7, gap: int = 4, min_segments: int = 18) -> int:
    available_width = max(0, int(width))
    segment_width = max(1, int(bar_width))
    segment_gap = max(0, int(gap))
    minimum = max(1, int(min_segments))
    if available_width <= 0:
        return minimum
    fitted = max(1, (available_width + segment_gap) // (segment_width + segment_gap))
    return max(minimum, fitted)


def meter_bar_geometry(width: int, bar_width: int = 7, gap: int = 4, min_segments: int = 18) -> tuple[int, int, int]:
    return meter_segment_count(width, bar_width, gap, min_segments), int(bar_width), int(gap)
