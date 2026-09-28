from emotioncore.x02.face import add_detections, iou, rank_tracks, selected_indices


def test_iou_tracking_and_area_presence_choice():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0
    tracks = []
    assert add_detections(tracks, 0, [(0, 0, 20, 20), (100, 0, 10, 10)]) == [0, 1]
    assert add_detections(tracks, 1, [(2, 0, 20, 20), (102, 0, 10, 10)]) == [0, 1]
    assert add_detections(tracks, 2, [(4, 0, 20, 20)]) == [0]
    ordered = rank_tracks(tracks, 3)
    assert ordered[0]["id"] == 0
    assert ordered[0]["presence_fraction"] == 1
    assert ordered[0]["target_score"] > ordered[1]["target_score"]


def test_selected_frames_are_even_and_bounded():
    assert selected_indices([1, 3, 5]) == [1, 3, 5]
    chosen = selected_indices(list(range(40)))
    assert len(chosen) == 8
    assert chosen[0] == 0 and chosen[-1] == 39
