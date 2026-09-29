from emotioncore.x03.media import FRAME_COUNT, equal_positions, select_nearest


def test_eight_equal_positions_keep_endpoints():
    positions = equal_positions(19)
    assert len(positions) == FRAME_COUNT
    assert positions[0] == 0 and positions[-1] == 18
    assert positions == sorted(positions)


def test_failed_face_frames_use_nearest_accepted_crop():
    assert select_nearest([0, 9, 18], 19) == [0, 0, 9, 9, 9, 9, 18, 18]
