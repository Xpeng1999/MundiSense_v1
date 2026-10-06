from emotioncore.x04.prepare import media_duration


def test_auxiliary_data_stream_cannot_extend_visual_sampling_window():
    info = {"format": {"duration": "4844.769"}, "streams": [
        {"codec_type": "video", "duration": "3.003003"},
        {"codec_type": "audio", "duration": "3.024"},
        {"codec_type": "data", "duration": "4844.769"}]}
    assert media_duration(info, "video") == (3.003003, "video_stream")
    assert media_duration(info, "audio") == (3.024, "audio_stream")


def test_header_without_stream_duration_uses_container_fallback():
    assert media_duration({"format": {"duration": "2.5"}, "streams": [{"codec_type": "audio"}]}, "audio") == (2.5, "container_fallback")
