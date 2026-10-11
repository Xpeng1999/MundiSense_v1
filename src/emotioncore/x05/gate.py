"""Fixed, pure availability routing; quality never changes routing in X05."""


def route(has_face, face_count, face_quality, has_speech):
    if face_count > 1:
        return 'reject', 'multi_face'
    if not has_face and not has_speech:
        return 'reject', 'no_modalities'
    if not has_face:
        return 'H_A', 'no_face'
    if not has_speech:
        return 'H_V', 'no_speech'
    return 'full', 'both_available'
