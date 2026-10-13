"""Fixed YuNet detector with bounded parallel low-resolution inputs."""
import threading, time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import nullcontext
from emotioncore.x06.face import detector

_LOCAL=threading.local()

def detect(frame, scale):
    import cv2
    if not hasattr(_LOCAL,'detector'): _LOCAL.detector=detector()
    h,w=frame.shape[:2]; dw,dh=max(1,round(w*scale)),max(1,round(h*scale))
    image=frame if scale==1 else cv2.resize(frame,(dw,dh),interpolation=cv2.INTER_LINEAR)
    _LOCAL.detector.setInputSize((dw,dh)); tic=time.monotonic()
    _,faces=_LOCAL.detector.detect(image)
    boxes=[] if faces is None else [[float(r[0])*w/dw,float(r[1])*h/dh,float(r[2])*w/dw,float(r[3])*h/dh] for r in faces]
    return boxes,time.monotonic()-tic


def pipeline(frames, scale, workers=8, capacity=16, executor=None):
    """Yield in input order; at most capacity full-resolution frames in flight."""
    if workers<1 or capacity<workers: raise ValueError('invalid queue bounds')
    pending=deque()
    with (nullcontext(executor) if executor is not None else ThreadPoolExecutor(max_workers=workers)) as pool:
        for item,frame in frames:
            pending.append((item,frame,pool.submit(detect,frame,scale)))
            if len(pending)>=capacity:
                meta,pixels,future=pending.popleft();boxes,seconds=future.result()
                yield meta,pixels,boxes,seconds
                del pixels,future
        while pending:
            meta,pixels,future=pending.popleft();boxes,seconds=future.result()
            yield meta,pixels,boxes,seconds
            del pixels,future
