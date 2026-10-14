"""Bounded sequential native decoder, retrieving only requested 5fps frames."""
import bisect
from emotioncore.x06b.common import MEDIA

class Video:
    def __init__(self, timeline):
        self.entries=timeline['entries']; self.ends=[e['end_s'] for e in self.entries]
        self.reader=None; self.path=None; self.native_index=-1; self.last=None
        self.retrieved_frames=0; self.grabbed_frames=0

    def frame(self, when):
        import cv2
        i=min(bisect.bisect_right(self.ends,when),len(self.entries)-1)
        entry=self.entries[i]; gap=entry['kind']=='gap'
        if gap: entry=self.entries[i-1]
        path=str(MEDIA/entry['video']['path'])
        if self.path!=path:
            self.close(); self.reader=cv2.VideoCapture(path); self.path=path
            if not self.reader.isOpened(): raise RuntimeError('STOP media decode failed: '+path)
            self.native_index=-1; self.last=None
            self.fps=float(self.reader.get(cv2.CAP_PROP_FPS)); self.count=int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.fps<=0 or self.count<=0: raise RuntimeError('STOP invalid source video timing')
        target=self.count-1 if gap else min(self.count-1, max(0,round((when-entry['start_s'])*self.fps)))
        if target<self.native_index: raise RuntimeError('STOP nonmonotonic decoder request')
        if target!=self.native_index:
            while self.native_index<target:
                if not self.reader.grab(): raise RuntimeError('STOP native decode failed')
                self.native_index+=1; self.grabbed_frames+=1
            ok,self.last=self.reader.retrieve(); self.retrieved_frames+=1
            if not ok: raise RuntimeError('STOP frame retrieval failed')
        return self.last

    def close(self):
        if self.reader is not None:self.reader.release()
        self.reader=None; self.path=None; self.last=None
