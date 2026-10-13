"""Incremental X02 association and original-resolution crop operations."""
import math,time
from pathlib import Path
from emotioncore.x02 import face as original


class Tracker:
    def __init__(self,directory):
        self.directory=Path(directory);self.tracks=[];self.paths={};self.frames=0
        self.requested=0;self.multiple=0;self.seconds=0.;self.closed=False
        self.resident_frame_count=0

    def add(self,frame,index,boxes):
        import cv2,numpy as np
        if self.closed:raise RuntimeError('tracker already released')
        tic=time.monotonic();self.resident_frame_count=1
        try:
            mapping=original.add_detections(self.tracks,index,boxes)
            self.frames+=1;self.requested=max(self.requested,index+1);self.multiple+=len(boxes)>=2
            if not boxes:return
            # Preserve the X02 JPEG78 input round trip at full resolution, then
            # crop immediately. No encoded full frame survives this call.
            ok,encoded=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,78])
            if not ok:raise RuntimeError('full-resolution JPEG conversion failed')
            pixels=cv2.imdecode(encoded,cv2.IMREAD_COLOR);height,width=pixels.shape[:2]
            scale=math.sqrt(50176/(width*height))
            whole=cv2.resize(pixels,(max(1,round(width*scale)),max(1,round(height*scale))))
            whole_path=self.directory/f'whole_{index:06d}.jpg';original._write_jpg(whole_path,whole,cv2)
            for box,track in zip(boxes,mapping):
                x,y,w,h=box
                left,top=max(0,math.floor(x-original.CROP_EXPANSION*w)),max(0,math.floor(y-original.CROP_EXPANSION*h))
                right=min(width,math.ceil(x+(1+original.CROP_EXPANSION)*w));bottom=min(height,math.ceil(y+(1+original.CROP_EXPANSION)*h))
                if right<=left or bottom<=top:continue
                crop=cv2.resize(pixels[top:bottom,left:right],(448,448))
                crop_path=self.directory/f'track_{track:04d}/crop_{index:06d}.jpg'
                original._write_jpg(crop_path,crop,cv2)
                self.paths[track,index]=(str(crop_path),str(whole_path))
        finally:
            self.resident_frame_count=0;self.seconds+=time.monotonic()-tic

    def finish(self):
        if self.closed:raise RuntimeError('tracker already released')
        ranked=original.rank_tracks(self.tracks,self.frames)
        result={'status':'no_face','requested_frames':self.requested,'decoded_frames':self.frames,
                'track_scores':[t['target_score'] for t in ranked],'tracks':len(ranked),
                'multi_face_frames':self.multiple,'crop_paths':[],'whole_paths':[],'face_seconds':self.seconds}
        if ranked:
            target=ranked[0]
            for index in original.selected_indices(sorted(target['boxes'])):
                if (target['id'],index) in self.paths:
                    crop,whole=self.paths[target['id'],index]
                    result['crop_paths'].append(crop);result['whole_paths'].append(whole)
            result.update(status='ok' if result['crop_paths'] else 'failed',target_track=target['id'],
                          target_score=target['target_score'],target_presence_fraction=target['presence_fraction'])
        self.tracks.clear();self.paths.clear();self.closed=True
        return result
