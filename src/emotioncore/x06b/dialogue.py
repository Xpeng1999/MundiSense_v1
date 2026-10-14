"""Dialogue-duration metadata only; other speakers' media are never loaded."""
from collections import defaultdict
import random
from emotioncore.x06b.common import FMH,TEST


def turn(row):return int(row['utt_id'].rsplit('_',1)[1])


def build(records,seed=20261013):
    groups=defaultdict(list)
    for r in records:groups[r['group'],r['dialogue']].append(r)
    output={}
    for speaker in sorted(FMH|TEST):
        selected=sorted([r for r in records if r['speaker_id']==speaker and (speaker in FMH or r['official_split']=='test')],key=lambda r:(r['group'],r['dialogue'],turn(r)))
        rng=random.Random(seed+int(speaker));entries=[];cursor=0.
        for i,row in enumerate(selected):
            duration=float(row['mp4']['duration_s'])
            entries.append({'kind':'utterance','synthetic':False,'speaker_id':speaker,'utt_id':row['utt_id'],
                'start_s':cursor,'end_s':cursor+duration,'duration_s':duration,'audio_duration_s':row['wav']['duration_s'],
                'label':row['labels']['M'],'video':row['mp4'],'audio':row['wav']});cursor+=duration
            if i+1==len(selected):continue
            nxt=selected[i+1];same=(row['group'],row['dialogue'])==(nxt['group'],nxt['dialogue'])
            others=[];pauses=[]
            if same:
                others=sorted([x for x in groups[row['group'],row['dialogue']] if turn(row)<turn(x)<turn(nxt)],key=turn)
                if any(x['speaker_id']==speaker for x in others):raise ValueError('STOP target sequence omitted a turn')
                ids=[speaker]+[x['speaker_id'] for x in others]+[speaker]
                changes=sum(a!=b for a,b in zip(ids,ids[1:])) or 1
                pauses=[rng.uniform(.3,1.) for _ in range(changes)]
                gap=sum(x['mp4']['duration_s'] for x in others)+sum(pauses)
            else:gap=3.
            entries.append({'kind':'gap','synthetic':True,'speaker_id':speaker,'start_s':cursor,'end_s':cursor+gap,
                'noise_reference_utt_id':row['utt_id'],'noise_seed':rng.getrandbits(32),'hold_previous_frame':True,
                'other_turns':[{'utt_id':x['utt_id'],'speaker_id':x['speaker_id'],'duration_s':x['mp4']['duration_s']} for x in others],
                'turn_pauses_s':pauses,'scene_change':not same})
            cursor+=gap
        output[speaker]={'speaker_id':speaker,'cohort':'fmh' if speaker in FMH else 'et_test','duration_s':cursor,'utterances':len(selected),'entries':entries}
    return output
