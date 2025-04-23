import os
import re
import cv2
import numpy as np

class DataLoader:
    """Loads video segments per split using sequences.txt."""
    SPLITS = {
        'train': {'person11','person12','person13','person14','person15','person16','person17','person18'},
        'val':   {'person19','person20','person21','person23','person24','person25','person01','person04'},
        'test':  {'person22','person02','person03','person05','person06','person07','person08','person09','person10'}
    }

    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.video_dir = os.path.join(data_dir, 'videos')
        self.seq_file = os.path.join(data_dir, 'sequences.txt')
        self.split = split
        self.entries = self._parse_sequences()
        print(f"[DEBUG] DataLoader for split='{self.split}' found {len(self.entries)} entries")

    def _parse_sequences(self):
        entries = []
        with open(self.seq_file, 'r') as f:
            for line in f:
                line = line.strip()
                # skip blanks or headers
                if not line or "frames" not in line:
                    continue

                # split on the word "frames"
                video_key, ranges_part = re.split(r"\s+frames\s+", line)

                # parse each start-end segment
                spans = []
                for token in ranges_part.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    start_str, end_str = token.split("-")
                    spans.append((int(start_str), int(end_str)))

                # split video_key into person and action
                parts = video_key.split("_")
                person = parts[0]
                action = parts[1]
                # filter by split membership
                if person not in self.SPLITS[self.split]:
                    continue

                # reconstruct video filename
                vid_name = f"{video_key}_uncomp.avi"
                vid_path = os.path.join(self.video_dir, vid_name)

                entries.append({
                    "video": vid_path,
                    "action": action,
                    "ranges": spans
                })
        return entries

    def load_segments(self):
        """
        Generator yielding (segment_frames: list[np.ndarray], label: str, id: str)
        """
        for idx, e in enumerate(self.entries):
            if not os.path.exists(e['video']):
                print(f"[DEBUG] Missing file: {e['video']}")
                continue
            # otherwise open it
            # look for cached .npy
            cache = e['video'].replace('.avi', '.npy')
            if os.path.exists(cache):
                vid = np.load(cache)
            else:
                cap = cv2.VideoCapture(e['video'])
                buf = []
                while True:
                    ret, frm = cap.read()
                    if not ret: break
                    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                    buf.append(gray)
                cap.release()
                vid = np.stack(buf, axis=0)
                np.save(cache, vid)

            # slice out each segment range
            frames = []
            for (start, end) in e['ranges']:
                frames.extend(vid[start - 1:end])
            yield frames, e['action'], f"seg{idx}"