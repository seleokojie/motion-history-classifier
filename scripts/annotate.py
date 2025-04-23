#!/usr/bin/env python3
import os
import argparse
import cv2
import numpy as np
from src.classifier import MHIClassifier
from src.mhi import compute_binary_sequence, compute_mhi
from src.features import extract_hu_features
from src.utils import annotate_frame


def main(args):
    # Load model
    clf = MHIClassifier.load(args.model_path)

    cap = cv2.VideoCapture(args.input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    frame_buffer = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_buffer.append(gray)
        if len(frame_buffer) > args.tau:
            frame_buffer.pop(0)
        if len(frame_buffer) > 1:
            B = compute_binary_sequence(frame_buffer, threshold=args.threshold)
            M = compute_mhi(B, tau=args.tau)
            feats = extract_hu_features(M).reshape(1,-1)
            pred = clf.predict(feats)[0]
        else:
            pred = 'none'
        disp = annotate_frame(frame, pred)
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(args.output_video, fourcc, cap.get(cv2.CAP_PROP_FPS), (w,h))
        out.write(disp)
    cap.release()
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_video', required=True)
    parser.add_argument('--output_video', required=True)
    parser.add_argument('--threshold', type=int, default=30)
    parser.add_argument('--tau', type=int, default=30)
    args = parser.parse_args()
    main(args)