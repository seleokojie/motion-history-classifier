#!/usr/bin/env python3
import sys
import os
import json
import argparse
import cv2
import traceback

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classifier import MHIClassifier
from src.mhi import compute_binary_sequence, compute_mhi
from src.features import extract_hu_features
from src.utils import annotate_frame

def main(args):
    try:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        thr_file = os.path.join(base, 'outputs', 'thresholds_per_action.json')
        if not os.path.exists(thr_file):
            print(f"[ERROR] Cannot find thresholds file: {thr_file}")
            sys.exit(1)

        with open(thr_file) as f:
            per_act = json.load(f)

        # If user did not override threshold, pick the one for this action
        if args.threshold is None:
            args.threshold = per_act.get(args.action)
            print(f"[INFO] Using default threshold={args.threshold} for action '{args.action}'")

        # Build input/output paths
        inp = os.path.join(base, 'data', 'videos',
                           f"person{args.person}_{args.action}_{args.condition}_uncomp.avi")
        out_dir = os.path.join(base, 'outputs', 'annotated_videos')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                                f"person{args.person}_{args.action}_{args.condition}.avi")

        print(f"[INFO] Annotating with τ={args.tau}, thr={args.threshold}")
        print(f"       Model:  {args.model_path}")
        print(f"       Input:  {inp}")
        print(f"       Output: {out_path}\n")

        # Sanity checks
        if not os.path.exists(args.model_path):
            print(f"[ERROR] Model not found: {args.model_path}")
            sys.exit(1)
        if not os.path.exists(inp):
            print(f"[ERROR] Video not found: {inp}")
            sys.exit(1)

        # Load model
        clf = MHIClassifier.load(args.model_path)

        cap = cv2.VideoCapture(inp)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {inp}")
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None
        buf = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            buf.append(gray)
            if len(buf) > args.tau:
                buf.pop(0)

            if len(buf) > 1:
                B = compute_binary_sequence(buf, threshold=args.threshold)
                M = compute_mhi(B, tau=args.tau)
                feats = extract_hu_features(M).reshape(1, -1)
                pred = clf.predict(feats)[0]
            else:
                pred = 'none'

            disp = annotate_frame(frame, pred)

            if out is None:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                if not out.isOpened():
                    print(f"[ERROR] Could not open writer for: {out_path}")
                    sys.exit(1)

            out.write(disp)
            if idx % 100 == 0:
                print(f"[INFO] Processed {idx} frames...")

        cap.release()
        if out:
            out.release()

        print(f"\n[SUCCESS] Done! Saved to: {out_path}")
    except Exception as e:
        print(f"\n[EXCEPTION] {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help="Path to the trained model (e.g. outputs/model_best.joblib)",
                        default='outputs/model_best.joblib')
    parser.add_argument('--person', type=int, required=True,
                        help="Person ID (e.g., 22)")
    parser.add_argument('--action', required=True,
                        choices=['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping'],
                        help="Action label")
    parser.add_argument('--condition', required=True,
                        choices=['d1', 'd2', 'd3', 'd4'],
                        help="Condition (e.g., d1–d4)")
    parser.add_argument('--threshold', type=int, default=10,
                        help="Motion threshold; if omitted, uses per-action default")
    parser.add_argument('--tau', type=int, default=260,
                        help="History length τ (default: 260)")
    args = parser.parse_args()
    main(args)
