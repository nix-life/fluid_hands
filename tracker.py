import argparse
import math
import numpy as np
import os
import socket
import struct
import sys
import time
import urllib.request

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


def log_line(s):
    print(s, flush=True)


def download_model(path):
    log_line(f"downloading {MODEL_URL} -> {path}")
    urllib.request.urlretrieve(MODEL_URL, path)
    log_line(f"saved {os.path.getsize(path)} bytes")


def open_camera(idx):
    import cv2
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {idx} "
                           "(check Windows Settings > Privacy > Camera)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def make_landmarker(model_path):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (HandLandmarker,
                                               HandLandmarkerOptions,
                                               RunningMode)
    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    return HandLandmarker.create_from_options(opts), mp


def fake_frame(t, w=320, h=240):
    y, x = np.mgrid[0:h, 0:w]
    ph = t * 2.0
    img = np.stack([
        (x / w * 180 + 40 * math.sin(ph)) % 255,
        (y / h * 160 + 60 * math.cos(ph * 0.7)) % 255,
        np.full((h, w), 90 + 50 * math.sin(ph * 0.5)),
    ], axis=-1)
    cx, cy = w * (0.5 + 0.3 * math.sin(ph)), h * (0.5 + 0.2 * math.cos(ph * 1.3))
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    img[..., 1] = np.clip(img[..., 1] + np.exp(-d2 / (2 * 45.0 ** 2)) * 160, 0, 255)
    return img.astype(np.uint8)


class _Lm:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


THUMB = (0.0, (-0.646, -0.763), (0.042, 0.067, 0.085, 0.103), (0.7, 0.8, 0.5))
FINGERS = [(-0.040, (0.0, -1.0), (0.070, 0.100, 0.125, 0.145), (0.9, 1.4, 0.9)),
           (-0.013, (0.0, -1.0), (0.075, 0.110, 0.135, 0.160), (0.9, 1.4, 0.9)),
           (0.013, (0.0, -1.0), (0.070, 0.100, 0.125, 0.140), (0.9, 1.4, 0.9)),
           (0.040, (0.0, -1.0), (0.060, 0.085, 0.105, 0.120), (0.9, 1.4, 0.9))]


def rodrigues(d, a, ang):
    c, s = math.cos(ang), math.sin(ang)
    dot = a[0] * d[0] + a[1] * d[1] + a[2] * d[2]
    cr = (a[1] * d[2] - a[2] * d[1], a[2] * d[0] - a[0] * d[2],
          a[0] * d[1] - a[1] * d[0])
    return (d[0] * c + cr[0] * s + a[0] * dot * (1 - c),
            d[1] * c + cr[1] * s + a[1] * dot * (1 - c),
            d[2] * c + cr[2] * s + a[2] * dot * (1 - c))


def bend_chain(ox, v, dists, curl, coeffs):
    n = math.hypot(v[0], v[1])
    vx, vy = v[0] / n, v[1] / n
    a = (-vy, vx, 0.0)
    pts = [(ox + vx * dists[0], vy * dists[0], 0.0)]
    ang = 0.0
    for j in range(3):
        ang += curl * coeffs[j]
        d = rodrigues((vx, vy, 0.0), a, ang)
        seg = dists[j + 1] - dists[j]
        p = pts[-1]
        pts.append((p[0] + d[0] * seg, p[1] + d[1] * seg, p[2] + d[2] * seg))
    return pts


def hand_model(curl):
    pts = [(0.0, 0.0, 0.0)]
    pts += bend_chain(THUMB[0], THUMB[1], THUMB[2], curl, THUMB[3])
    for f in FINGERS:
        pts += bend_chain(f[0], f[1], f[2], curl, f[3])
    return pts


def rot_yaw_roll(yaw, roll):
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    return ((cr * cy_, -sr, cr * sy_),
            (sr * cy_, cr, sr * sy_),
            (-sy_, 0.0, cy_))


def fake_hand3d(cx, cy, yaw, roll, k=2.0, curl=0.0):
    R = rot_yaw_roll(yaw, roll)
    lms, wlms = [], []
    for (x, y, z) in hand_model(curl):
        X = R[0][0] * x + R[0][1] * y + R[0][2] * z
        Y = R[1][0] * x + R[1][1] * y + R[1][2] * z
        Z = R[2][0] * x + R[2][1] * y + R[2][2] * z
        wlms.append(_Lm(X, Y, Z))
        lms.append(_Lm(cx + X * k, cy + Y * k, Z * k))
    return lms, wlms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--stream-width", type=int, default=320)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "hand_landmarker.task"))
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--curlphase", type=float, default=0.0)
    args = ap.parse_args()

    if args.log:
        lf = open(args.log, "w", buffering=1)
        sys.stdout = lf
        sys.stderr = lf

    if not args.fake and not os.path.exists(args.model):
        download_model(args.model)
    if args.download_only:
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = ("127.0.0.1", args.port)

    landmarker = None
    mp = None
    cap = None
    if not args.fake:
        landmarker, mp = make_landmarker(args.model)
        cap = open_camera(args.cam)
        import cv2
    else:
        cv2 = None

    frame_id = 0
    hands_seen = 0
    t0 = time.monotonic()
    last_stat = t0
    stat_frames = 0

    try:
        while True:
            now = time.monotonic()
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_aspect = rgb.shape[1] / rgb.shape[0]
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect_for_video(img, int(now * 1000))
                lms_list = res.hand_landmarks
                wlms_list = res.hand_world_landmarks
                handed = res.handedness
                src = rgb
            else:
                rgb = fake_frame(now - t0)
                cam_aspect = rgb.shape[1] / rgb.shape[0]
                tel = now - t0
                mt = int(tel // 7.5) * 6.0 + min(tel % 7.5, 6.0)
                ca = mt * 2 * math.pi / 6.0
                l0, w0 = fake_hand3d(0.45 + 0.15 * math.cos(ca), 0.5 + 0.15 * math.sin(ca),
                                     math.radians(40) * math.sin(mt * 2 * math.pi / 5.0),
                                     math.radians(30) * math.sin(mt * 2 * math.pi / 7.0),
                                     curl=0.5 + 0.5 * math.sin((mt + args.curlphase) * 2 * math.pi / 4.0))
                l1, w1 = fake_hand3d(0.5 + 0.2 * math.sin(tel * 2 * math.pi / 3.0), 0.5, 0.0, 0.0)
                lms_list = [l0, l1]
                wlms_list = [w0, w1]
                handed = [[type("C", (), {"category_name": n, "score": 0.99})()]
                          for n in ("Right", "Left")]
                src = rgb

            nh = min(len(lms_list), 2)
            if nh:
                hands_seen += 1
            pkt = struct.pack("<4sIfB", b"HND2", frame_id, cam_aspect, nh)
            for i in range(nh):
                name = handed[i][0].category_name if handed else "Right"
                score = handed[i][0].score if handed else 0.0
                pkt += struct.pack("<Bf", 0 if name == "Left" else 1, score)
                for lm in lms_list[i][:21]:
                    pkt += struct.pack("<fff", lm.x, lm.y, getattr(lm, "z", 0.0))
                wl = wlms_list[i] if i < len(wlms_list) else lms_list[i]
                for lm in wl[:21]:
                    pkt += struct.pack("<fff", lm.x, lm.y, getattr(lm, "z", 0.0))
            sock.sendto(pkt, dest)

            sw = args.stream_width
            sh = max(1, int(round(sw / cam_aspect)))
            if cv2 is not None:
                small = cv2.resize(src, (sw, sh))
            else:
                ys = np.linspace(0, src.shape[0] - 1, sh).astype(int)
                xs = np.linspace(0, src.shape[1] - 1, sw).astype(int)
                small = src[np.ix_(ys, xs)]
            data = small.tobytes()
            rows_per = max(1, min(60, 65500 // (sw * 3)))
            for rs in range(0, sh, rows_per):
                rc = min(rows_per, sh - rs)
                chunk = data[rs * sw * 3:(rs + rc) * sw * 3]
                sock.sendto(struct.pack("<4sIHHHH", b"IMG1", frame_id, sw, sh, rs, rc) + chunk, dest)

            frame_id += 1
            stat_frames += 1
            if now - last_stat >= 2.0:
                log_line(f"tracker: {stat_frames / (now - last_stat):.1f} fps  hands={nh}  "
                         f"frames={frame_id}")
                last_stat = now
                stat_frames = 0
            if args.frames and frame_id >= args.frames:
                break
            if args.fake:
                time.sleep(max(0.0, 1.0 / 30.0 - (time.monotonic() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        el = time.monotonic() - t0
        log_line(f"summary: {frame_id} frames in {el:.1f}s  avg fps={frame_id / max(el, 1e-6):.1f}  "
                 f"frames-with-hands={hands_seen}")
        if cap is not None:
            cap.release()
        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    main()
