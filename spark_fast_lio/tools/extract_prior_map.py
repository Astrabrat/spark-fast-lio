#!/usr/bin/env python3
"""Extract a prior map .pcd directly from a rosbag2 .db3, with no ROS installed.

Pulls the last PointCloud2 published on a topic (typically the accumulated map
from a previous mapping run) and writes it as a binary PCD with XYZI fields —
the format the relocalization node's `relocalization.map_file` expects.

This exists so a bag can be turned into its own prior map, which makes the
relocalization test self-contained: map the run once, then replay the same bag
and require the node to find itself in that map with no operator pose.

    python3 extract_prior_map.py <bag.db3> --topic /husky/global_map -o prior_map.pcd
    python3 extract_prior_map.py <bag.db3> --list

With --accumulate it also cuts the *query* side of the same test: N consecutive
clouds merged into one window, taken from anywhere in the recording. Both halves
of a T2/T3 fixture set therefore come out of one bag and one tool:

    # the map, from the end of the run
    ... --topic /husky/global_map -o prior_map.pcd
    # a 5-scan query window from 60% of the way in, in the odom frame
    ... --topic /husky/fast_lio/cloud_registered --accumulate 5 --at 0.6 -o query.pcd

Merging is a plain concatenation, which is only valid because cloud_registered
is already in a common frame. Do not point --accumulate at a sensor-frame topic
such as ouster/points; the result would be N scans piled up at the origin.
"""

import argparse
import math
import sqlite3
import struct
import sys

_DTYPE = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
_SIZE = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}


class Reader:
    """Minimal little-endian CDR reader. Alignment is relative to the payload,
    i.e. to the byte after the 4-byte encapsulation header."""

    def __init__(self, buf):
        self.b = buf
        self.p = 0
        self.base = 4

    def _align(self, n):
        self.p = (self.p + n - 1) // n * n

    def u8(self):
        v = self.b[self.base + self.p]
        self.p += 1
        return v

    def u32(self):
        self._align(4)
        (v,) = struct.unpack_from("<I", self.b, self.base + self.p)
        self.p += 4
        return v

    def i32(self):
        self._align(4)
        (v,) = struct.unpack_from("<i", self.b, self.base + self.p)
        self.p += 4
        return v

    def string(self):
        n = self.u32()
        v = self.b[self.base + self.p : self.base + self.p + n - 1].decode("utf8", "replace")
        self.p += n
        return v

    def blob(self, n):
        v = self.b[self.base + self.p : self.base + self.p + n]
        self.p += n
        return v


def parse_pointcloud2(buf):
    r = Reader(buf)
    r.i32()  # stamp.sec
    r.u32()  # stamp.nanosec
    frame_id = r.string()
    height = r.u32()
    width = r.u32()

    fields = []
    for _ in range(r.u32()):
        name = r.string()
        offset = r.u32()
        datatype = r.u8()
        count = r.u32()
        fields.append((name, offset, datatype, count))

    r.u8()  # is_bigendian
    point_step = r.u32()
    r.u32()  # row_step
    data = r.blob(r.u32())
    return frame_id, height, width, fields, point_step, data


def to_xyzi(width, height, fields, point_step, data):
    """Returns a list of (x, y, z, intensity), skipping non-finite points."""
    by_name = {f[0]: f for f in fields}
    for req in ("x", "y", "z"):
        if req not in by_name:
            raise SystemExit(f"cloud has no '{req}' field (fields: {list(by_name)})")

    def unpack(name, default=0.0):
        if name not in by_name:
            return lambda _buf, _o: default
        _, off, dt, _ = by_name[name]
        if dt not in _DTYPE:
            return lambda _buf, _o: default
        fmt = "<" + _DTYPE[dt].replace("f4", "f").replace("f8", "d").replace(
            "u1", "B").replace("i1", "b").replace("u2", "H").replace("i2", "h").replace(
            "u4", "I").replace("i4", "i")
        return lambda buf, o: struct.unpack_from(fmt, buf, o + off)[0]

    gx, gy, gz = unpack("x"), unpack("y"), unpack("z")
    gi = unpack("intensity", 0.0)

    out = []
    n = width * height
    for i in range(n):
        o = i * point_step
        x, y, z = gx(data, o), gy(data, o), gz(data, o)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        out.append((x, y, z, float(gi(data, o))))
    return out


def write_pcd(path, pts):
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(pts)}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p in pts:
            f.write(struct.pack("<4f", *p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db3")
    ap.add_argument("--topic", default="/husky/global_map")
    ap.add_argument("-o", "--out", default="prior_map.pcd")
    ap.add_argument("--list", action="store_true", help="list PointCloud2 topics and exit")
    ap.add_argument("--accumulate", type=int, default=1, metavar="N",
                    help="merge N consecutive messages (only for already-registered clouds)")
    ap.add_argument("--at", type=float, default=1.0, metavar="F",
                    help="where in the recording to take them, 0.0=start 1.0=end (default 1.0)")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db3}?mode=ro", uri=True)
    cur = con.cursor()

    if args.list:
        cur.execute(
            "select t.name, count(m.id) from topics t left join messages m on m.topic_id=t.id "
            "where t.type='sensor_msgs/msg/PointCloud2' group by t.id order by t.name"
        )
        for name, n in cur.fetchall():
            print(f"{n:8d}  {name}")
        return 0

    cur.execute("select id from topics where name=?", (args.topic,))
    row = cur.fetchone()
    if not row:
        print(f"topic not found: {args.topic}  (try --list)", file=sys.stderr)
        return 1
    topic_id = row[0]
    cur.execute("select count(*) from messages where topic_id=?", (topic_id,))
    total = cur.fetchone()[0]
    if not total:
        print(f"no messages on {args.topic}", file=sys.stderr)
        return 1

    # --at 1.0 has to keep meaning "the last N", so the window is clamped to the end
    # rather than centred: a prior map is only complete in the final message.
    n = max(1, min(args.accumulate, total))
    start = int(round(args.at * (total - n)))
    start = max(0, min(start, total - n))
    cur.execute(
        "select data from messages where topic_id=? order by timestamp limit ? offset ?",
        (topic_id, n, start),
    )
    rows = cur.fetchall()

    pts, frame_id = [], None
    for r in rows:
        fid, h, w, fields, step, data = parse_pointcloud2(bytes(r[0]))
        if frame_id is None:
            frame_id = fid
        elif fid != frame_id:
            # Concatenating across frames would silently produce a cloud that is
            # geometrically meaningless, and it would still look like a valid PCD.
            print(f"frame_id changed mid-window: {frame_id} -> {fid}", file=sys.stderr)
            return 1
        pts.extend(to_xyzi(w, h, fields, step, data))
    write_pcd(args.out, pts)
    if n > 1:
        print(f"merged    : {n} messages, {start}..{start + n - 1} of {total}")

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    print(f"topic     : {args.topic}")
    print(f"frame_id  : {frame_id}   <- must match relocalization.map_frame")
    print(f"points    : {len(pts)} (of {w*h} declared)")
    print(
        "extent    : x[%.1f, %.1f]  y[%.1f, %.1f]  z[%.1f, %.1f]   -> %.1f x %.1f x %.1f m"
        % (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs),
           max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    )
    diag = math.dist((min(xs), min(ys)), (max(xs), max(ys)))
    print(f"diagonal  : {diag:.1f} m")
    print(f"wrote     : {args.out}")
    print()
    print("Suggested starting parameters for a map this size:")
    print(f"  relocalization.submap.radius            {max(15.0, diag):.0f}")
    print(f"  relocalization.global.voxel_size        {max(0.1, diag/60.0):.2f}")
    print(f"  relocalization.global.map_voxel_size    {max(0.1, diag/60.0):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
