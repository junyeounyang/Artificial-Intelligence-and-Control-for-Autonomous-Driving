"""대학로 도로망(net.xml)에서 한 번 읽어 두는 정보.

환경(daehakro_env.py)이 매 스텝 SUMO 에 묻지 않아도 되게, 변하지 않는 값을 미리 표로 만든다.

    car_ok[차로]        승용차가 다닐 수 있는 차로인가 (보도·버스 전용 차로는 False)
    change_ok[차로]     그 차로에서 (왼쪽, 오른쪽)으로 차로변경을 해도 되는가 (실선 구간은 False)
    links[(차로, 다음 도로)]  그 차로에서 다음 도로로 이어지는 연결들 [(도착 차로, 교차로 안에서 가는 거리)]
    reach_table(경로)   경로의 각 차로에서 차로변경 없이 경로를 따라 갈 수 있는 거리
    lanes_ahead(...)    어떤 차로의 앞으로 이어지는 차로들과 거기까지의 거리
    lanes_behind(...)   어떤 차로로 들어오는 뒤쪽 차로들과 거기까지의 거리
    speed[차로]         제한속도 (m/s)
    length[차로]        길이 (m)
    path_ahead(...)     지금 차로에서 경로를 따라 앞으로 지나갈 차로들 (교차로 안 차로 포함)
    cross[횡단보도]     모양, 길이, 방향, 보행 신호, 양 끝의 보행자 대기 공간
    next_crossing(...)  앞으로 처음 만나는 횡단보도와 거기까지의 거리

횡단보도는 SUMO 에서 교차로 안에 놓이므로, 차는 교차로 안 차로(":" 로 시작)를 지나면서
횡단보도를 밟는다. 그래서 '교차로 안 차로가 횡단보도와 만나는 위치'를 미리 계산해 둔다.
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

if "SUMO_HOME" in os.environ:
    _tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if _tools not in sys.path:
        sys.path.append(_tools)
import sumolib  # noqa: E402


def _seg_cross(p, p2, q, q2):
    """선분 p-p2 와 q-q2 가 만나면 (p 쪽 비율, q 쪽 비율), 아니면 None."""
    rx, ry = p2[0] - p[0], p2[1] - p[1]
    sx, sy = q2[0] - q[0], q2[1] - q[1]
    den = rx * sy - ry * sx
    if abs(den) < 1e-12:
        return None
    t = ((q[0] - p[0]) * sy - (q[1] - p[1]) * sx) / den
    u = ((q[0] - p[0]) * ry - (q[1] - p[1]) * rx) / den
    return (t, u) if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 else None


def _poly_cross(a, b):
    """꺾은선 a 가 꺾은선 b 와 처음 만나는 곳 → (a 위 거리, b 위 거리, 두 선이 이루는 각의 sin). 없으면 None."""
    off_a = 0.0
    for i in range(len(a) - 1):
        len_a = math.dist(a[i], a[i + 1])
        best, off_b = None, 0.0
        for j in range(len(b) - 1):
            len_b = math.dist(b[j], b[j + 1])
            r = _seg_cross(a[i], a[i + 1], b[j], b[j + 1])
            if r and (best is None or r[0] < best[0]):
                ax, ay = a[i + 1][0] - a[i][0], a[i + 1][1] - a[i][1]
                bx, by = b[j + 1][0] - b[j][0], b[j + 1][1] - b[j][1]
                sin = abs(ax * by - ay * bx) / max(len_a * len_b, 1e-9)
                best = (r[0], off_b + r[1] * len_b, sin)
            off_b += len_b
        if best:
            return off_a + best[0] * len_a, best[1], best[2]
        off_a += len_a
    return None


def _xy(shape):
    return [(p[0], p[1]) for p in shape]


class RoadMap:
    def __init__(self, net_file):
        net = sumolib.net.readNet(net_file, withInternal=True)
        self.car_ok, self.speed, self.length = {}, {}, {}
        self.lanes_of = {}            # edge → 차로 id 목록 (index 순)
        self.change_ok = {}           # 차로 → (왼쪽으로 차로변경 가능, 오른쪽으로 가능)
        self.links = {}               # (차로, 다음 edge) → [(도착 차로, 교차로 안 차로들의 길이 합)]  승용차가 갈 수 있는 것만
        self._out = {}                # 차로 → [(도착 차로, [그 사이 교차로 안 차로들])]  모든 차종
        self._in = {}                 # 차로 → [(출발 차로, [그 사이 교차로 안 차로들])]  모든 차종
        self._span = {}               # lanes_ahead / lanes_behind 결과를 기억해 둔다
        self._conn = {}               # (차로, 다음 edge) → 그 연결로 처음 들어서는 차로 (교차로 안 차로 또는 도착 차로)
        self._next = {}               # 교차로 안 차로 → 다음 차로
        shapes = {}

        for edge in net.getEdges(withInternal=True):
            func = edge.getFunction()
            if func in ("crossing", "walkingarea"):
                continue
            ids = []
            for lane in edge.getLanes():
                lid = lane.getID()
                ids.append(lid)
                self.car_ok[lid] = bool(lane.allows("passenger"))
                self.speed[lid] = float(lane.getSpeed())
                self.length[lid] = float(lane.getLength())
                if func == "internal":
                    shapes[lid] = _xy(lane.getShape())
            self.lanes_of[edge.getID()] = ids

        for edge in net.getEdges(withInternal=True):
            func = edge.getFunction()
            if func in ("crossing", "walkingarea"):
                continue
            for lane in edge.getLanes():
                for c in lane.getOutgoing():
                    to_lane = c.getToLane()
                    if not self.car_ok.get(to_lane.getID(), False):
                        continue
                    first = c.getViaLaneID() or to_lane.getID()
                    if func == "internal":
                        self._next.setdefault(lane.getID(), first)
                    else:
                        self._conn.setdefault((lane.getID(), c.getTo().getID()), first)

        # 차로와 차로를 잇는 연결. 교차로 안에서 지나는 차로들도 함께 적어 둔다.
        inner_next = {}               # 교차로 안 차로 → 그 다음 차로 (모든 차종)
        for edge in net.getEdges(withInternal=True):
            if edge.getFunction() == "internal":
                for lane in edge.getLanes():
                    for c in lane.getOutgoing():
                        inner_next.setdefault(lane.getID(), c.getViaLaneID() or c.getToLane().getID())
        for edge in net.getEdges(withInternal=False):
            if edge.getFunction() in ("crossing", "walkingarea"):
                continue
            for lane in edge.getLanes():
                lid = lane.getID()
                for c in lane.getOutgoing():
                    to_lane = c.getToLane().getID()
                    if to_lane not in self.length:
                        continue
                    chain, via = [], c.getViaLaneID()
                    while via and via.startswith(":") and via in self.length and len(chain) < 8:
                        chain.append(via)
                        via = inner_next.get(via)
                    self._out.setdefault(lid, []).append((to_lane, chain))
                    self._in.setdefault(to_lane, []).append((lid, chain))
                    if self.car_ok.get(lid, False) and self.car_ok.get(to_lane, False):
                        inside = sum(self.length[v] for v in chain)
                        self.links.setdefault((lid, c.getTo().getID()), []).append((to_lane, inside))

        root = ET.parse(net_file).getroot()
        # 차로변경 허용: net.xml 의 changeLeft / changeRight 는 '그 방향으로 차로변경이 허용된 차종' 목록이다.
        # 적혀 있지 않으면 모두 허용. 승용차가 목록에 없으면 그 방향은 실선으로 본다.
        for e in root.iter("edge"):
            if e.get("function") in ("crossing", "walkingarea", "internal"):
                continue
            for ln in e.iter("lane"):
                self.change_ok[ln.get("id")] = tuple(
                    a is None or a == "all" or "passenger" in a.split()
                    for a in (ln.get("changeLeft"), ln.get("changeRight")))

        # ── 횡단보도 ──
        self.cross = {}
        for e in root.iter("edge"):
            if e.get("function") != "crossing":
                continue
            ln = e.find("lane")
            pts = [tuple(map(float, p.split(",")[:2])) for p in ln.get("shape").split()]
            cid = e.get("id")
            dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
            self.cross[cid] = {
                "node": cid[1:cid.rindex("_c")],
                "shape": pts,
                "length": float(ln.get("length")),
                "half": float(ln.get("width", "4.0")) / 2,
                "angle": math.degrees(math.atan2(dx, dy)) % 360.0,   # SUMO 각도: 북 = 0, 시계 방향
                "tl": None, "idx": None,        # 보행 신호 (없으면 신호 없는 횡단보도)
                "wa": set(),                    # 양 끝의 보행자 대기 공간(walkingarea)
            }
        for c in root.iter("connection"):
            f, t = c.get("from", ""), c.get("to", "")
            if t in self.cross:
                self.cross[t]["wa"].add(f)
                if c.get("tl") is not None:
                    self.cross[t]["tl"], self.cross[t]["idx"] = c.get("tl"), int(c.get("linkIndex"))
            elif f in self.cross:
                self.cross[f]["wa"].add(t)

        # ── 교차로 안 차로가 횡단보도와 만나는 위치 ──
        #    hits[차로] = [(차로 위 위치 m, 횡단보도 id, 횡단보도 위 위치 m, 차로 방향으로 잰 횡단보도 반폭 m)]
        self.hits = {}
        for lid, shp in shapes.items():
            node = lid[1:].rsplit("_", 2)[0]
            found = []
            for cid, c in self.cross.items():
                if c["node"] != node:
                    continue
                r = _poly_cross(shp, c["shape"])
                if r:
                    half = min(c["half"] / max(r[2], 0.34), 3.0 * c["half"])   # 비스듬히 지나면 더 길게
                    found.append((r[0], cid, r[1], half))
            if found:
                self.hits[lid] = sorted(found)

    # ------------------------------------------------------------------
    def _first_lane_to(self, lane_id, next_edge):
        """lane_id 에서 next_edge 로 가는 연결로 처음 들어서는 차로.

        그 차로에 연결이 없으면(경로상 틀린 차로) 같은 도로에서 가장 가까운, 연결 있는 차로의 것을 쓴다.
        """
        first = self._conn.get((lane_id, next_edge))
        if first is not None:
            return first
        edge, idx = lane_id.rsplit("_", 1)
        best = None
        for other in self.lanes_of.get(edge, ()):
            f = self._conn.get((other, next_edge))
            if f is not None:
                gap = abs(int(other.rsplit("_", 1)[1]) - int(idx))
                if best is None or gap < best[0]:
                    best = (gap, f)
        return best[1] if best else None

    def reach_table(self, route):
        """경로(route: edge 목록)의 각 차로 → 그 차로의 시작점에서 차로변경 없이 경로를 따라 갈 수 있는 거리 (m).

        교차로 안에서 가는 거리도 포함한다. 경로 끝까지 이어지는 차로는 inf.
        승용차가 못 가는 차로(보도·버스 전용)는 표에 없다.
        """
        table = {}
        last = len(route) - 1
        for i in range(last, -1, -1):
            for lane in self.lanes_of.get(route[i], ()):
                if not self.car_ok.get(lane, False):
                    continue
                if i == last:
                    table[lane] = math.inf
                    continue
                onward = [inside + table[to] for to, inside in self.links.get((lane, route[i + 1]), ())
                          if to in table]
                table[lane] = self.length[lane] + (max(onward) if onward else 0.0)
        return table

    def lanes_ahead(self, lane_id, dist):
        """lane_id 의 끝에서 앞으로 dist 안에 이어지는 모든 차로 (갈라지는 길 모두, 교차로 안 차로 포함).

        Returns: {차로: lane_id 의 시작점에서 그 차로의 시작점까지 거리 m}
        """
        key = ("a", lane_id, dist)
        if key not in self._span:
            out, todo = {}, [(lane_id, self.length[lane_id])]      # (차로, lane_id 시작점에서 그 차로 끝까지의 거리)
            limit = self.length[lane_id] + dist
            while todo:
                cur, end = todo.pop()
                for to_lane, chain in self._out.get(cur, ()):
                    off = end
                    for via in chain:
                        out[via] = min(out.get(via, math.inf), off)
                        off += self.length[via]
                    if off < limit and off < out.get(to_lane, math.inf):
                        out[to_lane] = off
                        todo.append((to_lane, off + self.length[to_lane]))
            self._span[key] = out
        return self._span[key]

    def lanes_behind(self, lane_id, dist):
        """lane_id 의 시작점으로 들어오는, 뒤로 dist 안의 모든 차로 (합쳐지는 길 모두, 교차로 안 차로 포함).

        Returns: {차로: 그 차로의 시작점에서 lane_id 의 시작점까지 거리 m}
        """
        key = ("b", lane_id, dist)
        if key not in self._span:
            out, todo = {}, [(lane_id, 0.0)]                       # (차로, 그 차로 시작점에서 lane_id 시작점까지의 거리)
            while todo:
                cur, back = todo.pop()
                for from_lane, chain in self._in.get(cur, ()):
                    d = back
                    for via in reversed(chain):
                        d += self.length[via]
                        out[via] = min(out.get(via, math.inf), d)
                    whole = d + self.length[from_lane]
                    if d < dist and whole < out.get(from_lane, math.inf):
                        out[from_lane] = whole
                        todo.append((from_lane, whole))
            self._span[key] = out
        return self._span[key]

    def path_ahead(self, lane_id, lane_pos, route, max_dist):
        """지금 차로에서 경로(route: edge 목록)를 따라 max_dist 앞까지 지나갈 차로들.

        Returns: [(차로 id, 차 앞머리에서 그 차로 시작점까지의 거리)]. 지금 차로는 거리가 음수(-lane_pos).
        """
        if lane_id not in self.length:
            return []
        out = [(lane_id, -lane_pos)]
        dist = self.length[lane_id] - lane_pos
        cur = lane_id
        while dist < max_dist:
            if cur.startswith(":"):
                nxt = self._next.get(cur)
            else:
                edge = cur.rsplit("_", 1)[0]
                try:
                    i = route.index(edge)
                except ValueError:
                    break
                if i + 1 >= len(route):
                    break
                nxt = self._first_lane_to(cur, route[i + 1])
            if nxt is None or nxt not in self.length:
                break
            out.append((nxt, dist))
            dist += self.length[nxt]
            cur = nxt
        return out

    def next_crossing(self, path):
        """path(path_ahead 결과)에서 처음 만나는 횡단보도.

        Returns: (횡단보도 id, 횡단보도 가장자리까지 거리 m, 내 경로가 횡단보도를 지나는 위치 m) 또는 None.
                 이미 횡단보도 위에 올라서 있으면 거리는 0.
        """
        for lane_id, start in path:
            for pos, cid, s, half in self.hits.get(lane_id, ()):
                if start + pos + half < 0.0:
                    continue                      # 이미 지나간 횡단보도
                return cid, max(start + pos - half, 0.0), s
        return None
