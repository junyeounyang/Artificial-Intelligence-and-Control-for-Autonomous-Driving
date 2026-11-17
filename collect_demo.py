"""BC 용 시범 수집 — 시범 운전자가 AV 를 운전하고, 관측과 '정책이 냈어야 할 행동'만 저장한다.

수업의 collect_pomdp_data.py 는 스텝마다 모든 차의 상태까지 저장해서 느리다.
BC 학습에는 관측·행동·에피소드 번호만 있으면 되므로 여기서는 그것만 모은다.
(최종 제출용 데이터는 collect_pomdp_data.py 로 따로 모은다.)

시범 운전자 — env/daehakro_env.py 의 '시범 주행' 설정과 _demo_command / _demo_action 참고.
    · SUMO 의 안전장치를 켠 채 달린다 → 신호·앞차·양보를 SUMO 가 지킨다
    · 보행자: 기다려야 할 보행자가 보이면(관측 crosswalk_ped = 1) 횡단보도 앞에서 선다
    · 앞차와는 SUMO 기본보다 넉넉히 띄우고, 부드럽게(2.5 m/s²) 일찍 멈춘다
    · 스텝마다의 속도 변화를 '정책이 냈어야 할 가감속'으로 바꿔 정답으로 기록한다.
      서 있는 동안과 멈춰 서는 순간은 '브레이크를 확실히 밟음'으로 기록한다
    · 차로변경: 차로 안내(관측 [47])가 가리키는 쪽으로 차로변경 값을 낸다. 옮길 때까지 스텝마다 낸다.
      정책이 운전할 때와 똑같이 '행동의 차로변경 값'으로만 차로가 바뀐다 (SUMO 가 스스로 바꾸지 않는다)
    · 틈 맞추기: 옮겨야 하는데 옆 차로의 차 때문에 못 옮기는 동안은 그 차보다 조금 느리게 가서 뒤쪽 틈으로 빠진다

저장하지 않는 스텝
    · 흔드는 스텝 (--noise): 일부러 차를 굼뜨게 또는 급하게 몰아 평소 가지 않는 상태로 보내는 동안.
      그 뒤 SUMO 가 바로잡는 장면(다시 출발 / 세게 밟기)은 저장한다
    · 이유가 관측에 보이지 않는데 SUMO 가 서 있던 스텝 (SUMO 는 관측에 없는 내부 사정으로도 기다린다)
    · 오래 서 있는 구간의 대부분 (같은 장면의 반복이라 일부만 남긴다)
    · 도착하지 못한 시범 전체 (정체에 갇힌 경우 등)

사용:
    python collect_demo.py --episodes 400 --workers 4 --name demo
    python train_bc.py --data data/demo.npz --epochs 100
    python test.py results/bc_월일_시분/model.pt --episodes 100 --nogui

출력 data/<name>.npz  (train_bc.py 가 읽는 형식)
    observation (N, 49)   행동을 정할 때의 관측
    action_raw  (N, 2)    정답 행동 [가감속, 차로변경]. 차로변경은 +1 (왼쪽으로) / 0 (유지) / −1 (오른쪽으로)
    episode     (N,)      에피소드 번호
    t, reward, terminated, truncated   참고용
"""
import argparse
import multiprocessing
import os
import time
from datetime import datetime

import numpy as np

from env.mdp_config import SIMULATION, OBSERVATION, ACTION, REWARD
from env import daehakro_env
from env.daehakro_env import DaehakroEnv, DEMO_HOLD

BASE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("observation", "action_raw", "episode", "t", "reward", "terminated", "truncated")
HOLD_KEEP_FIRST = 6      # 서 있는 구간에서 처음 이만큼의 스텝은 모두 저장
HOLD_KEEP_RATE = 0.15    # 그 뒤로는 이 비율만 저장
VIOLATIONS = {"red_light": "신호", "crosswalk_violation": "횡단보도", "yield_violation": "양보"}


def collect(worker_id, episodes, sumocfg, seed, gui, tag, noise):
    """SUMO 하나로 episodes(에피소드 번호 목록)만큼 시범을 모은다. 반환: (배열 dict, 요약 dict)."""
    np.random.seed((seed * 1000 + worker_id) % 2**32)      # 환경이 SUMO 시드를 여기서 뽑는다
    rng = np.random.default_rng((seed * 1000 + worker_id) % 2**32)
    env = DaehakroEnv(sumocfg, None, None, SIMULATION, OBSERVATION, ACTION, REWARD,
                      gui=gui, gui_autostart=True, demo=True)
    env.demo_noise = noise
    out = {k: [] for k in KEYS}
    stat = {"episodes": 0, "kept": 0, "steps": 0, "ends": {}, "violations": {}, "disturbed": 0, "unexplained": 0,
            "lane_changes": 0}
    try:
        for ep in episodes:
            env.reset()
            rows = {k: [] for k in KEYS}
            done, t, violations, holding, lane_changes = False, 0, {}, 0, 0
            while not done:
                # 시범 주행에서는 넘긴 행동이 무시된다 (SUMO 가 운전).
                _, reward, terminated, truncated, info = env.step([0.0, 0.0])
                done = terminated or truncated
                for key, label in VIOLATIONS.items():
                    if info[key]:
                        violations[label] = violations.get(label, 0) + 1
                lane_changes += int(info["lane_changed"] != 0)
                stat["disturbed"] += int(info["demo_disturbed"])
                stat["unexplained"] += int(info["demo_unexplained"])
                # 정답이 없는 스텝은 None: 흔드는 스텝, 이유 없이 서 있는 스텝, 마지막 스텝 (env/daehakro_env.py 참고)
                action = info["demo_action"]
                # 오래 서 있는 구간은 같은 장면의 반복이라, 처음 HOLD_KEEP_FIRST 스텝 뒤로는 일부만 남긴다.
                holding = holding + 1 if (action is not None and action[0] == DEMO_HOLD) else 0
                if holding > HOLD_KEEP_FIRST and rng.random() > HOLD_KEEP_RATE:
                    action = None
                if action is not None:
                    # 저장하는 관측은 info["demo_obs"]: SUMO 운전자가 그 스텝에 실제로 본 신호·보행자로 맞춘 것
                    rows["observation"].append(info["demo_obs"])
                    rows["action_raw"].append(action)
                    rows["episode"].append(ep)
                    rows["t"].append(t)
                    rows["reward"].append(reward)
                    rows["terminated"].append(terminated)
                    rows["truncated"].append(truncated)
                t += 1
            end = ("충돌" if info["collided"] else "도착" if info["arrived"]
                   else "경로 이탈" if info["off_route"] else "오래 멈춤" if info["stuck"] else "시간초과")
            stat["episodes"] += 1
            stat["steps"] += t
            stat["ends"][end] = stat["ends"].get(end, 0) + 1
            for label, n in violations.items():
                stat["violations"][label] = stat["violations"].get(label, 0) + n
            if end == "도착":                              # 도착하지 못한 시범(정체에 갇힘 등)은 버린다
                stat["kept"] += 1
                stat["lane_changes"] += lane_changes
                for k in KEYS:
                    out[k].extend(rows[k])
            print(f"{tag}시범 {ep + 1:4d} | {t:4d} 스텝 | {end}"
                  + (" | 위반: " + ", ".join(f"{k} {n}" for k, n in violations.items()) if violations else ""),
                  flush=True)
    finally:
        env.close()
    arrays = {
        "observation": np.asarray(out["observation"], dtype=np.float32).reshape(-1, env.obs_dim),
        "action_raw": np.asarray(out["action_raw"], dtype=np.float32).reshape(-1, 2),
        "episode": np.asarray(out["episode"], dtype=np.int32),
        "t": np.asarray(out["t"], dtype=np.int32),
        "reward": np.asarray(out["reward"], dtype=np.float32),
        "terminated": np.asarray(out["terminated"], dtype=np.bool_),
        "truncated": np.asarray(out["truncated"], dtype=np.bool_),
    }
    return arrays, stat


def main():
    ap = argparse.ArgumentParser(description="BC 용 시범 수집 (SUMO 가 AV 를 운전)")
    ap.add_argument("--episodes", type=int, default=400, help="시범 에피소드 수")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="동시에 돌릴 SUMO 수 (--gui 면 1)")
    ap.add_argument("--name", default=None, help="출력 파일 이름 (생략 시 날짜·시각)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.04,
                    help="흔들기를 시작할 확률(스텝당). 0 이면 흔들지 않는다 (env/daehakro_env.py 참고)")
    ap.add_argument("--gui", action="store_true", help="sumo-gui 로 보면서 수집")
    ap.add_argument("--out-dir", default=os.path.join(BASE, "data"))
    args = ap.parse_args()
    if args.episodes < 2:
        ap.error("--episodes 는 2 이상이어야 합니다 (train_bc.py 가 학습/검증으로 나눕니다).")

    workers = 1 if args.gui else max(1, min(args.workers, args.episodes))
    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or datetime.now().strftime("demo_%Y%m%d_%H%M%S")
    path = os.path.join(args.out_dir, name + ".npz")

    sumocfg = daehakro_env.build()       # 도로 파일은 여기서 한 번만 만든다
    t0 = time.time()
    if workers == 1:
        results = [collect(0, list(range(args.episodes)), sumocfg, args.seed, args.gui, "", args.noise)]
    else:
        chunks = [c.tolist() for c in np.array_split(np.arange(args.episodes), workers)]
        jobs = [(w, chunks[w], sumocfg, args.seed, False, f"[w{w}] ", args.noise) for w in range(workers)]
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            results = pool.starmap(collect, jobs)
    elapsed = time.time() - t0

    data = {k: np.concatenate([r[0][k] for r in results]) for k in KEYS}
    np.savez_compressed(path, **data)

    n = len(data["episode"])
    episodes = sum(r[1]["episodes"] for r in results)
    kept = sum(r[1]["kept"] for r in results)
    steps = sum(r[1]["steps"] for r in results)
    ends = {}
    for r in results:
        for k, v in r[1]["ends"].items():
            ends[k] = ends.get(k, 0) + v
    violations = {}
    for r in results:
        for k, v in r[1]["violations"].items():
            violations[k] = violations.get(k, 0) + v
    accel = data["action_raw"][:, 0] if n else np.zeros(1)
    lane = data["action_raw"][:, 1] if n else np.zeros(1)
    print("\n시범 수집 완료")
    print(f"  에피소드: {episodes} 회 ({', '.join(f'{k} {v}' for k, v in ends.items())}) → 저장 {kept} 회")
    print("  시범 운전자의 위반: "
          + (", ".join(f"{k} {v} 건" for k, v in violations.items()) if violations else "없음"))
    print(f"  저장한 스텝: {n}  (관측 {data['observation'].shape[1] if n else 0}차원)")
    print(f"  저장하지 않은 스텝: 흔든 스텝 {sum(r[1]['disturbed'] for r in results)}, "
          f"이유 없이 서 있던 스텝 {sum(r[1]['unexplained'] for r in results)}")
    print(f"  정답 행동: 브레이크 유지 {np.mean(accel == np.float32(DEMO_HOLD)):.0%}, "
          f"가감속 범위 {accel.min():.2f} ~ {accel.max():.2f}")
    print(f"  차로변경: 저장한 시범에서 {sum(r[1]['lane_changes'] for r in results)} 번 "
          f"(에피소드당 {sum(r[1]['lane_changes'] for r in results) / max(kept, 1):.1f} 번). "
          f"정답이 '왼쪽으로'인 스텝 {np.mean(lane > 0):.1%}, '오른쪽으로' {np.mean(lane < 0):.1%}")
    print(f"  걸린 시간: {elapsed / 60:.1f} 분  ({steps / max(elapsed, 1e-9):.0f} 스텝/초, SUMO {workers} 개)")
    print(f"  파일: {path}  ({os.path.getsize(path) / 2**20:.1f} MB)")
    print(f"\n다음: python train_bc.py --data {os.path.relpath(path, BASE)} --epochs 100")


if __name__ == "__main__":
    main()
