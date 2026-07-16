"""Self-contained MPPI + ACT-prior harness (touches NO existing mjpc C++).

Implements the paper's three prior-injection modes exactly (Algorithm 1), in
joint-position space (position actuators), with the handoff_single_target ACT
policy as the prior U_p:

  standard    : sample ~ N(U, Sigma),  U carried/self-correcting,  alpha=0
  warmstart   : sample ~ N(U_p, Sigma), re-centered at U_p each step, alpha=0
  cost-residual: sample ~ N(U, Sigma),  U carried,  cost += alpha*||v-U_p||^2

U_p = ACT chunk (desired joint angles) resampled onto the MPPI horizon.
Task cost = reach: EE position (+ light orientation) error to the commanded target.

Usage:
  python mppi_prior_harness.py --mode warmstart --sigma 0 --ty 0.0
  python mppi_prior_harness.py --mode warmstart --sigma 0 --ty 0.35
"""
import argparse, os, numpy as np, mujoco, onnxruntime as ort

BUILD = "/home/kkomji/Euncheol/mujoco_mpc/build/mjpc/tasks/Fr3OodSim2Real"
CKPT = f"{BUILD}/checkpoints/act_single_target"
HOME = np.array([0, -0.78539816, 0, -2.35619449, 0, 1.57079632, 0.78539816])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["standard", "warmstart", "cost"], default="warmstart")
    ap.add_argument("--tx", type=float, default=0.5)
    ap.add_argument("--ty", type=float, default=0.0)
    ap.add_argument("--tz", type=float, default=0.336)
    ap.add_argument("--sigma", type=float, default=0.05, help="per-joint sampling std (rad)")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--H", type=int, default=10, help="horizon knots")
    ap.add_argument("--alpha", type=float, default=1.0, help="cost-residual weight")
    ap.add_argument("--lam", type=float, default=0.05, help="MPPI temperature")
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.chdir(BUILD)
    m = mujoco.MjModel.from_xml_path("task.xml")
    d = mujoco.MjData(m)
    ee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hand_site")
    jr = m.jnt_range[:7].copy()
    sess = ort.InferenceSession(f"{CKPT}/act.onnx", providers=["CPUExecutionProvider"])

    ctrl_dt = 0.02                                  # 50 Hz (matches ACT)
    nsub = max(1, round(ctrl_dt / m.opt.timestep))  # sim substeps per control step
    target = np.array([args.tx, args.ty, args.tz])
    rng = np.random.default_rng(args.seed)
    W_POS = 100.0

    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)

    scratch = mujoco.MjData(m)

    def act_chunk():
        q, qd = d.qpos[:7].copy(), d.qvel[:7].copy()
        e = d.site_xpos[ee].copy()
        s = np.concatenate([q, qd, e]).astype(np.float32)[None]
        g = np.zeros((1, 6), np.float32)
        return sess.run(None, {"state": s, "goal": g})[0].reshape(40, 7)

    def U_p_from_chunk(chunk):
        # resample 40-step ACT chunk (dt=0.02) onto H knots at ctrl_dt=0.02 -> steps 0..H-1
        return np.stack([chunk[min(t, 39)] for t in range(args.H)])  # (H,7)

    def rollout(seq):
        # seq: (H,7) position setpoints. Returns summed reach cost from current state.
        scratch.qpos[:] = d.qpos; scratch.qvel[:] = d.qvel
        scratch.act[:] = d.act if m.na else scratch.act
        mujoco.mj_forward(m, scratch)
        c = 0.0
        for t in range(args.H):
            scratch.ctrl[:7] = np.clip(seq[t], jr[:, 0], jr[:, 1])
            for _ in range(nsub):
                mujoco.mj_step(m, scratch)
            p = scratch.site_xpos[ee]
            c += W_POS * np.sum((p - target) ** 2)
        return c

    U = np.tile(HOME, (args.H, 1))                  # carried nominal (standard/cost)
    nsteps = int(args.secs / ctrl_dt)
    for k in range(nsteps):
        chunk = act_chunk()
        Up = U_p_from_chunk(chunk)                  # (H,7) prior control seq
        center = Up if args.mode == "warmstart" else U
        # sample K perturbations
        noise = rng.normal(0.0, args.sigma, size=(args.K, args.H, 7))
        seqs = center[None] + noise                 # (K,H,7)
        costs = np.empty(args.K)
        for i in range(args.K):
            c = rollout(seqs[i])
            if args.mode == "cost":
                c += args.alpha * np.sum((seqs[i] - Up) ** 2)
            costs[i] = c
        w = np.exp(-(costs - costs.min()) / args.lam)
        w /= w.sum()
        newU = np.tensordot(w, seqs, axes=(0, 0))   # (H,7) weighted average
        # execute first control on the real sim
        d.ctrl[:7] = np.clip(newU[0], jr[:, 0], jr[:, 1])
        for _ in range(nsub):
            mujoco.mj_step(m, d)
        # nominal carry: standard/cost shift-and-hold; warmstart discards (re-seed U_p next step)
        if args.mode != "warmstart":
            U = np.vstack([newU[1:], newU[-1]])

    p = d.site_xpos[ee]
    dist = float(np.linalg.norm(p - target))
    dprior = float(np.linalg.norm(p - np.array([0.5, 0.0, 0.336])))
    print(f"mode={args.mode} sigma={args.sigma} target=({args.tx},{args.ty},{args.tz}) "
          f"| final ee=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
          f"| dist_to_target={dist*1000:.1f}mm | dist_to_prior(0.5,0,0.336)={dprior*1000:.1f}mm")


if __name__ == "__main__":
    main()
