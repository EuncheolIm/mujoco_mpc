#!/usr/bin/env python3
"""Plot a Carry run from MJPC_CARRY_CSV.

  python3 scripts/plot_carry.py run1.csv [run2.csv ...] [-o out.png]

The PNG goes to ~/carry_logs/<first-csv-name>_plot.png unless -o says otherwise -- NOT
next to the CSV, because the CSVs are usually written to /tmp and anything left there is
gone on reboot and a pain to find.

Left  : 3D object trajectory, with the box drawn as a wire cuboid at the START pose and
        at the GOAL pose (real dimensions, rotated by the logged quaternions, so a tipped
        box shows up as a tipped box).
Right : |obj - tgt| over time, DECOMPOSED. |obj-tgt| alone is ambiguous, because the
        object is driven through the grasp: a residual error is either the arm not
        having arrived or the box having shifted in the jaws, and those need opposite
        fixes. So the grasp offset obj-ee is expressed in the HAND frame, where it is
        constant for a rigid grasp -- its drift from the value at first contact is
        SLIP, and |obj-tgt| minus that is the arm's own tracking error. Also plots
        |cam - obj| when the camera is valid; the stretch where injection is paused
        (held=1) is shaded, and inside it cam vs obj is SUPPOSED to diverge.
Several runs given together are overlaid, so configurations compare directly.

numpy + matplotlib only: the system python3 has those but no pandas, and the judo venv
has neither, so this deliberately avoids both.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BOX = np.array([0.040, 0.094, 0.174]) / 2.0   # half-extents, m

def quat2R(q):
    n = np.linalg.norm(q)
    if n < 1e-12: return np.eye(3)
    w, x, y, z = q / n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])

def wire(ax, c, q, col, lw=1.4, ls="-", label=None):
    R = quat2R(q)
    s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    v = (R @ (s * BOX).T).T + c
    E = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    for k, (a, b) in enumerate(E):
        ax.plot(*zip(v[a], v[b]), color=col, lw=lw, ls=ls,
                label=label if k == 0 else None)

def slip_mm(d):
    """Drift of the grasp offset, in mm, measured in the hand frame.

    obj - ee is a world vector, so it rotates as the arm turns even for a perfectly
    rigid grasp -- expressing it in the hand frame removes that and leaves only real
    motion of the box relative to the pads. Referenced to the first sample where the
    box is off the table, which is the closest thing to "the moment of grasp" that the
    raw log gives us. Returns (slip, t0_index) or (None, None) if ee was not logged.
    """
    if "ee_x" not in d.dtype.names: return None, None
    o, e = col3(d, "obj_"), col3(d, "ee_")
    q = quat4(d, "ee_")
    g = np.zeros_like(o)
    for k in range(len(d)):
        g[k] = quat2R(q[k]).T @ (o[k] - e[k])       # world -> hand frame
    lifted = np.where(o[:, 2] > o[0, 2] + 0.02)[0]  # 20 mm off its start height
    i0 = lifted[0] if len(lifted) else 0
    return np.linalg.norm(g - g[i0], axis=1) * 1000, i0

def col3(d, pre):  return np.column_stack([d[pre + s] for s in ("x", "y", "z")])
def quat4(d, pre): return np.column_stack([d[pre + s] for s in ("qw","qx","qy","qz")])

def main(paths, out_arg=None):
    fig = plt.figure(figsize=(14, 6))
    a3 = fig.add_subplot(1, 2, 1, projection="3d")
    a2 = fig.add_subplot(1, 2, 2)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    allpts, loaded, shaded = [], [], False

    for i, p in enumerate(paths):
        # A missing file must not kill the runs that DO exist: the usual call lists
        # several runs and only some have been recorded yet.
        if not os.path.exists(p):
            print(f"skip {p}: not found"); continue
        d = np.genfromtxt(p, delimiter=",", names=True)
        if d.ndim == 0: d = d.reshape(1)
        # Row 0 is written before the object pose is first read, so it is all zeros.
        # Left in, it puts the START box at the origin and rescales the whole 3D view.
        keep = np.any(col3(d, "obj_") != 0.0, axis=1)
        if not keep.all():
            print(f"{os.path.basename(p)}: dropped {int((~keep).sum())} pre-init row(s)")
            d = d[keep]
        if len(d) == 0:
            print(f"skip {p}: no valid rows"); continue
        tag = os.path.basename(p).replace(".csv", "")
        loaded.append(p)
        c = cols[i % len(cols)]
        o, t = col3(d, "obj_"), col3(d, "tgt_")
        allpts += [o, t]

        a3.plot(o[:,0], o[:,1], o[:,2], color=c, lw=1.8, label=f"{tag} object")
        wire(a3, o[0],  quat4(d,"obj_")[0],  c, ls="--", label=f"{tag} start")
        wire(a3, t[-1], quat4(d,"tgt_")[-1], c, lw=2.2,  label=f"{tag} goal")

        err = np.linalg.norm(o - t, axis=1) * 1000
        a2.plot(d["t"], err, color=c, lw=1.8, label=f"{tag}  |obj-tgt|")

        sl, i0 = slip_mm(d)
        if sl is not None:
            a2.plot(d["t"], sl, color=c, lw=1.3, ls="--",
                    label=f"{tag}  slip (grasp offset drift)")
            a2.axvline(d["t"][i0], color=c, lw=0.8, alpha=.5)

        if "cam_ok" in d.dtype.names and np.any(d["cam_ok"] == 1):
            m = d["cam_ok"] == 1
            ce = np.full(len(d), np.nan)
            ce[m] = np.linalg.norm(col3(d,"cam_")[m] - o[m], axis=1) * 1000
            a2.plot(d["t"], ce, color=c, lw=1.1, ls=":", label=f"{tag}  |cam-obj|")

        if "held" in d.dtype.names and np.any(d["held"] == 1) and not shaded:
            h = (d["held"] == 1).astype(int)
            e = np.diff(np.concatenate(([0], h, [0])))
            starts, ends = np.where(e == 1)[0], np.where(e == -1)[0]
            for k, (s0, s1) in enumerate(zip(starts, ends)):
                a2.axvspan(d["t"][s0], d["t"][min(s1, len(d)-1)], color="0.86", zorder=0,
                           label="held (injection paused)" if k == 0 else None)
            shaded = True

        line = (f"{tag}: {len(d)} rows, {d['t'][-1]:.2f} s | "
                f"final |obj-tgt| {err[-1]:.1f} mm | min {err.min():.1f} mm")
        if sl is not None:
            line += f" | slip final {sl[-1]:.1f} max {sl.max():.1f} mm (ref t={d['t'][i0]:.2f}s)"
        else:
            line += " | slip: n/a (no ee_ columns -- rebuild and re-run)"
        print(line)

    if not allpts:
        sys.exit("no usable CSV given")
    P = np.vstack(allpts)
    ctr = P.mean(0); rng = max(np.ptp(P, axis=0).max(), 0.25) / 2 * 1.3
    a3.set_xlim(ctr[0]-rng, ctr[0]+rng); a3.set_ylim(ctr[1]-rng, ctr[1]+rng)
    a3.set_zlim(ctr[2]-rng, ctr[2]+rng)
    a3.set_xlabel("x [m]"); a3.set_ylabel("y [m]"); a3.set_zlabel("z [m]")
    a3.set_title("object trajectory  (dashed = start box, solid = goal box)")
    a3.legend(fontsize=7, loc="upper left")
    a2.set_xlabel("t [s]"); a2.set_ylabel("error [mm]")
    a2.set_title("|obj - tgt|  vs  grasp slip")
    a2.grid(alpha=.3); a2.legend(fontsize=8)
    if out_arg:
        out = os.path.abspath(out_arg)
        d = os.path.dirname(out)
        if d: os.makedirs(d, exist_ok=True)
    else:
        outdir = os.path.join(os.path.expanduser("~"), "carry_logs")
        os.makedirs(outdir, exist_ok=True)
        base = os.path.basename(os.path.splitext(loaded[0])[0])
        out = os.path.join(outdir, base + "_plot.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)

if __name__ == "__main__":
    a = sys.argv[1:]
    out_arg = None
    if "-o" in a:
        i = a.index("-o")
        if i + 1 >= len(a): sys.exit("-o needs a path")
        out_arg = a[i + 1]
        a = a[:i] + a[i + 2:]
    if not a: sys.exit(__doc__)
    main(a, out_arg)
