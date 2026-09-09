## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update tasks/lessons.md with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

### 7. Periodic Progress Reporting
- **Long-running background work (sweeps, experiments, builds) MUST be reported on
  periodically — do NOT sit silent until completion.** Each report includes:
  progress (done/total), the interim aggregate numbers so far, current CPU load,
  and a revised time estimate.
- Read partial results straight off the output file/CSV (`wc -l` + `awk`) instead
  of waiting for the final summary.
- **CPU budget**: total workers (concurrent processes x threads per process) must
  stay at or below `nproc / 2`, and every run gets `nice -n 15`. The user works on
  the same machine — once in 2026-08 a 30-process sweep killed their VS Code
  session. Sweep drivers must self-throttle (no unbounded `&` + `wait`).

## Task Management

1. **Plan First**: Write plan to tasks/todo.md with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to tasks/todo.md
6. **Capture Lessons**: Update tasks/lessons.md after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimat Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Pot cooperative carry (the only task line on this branch)

Three tasks remain: `FR3_H_Gripper_Dual`, `FR3_H_Gripper_PotDual`,
`FR3_H_Gripper_CoCarry`. Everything else was removed.

Read before touching them:
- `tasks/todo_cocarry.md`, `tasks/todo_cocarry_pregrasp.md` — CoCarry work log,
  including which failure causes have been ruled out and with what evidence.
- `tasks/todo_pot_dual.md` — PotDual design decisions.
- `mjpc/planners/FlowMPPI/DETERMINISM.md` — why the sampler had to move off
  `absl::BitGen`, and what else is needed for a run to be reproducible.

Two settings are load-bearing and easy to get wrong: `MJPC_PLANNER=9` (the
task.xml default of 14 uses a single softmax and cannot even reach pre-grasp)
and `MJPC_FM_CONFIG` pointing at the task's own `fm_config.yaml` (without it
the loader falls back to a single-arm FM prior).
