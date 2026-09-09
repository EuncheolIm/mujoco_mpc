"""냄비 질량 변형(난이도 2축) XML 을 생성한다.

왜 파일을 갈아 끼우는가: `model->body_mass` 를 런타임에 고치면 (a) 관성/합성질량 파생값을
`mj_setConst` 로 다시 만들어야 하고, (b) 그 변경이 플래너가 들고 있는 모델 사본에 전파되지
않는다(에이전트가 `agent.cc:76` 에서 모델을 복사한다). 즉 롤아웃과 실제 물리가 서로 다른
무게의 냄비를 들게 된다. 파일을 고르면 두 모델이 항상 일치한다.

`Fr3HGripperCoCarry/fr3.cc` 의 `XmlPath()` 가 `MJPC_CC_MASS` 로 이 파일들을 고른다.

task.xml 이나 pot.xml 을 고친 뒤에는 **반드시 다시 실행**한다:
    python3 results/cocarry/gen_mass_variants.py
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TDIR = os.path.join(ROOT, "mjpc", "tasks", "Fr3HGripperCoCarry")
SCALES = {2: "m2", 3: "m3"}


def scale_pot(scale, suffix):
    src = open(os.path.join(TDIR, "pot.xml")).read()
    n = [0]

    def rep(m):
        v = float(m.group(1))
        if v == 0.0:                     # pot_visual: 시각 전용, 질량 0 유지
            return m.group(0)
        n[0] += 1
        return 'mass="%.4g"' % (v * scale)

    out = re.sub(r'mass="([0-9.eE+-]+)"', rep, src)
    out = out.replace(
        '<mujoco model="pot">',
        '<mujoco model="pot_%s">\n  <!-- 자동 생성 파일 -- 직접 고치지 말 것.\n'
        '       results/cocarry/gen_mass_variants.py 가 pot.xml 의 mass 를 x%d 한 것이다.\n'
        '       (pot_visual 의 mass="0" 은 시각 전용이라 그대로 둔다.) -->' % (suffix, scale))
    path = os.path.join(TDIR, "pot_%s.xml" % suffix)
    open(path, "w").write(out)
    return path, n[0], scale * 1.0


def scale_task(suffix):
    src = open(os.path.join(TDIR, "task.xml")).read()
    old = '<include file="pot.xml"/>'
    assert src.count(old) == 1, "task.xml 의 pot include 를 못 찾았다"
    out = src.replace(old, '<include file="pot_%s.xml"/>' % suffix)
    out = out.replace(
        '<mujoco model="Fr3HGripperCoCarry">',
        '<mujoco model="Fr3HGripperCoCarry_%s">\n  <!-- 자동 생성 파일 -- 직접 고치지 말 것.\n'
        '       task.xml 에서 pot include 만 바꾼 것이다 (gen_mass_variants.py).\n'
        '       cost/FSM/keyframe 은 기준선과 동일해야 하므로 여기서 갈라지면 비교가 깨진다. -->'
        % suffix, 1)
    path = os.path.join(TDIR, "task_%s.xml" % suffix)
    open(path, "w").write(out)
    return path


if __name__ == "__main__":
    base = open(os.path.join(TDIR, "pot.xml")).read()
    tot = sum(float(v) for v in re.findall(r'mass="([0-9.eE+-]+)"', base))
    print("기준 pot.xml 질량 합 = %.3f kg" % tot)
    for scale, suffix in sorted(SCALES.items()):
        p, cnt, s = scale_pot(scale, suffix)
        t = scale_task(suffix)
        newtot = sum(float(v) for v in
                     re.findall(r'mass="([0-9.eE+-]+)"', open(p).read()))
        print("x%d: %s (geom %d개, 합 %.3f kg)  +  %s"
              % (scale, os.path.basename(p), cnt, newtot, os.path.basename(t)))
