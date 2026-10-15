import glob, zipfile, sys
wheels = glob.glob('/mnt/wd4tb/scratch/candidate-1514-b51fb194bb-20261015/artifacts/*.whl')
assert wheels, "no wheel was built"
z = zipfile.ZipFile(wheels[0])
name = next(n for n in z.namelist() if n.endswith("dist-info/METADATA"))
md = z.read(name).decode()
lines = [l for l in md.splitlines() if l.startswith("Requires-Python:")]
assert lines, "built wheel has no Requires-Python -- pip cannot decline early"
print(lines[0])
assert "Operating System :: OS Independent" not in md, \
    "OS Independent classifier is back; it is not true (no Windows-on-ARM support)"
