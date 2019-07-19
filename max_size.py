import glob
import os.path

sizes = [(0, "unk")]*10

for file in glob.glob("txt/**/*.txt", recursive=True):
    with open(file) as f:
        sizes.append((
            len(f.read()),
            os.path.basename(file)
        ))
    sizes = sorted(sizes, key=lambda x: x[0], reverse=True)
    sizes = sizes[:10]


from pprint import pprint

pprint(sizes)