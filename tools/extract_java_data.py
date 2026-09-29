"""One-off: pull the lexicon, lemma dictionary and stopwords out of the original Java sources."""
import json, re, sys, pathlib
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lex = {}
for m in re.finditer(r'add\("([^"]+)",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)',
                     (src / "features/EmotionLexicon.java").read_text()):
    lex[m.group(1)] = [float(x) for x in m.groups()[1:]]
pre = (src / "preprocessing/Preprocessor.java").read_text()
lem = dict(re.findall(r'put\("([^"]+)",\s*"([^"]+)"\)', pre))
block = re.search(r'STOPWORDS = new HashSet<>\(Arrays.asList\((.*?)\)\);', pre, re.S).group(1)
stop = sorted(set(re.findall(r'"([^"]+)"', block)))
(out / "lexicon.json").write_text(json.dumps(lex))
(out / "lemmas.json").write_text(json.dumps(lem))
(out / "stopwords.json").write_text(json.dumps(stop))
print(len(lex), "lexicon words,", len(lem), "lemmas,", len(stop), "stopwords")
