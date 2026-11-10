import json
import random
import sys

NOISE = list("!'+,-./:;<=>?[]_{}")
PLAIN = "abcdefghijklmnopqrstuvwxyz "
CIPHER = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ")
SRC = sys.argv[1]
OUT = sys.argv[2]
MODE = sys.argv[3]
PERIOD = int(sys.argv[4]) if len(sys.argv) > 4 else 7
NOISE_FRAC = 0.115
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0

rng = random.Random(SEED)
key = [rng.randrange(63) for _ in range(PERIOD)]
q = NOISE_FRAC

with open(SRC, encoding="utf-8") as f, \
     open(OUT + "_cipher.jsonl", "w", encoding="utf-8") as fc, \
     open(OUT + "_cleaned.jsonl", "w", encoding="utf-8") as fp:
    for line in f:
        rec = json.loads(line)
        text = rec["text"]
        out = []
        real_i = 0
        raw_i = 0
        for ch in text:
            while rng.random() < q:
                out.append(rng.choice(NOISE))
                raw_i += 1
            pos = real_i if MODE == "real" else raw_i
            if MODE == "mono":
                idx = (2 * PLAIN.index(ch) + key[0]) % 63
            else:
                idx = (2 * PLAIN.index(ch) + key[pos % PERIOD]) % 63
            out.append(CIPHER[idx])
            real_i += 1
            raw_i += 1
        fc.write(json.dumps({"id": rec["id"], "ciphertext": "".join(out)}) + "\n")
        fp.write(json.dumps({"id": rec["id"], "text": text}) + "\n")

