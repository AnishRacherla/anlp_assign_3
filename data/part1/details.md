1. Dataset Dimensions & Splits

I moved and organized the data into the required data/part1/ structure:

train/: 569,670 sequences (~578 MB cipher, ~510 MB cleaned text).
validation/: 1,153 sequences (~1.2 MB).
test/: 1,318 sequences (~1.3 MB).
Domain: Natural text from Wikipedia articles (titles, lead paragraphs, and section bodies).
Sequence Lengths: Range from 3 characters to ~10,300 characters (mean sequence length: ~1,020 characters).

2. The Cryptographic Breakdown & Key Discovery
1) The Plaintext Vocabulary (27 Characters)
The plaintext consists strictly of:
Lowercase letters a–z (26 characters)
Space ' ' (1 character)
Total: 27 unique characters.
2) The Ciphertext Vocabulary (81 Characters)
The raw ciphertext contains 81 characters (
27
×
3
27×3):
Digits 0–9 (10)
Lowercase a–z (26)
Uppercase A–Z (26)
Space ' ' (1)
18 punctuation/bracket symbols (18): ! ' + , - . / : ; < = > ? [ ] _ { }
3. The Big Breakthrough: The Noise-Injection Mechanism 🎯

When inspecting why ciphertexts are slightly longer than plaintexts, I analyzed the exact distribution of characters:

In 100% of samples across validation and train:
len
(
ciphertext
)
−
len
(
plaintext
)
=
Count of the 18 punctuation symbols in ciphertext
len(ciphertext)−len(plaintext)=Count of the 18 punctuation symbols in ciphertext
Every single symbol among !'+,-./:;<=>?[]_{} is a synthetic noise / dummy character inserted at random intervals.
When you strip these 18 symbols, the ciphertext length becomes EXACTLY EQUAL to the plaintext length (
1
-to-
1
1-to-1 alignment).



Overall corpus entropy: 
𝐻
(
plain
∣
cipher
)
=
4.12
 bits
H(plain∣cipher)=4.12 bits (the unigram entropy of English).
When we examined individual samples:
For the same word (e.g. "valkyria"), Sample 0 is suD6OZeQ, Sample 1 is 7x6c 3S8, and Sample 3 is 1p5GD OJ.
Within each document, single cipher characters map to different plaintext characters depending on temporal position and state.
Conclusion: The Dominion cipher is a polyalphabetic/stream substitution cipher where the key/state shifts dynamically over time.
This is the ideal proving ground for Mamba and DeltaNet: because the mapping changes over time, models with recurrent hidden state tracking are fundamentally suited to crack it!