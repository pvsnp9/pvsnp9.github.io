---
layout: distill
title: Embedding & Toeknization
description: Understanding embedding and toeknization in natural language processing. An implementation of Byte Pair Encoding.
tags: Embedding
giscus_comments: false
date: 2024-05-03

authors:
  - name: Tsuyog Basnet
    url: "https://www.linkedin.com/in/tsuyog/"
    affiliations:
      name: Vector Lab

toc:
  - name: Embeddings 
  - name: Tokenization
    subsections:
      - name: Introduction
      - name: Text tokenization
      - name: Creating Vocabulary
        subsections:
          - name: Special tokens
      - name: Byte Pair Encoding (BPE)
        subsections:
          - name: Algorithm 
          - name: Implementation
  - name: Conclusion
  - name: Citations

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---



> Lets try to communicate with computers.
<br>
<div class="row justify-content-sm-center">
    
{% include figure.html path="assets/img/intro.png" title="NLP" class="img-fluid rounded z-depth-1" %}
</div>
<br>

A user is curious to know someone's well being. But, the recipient is perplexed about what was asked!! This is exactly why we need tokenization and embedding to communicate with machines (computers) via natural language.

## Embedding

"Embedding is the process of representing words or sentences as numerical vectors in a multi-dimensional space. These numerical representations capture semantic meaning and relationships, enabling machines to understand and process natural language more effectively in various tasks."

Imagine you have a dictionary, and instead of definitions, each word has a unique set of numbers associated with it. These numbers are like coordinates in a multi-dimensional space. Each word's set of numbers represents its meaning and context in relation to other words.

For example, in this numerical space, the word `"Nepal"` might be represented by the numbers `(0.3, -0.1, 0.5)`, while `"Everest"` might be represented by `(0.2, -0.3, 0.6)`. Notice how similar words like "Nepal" and "Everest" have similar sets of numbers, indicating their semantic similarity.

Now, these numerical representations, or embeddings, are incredibly useful for machines. They allow computers to understand the meaning of words based on their context and relationships with other words. This understanding is crucial for tasks like sentiment analysis, machine translation, text classification and text generation.

The embedding technique is not only limited to langauge/text but also is employed in audio and video embeddings.

<div class="row justify-content-sm-center">
    
{% include figure.html path="assets/img/embeddings.png" title="Embeddings" class="img-fluid rounded z-depth-1" %}
</div>

There are several embedding models such as WOrd2Vec, GloVe(Global Vectors for Word Representation), FastText, BERT (Bidirectional Encoder Representations from Transformers), ELMo (Embeddings from Language Models), and etc.

**If every word has single unique vector representation, which is good. But same word (token) has different meaning based on the sentence.**

Here is an example: `"I traveled to Nepal to explore the breathtaking Himalayan mountains."` and `"One of my friend's name is X Nepal."`. Words often have multiple senses or meanings, and their interpretation can change based on the words. First example referes to country due to context like `"to explore," "breathtaking," and "Himalayan mountains"`, and second represents a noun with context of `"friend", "name"`. Contextualized embedding models like BERT and ELMo are designed to capture these nuances by considering the surrounding words when generating word embeddings, allowing for more accurate representations of words in different contexts.

<div class="row justify-content-sm-center">
    
{% include figure.html path="assets/img/embedding_viz.png" title="Embedding visaulization" class="img-fluid rounded z-depth-1" %}
</div>
<div class="caption">Visaulization of two dimensional embedding vectors. However, the embedding dimensions are different according to model. Similar items are in same cluster, in another words they have shorter distance. </div>


***

## Tokenization

Tokenization is the process of breaking down a text into smaller units called tokens. These tokens can be words, phrases, symbols, or any meaningful units, which are then used for further analysis in natural language processing (NLP) tasks.

### Introduction

Imagine you have a long paragraph of text. Tokenization is like chopping it into smaller, manageable pieces, kind of like slicing a cake into individual slices. But instead of using a knife, we use rules to decide where to make these cuts.

Each slice we get after chopping is called a "token". Tokens can be words, but they can also be punctuation marks, numbers, or even emojis! Essentially, anything that makes sense as a unit in the text.

For example, in the sentence "I love natural language processing!", the tokens would be "I", "love", "natural", "language", "processing", and "!". Each of these is a separate token.

Once we've chopped up our text into tokens, we can do all sorts of cool things with them, like counting how many times each word appears, figuring out the meaning of a sentence, or even teaching computers to understand and generate human-like language!


<div class="row justify-content-sm-center">
    
{% include figure.html path="assets/img/tokenizer.png" title="Tokenization process" class="img-fluid rounded z-depth-1" %}
</div>
<div class="caption">Visaulization of tokenization in LLM. First, we split the input text into individual tokens that are either words or characters, and encode using tools like tiktoken or custom encoder.</div>

Here is sample code split the input sentence:

```python
import re

input_text = f"I traveled to Nepal to explore the breathtaking Himalayan mountains."
tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
tokens = [token.strip() for token in tokens if token.strip()]
['I', 'traveled', 'to', 'Nepal', 'to', 'explore', 'the', 'breathtaking', 'Himalayan', 'mountains', '.']
```
We must employ complex regex patterns extract useful text from dataset. 

### Creating Vocabulary

Vocabulary refers to a set of unique words that occur in a given corpus or dataset. It represents the entire range of words used in the text data being analyzed. In python dialect, it is a dictionary for all possible words in corpus mapped to numerical IDs.

To understand, imagine you have a large collection of books. Each book contains many different words, right? Now, if you were to make a list of all the unique words that appear across all the books, that list would be your vocabulary.

So, essentially, a vocabulary is like a dictionary of words used in a specific context or dataset. It includes every distinct word found in the text data, without repetition.

For example, if you were analyzing a set of news articles, your vocabulary might include words like "politics," "election," "economy," and so on.


<div class="row justify-content-sm-center">
    
{% include figure.html path="assets/img/vocab.png" title="vocabulary illustration" class="img-fluid rounded z-depth-1" %}
</div>
<div class="caption">Illustration for creating tokenizer vocabulary.The values are taken from tiktoken encoding.</div>

**Qestion: any number and any range would do the job?** Yes, the sole purpose of vocabulary is lookup, string to id and vice versa. In case of size, it is task dependent.

Following code will demonstrate a simple implementation (reference to previous code example):
```python 
unique_words = sorted(list(set(tokens)))
vocab = {token:i for i, token in enumerate(unique_words)}
for item in vocab.items(): print(item)

('.', 0)
('Himalayan', 1)
('I', 2)
('Nepal', 3)
('breathtaking', 4)
('explore', 5)
('mountains', 6)
('the', 7)
('to', 8)
('traveled', 9)
```
Using the prior knowledge, lets create a very simple tokenizer. 
```python
class ToeknizerV1:
    def __init__(self, vocab:dict):
        self.s_to_i=  vocab
        self.i_to_s = {i:s for s,i in vocab.items()}
        
    def encode(self, text:str)->list[int]:
        # use regex to tokenize the input text
        tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
        tokens = [token.strip() for token in tokens if token.strip()]
        
        idxs = [self.s_to_i[token] for token in tokens]
        return idxs
    
    def decode(self, idxs:list[int])->str:
        text = " ".join([self.i_to_s[idx] for idx in idxs])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```

<div class="row justify-content-sm-center">
    
{% include figure.html path="assets/img/enc-dec.png" title="vocabulary illustration" class="img-fluid rounded z-depth-1" %}
</div>
<div class="caption">Illustration for encoding-decoding in tokenizer.</div>

Lets see it in action

```python
tokenizer_v1 = ToeknizerV1(vocab)
idxs = tokenizer_v1.encode(input_text)
text = tokenizer_va.decode(idxs)
print(f"{idxs}\n{text}")

```
The encoder output is `[IDS]`
```python
[2, 9, 8, 3, 8, 5, 7, 4, 1, 6, 0]
```
The decoder outpus is `'str'`
```python
'I traveled to Nepal to explore the breathtaking Himalayan mountains.'
```

Try this:

```python
try_input = f"Kathmandu is capital city of Nepal"
tokenizer_v1.encode(input_text)
```
**oops...**
```python 
KeyError: 'Kathmandu'
```
#### Special tokens

The token `'Kathmandu'` does not exists in our vocab. Hence, the vocab should be rich enough to afford all tokens. However, a quick fix is to add special tokens such as `<|unk|>` , `<|sos|>`, and `<|eos|>`. 
Lets extend the unique words and vocabulary by adding special tokens.
```python 
unique_words.extend(["<|unk|>" , "<|sos|>","<|eos|>"])
vocab = {token:i for i, token in enumerate(unique_words)}
print(vocab)
```
New vocab list: 
```python
{'.': 0,
 'Himalayan': 1,
 'I': 2,
 'Nepal': 3,
 'breathtaking': 4,
 'explore': 5,
 'mountains': 6,
 'the': 7,
 'to': 8,
 'traveled': 9,
 '<|unk|>': 10,
 '<|sos|>': 11,
 '<|eos|>': 12}
```
wait wait.... we still need to modify `Tokenizer` encoding method.

```python
class ToeknizerV2:
    def __init__(self, vocab:dict):
        self.s_to_i=  vocab
        self.i_to_s = {i:s for s,i in vocab.items()}
        
    def encode(self, text:str)->list[int]:
        # use regex to tokenize the input text
        tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
        tokens = [token.strip() for token in tokens if token.strip()]
        tokens = [item if item in self.s_to_i else "<|unk|>" for item in tokens]
        
        idxs = [self.s_to_i[token] for token in tokens]
        return idxs
    
    def decode(self, idxs:list[int])->str:
        text = " ".join([self.i_to_s[idx] for idx in idxs])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
```
Test out new tokenizer with `out-of-bag/vocabulary (OOB)` text.
```python
try_input = f"Kathmandu is capital city of Nepal.<|eos|>"
t_v2 = ToeknizerV2(vocab)
t_v2.encode(try_input)
```
The encoder output
```python
[10, 10, 10, 10, 10, 3, 0, 12]
```
```python
t_v2.decode(t_v2.encode(try_input))
```
the decoder output:
```python
'<|unk|> <|unk|> <|unk|> <|unk|> <|unk|> Nepal. <|eos|>'
```
umm..... we solve the error but it is not going be useful during model training. In the following section we will develop BPE to address such issue.

***

### Byte Pair Encoding (BPE)

Byte Pair Encoding (BPE) is a subword tokenization technique used in natural language processing (NLP) to break down words into smaller, meaningful units called subword tokens. This technique is widely used in tasks such as machine translation, text generation, and language modeling.

Similar tokenization techniques include WordPiece and SentencePiece, which are also subword tokenization methods that segment words into smaller units. These techniques are particularly useful for handling out-of-vocabulary words, morphologically rich languages, and reducing the size of the vocabulary in NLP models.

Imagine you have a large collection of words in a language, like English. Some words are very common, like "the" or "and," while others are rare or even completely new. Byte Pair Encoding helps in representing all these words by breaking them down into smaller parts called subword tokens.

#### Algorithm

- **Step 1: Vocabulary initialization** - Start with a vocabulary containing all unique characters present in the training data.
- **Step 2: Merging:**- Iteratively merge the most frequent pair of adjacent subword tokens in the vocabulary. Repeat this process for a specified number of iterations or until a certain vocabulary size is reached.
- **Step 3: Vocabulary Expansion** -  After each merge, update the vocabulary to include the newly created subword tokens.
- **Step 4: Tokenization** - Segment input text into subword tokens based on the learned vocabulary. Replace words not in the vocabulary with a special token, such as `<unk>` for unknown.

By iteratively applying these steps, Byte Pair Encoding effectively captures both frequent and rare patterns in the data, leading to compact and efficient representations for a wide range of words in the language.

<div class="row justify-content-sm-center">
    
{% include figure.html path="assets/img/bpe.png" title="vocabulary illustration" class="img-fluid rounded z-depth-1" %}
</div>
<div class="caption">Illustration for byte pair encoding tokenizer. It breaks down unknown words into subwords and characters.</div>

Let's use Byte Pair Encoding (BPE) to tokenize the input text. We'll start by initializing the vocabulary with individual unique characters, then iteratively merge the most frequent pairs of adjacent subword tokens. For this example, let's perform two iterations of BPE:

**Initialization:**
- Vocabulary: `{"I", " ", "t", "r", "a", "v", "e", "l", "d", "N", "p", "l", "o", "x", "u", "h", "b", "m", "i", "n", "s", "."}`

**Iteration 1:**
- Most frequent pair: `("t", "o")`
- Merge `"t"` and `"o"` to create a new subword token `"to"`.
- Updated vocabulary: `{"I", " ", "to", "r", "a", "v", "e", "l", "d", "N", "p", "l", "o", "x", "u", "h", "b", "m", "i", "n", "s", "."}`

**Iteration 2:**
- Most frequent pair: `("e", " ")`
- Merge `"e"` and `" "` to create a new subword token `"e "`.
- Updated vocabulary: `{"I", " ", "to", "r", "a", "v", "e ", "l", "d", "N", "p", "l", "o", "x", "u", "h", "b", "m", "i", "n", "s", "."}`

Now, let's tokenize the text using this vocabulary:

**Text tokenization:**
`["I", " ", "trav", "eled", " ", "to", " ", "N", "ep", "al", " ", "to", " ", "exp", "lor", "e", " ", "the", " ", "br", "eat", "htaking", " ", "H", "ima", "lay", "an", " ", "m", "oun", "tains", "."]`

This tokenization captures both common patterns like `"to"` and `"e "` as well as less frequent patterns like "trav" and "moun", resulting in a more compact representation of the text compared to simple word tokenization.

#### BPE Implmentation

We are going to use unicode characters to make more generic tokenizer. Lets create dataset.

```python

import os, json
import re
import regex as re

raw_demo_text = '''😄 0123456789 Webb telescope probably didn't find life on an exoplanet -- yet Claims of biosignature gas detection were premature. Recent reports of NASA's James Webb Space Telescope finding signs of life on a distant planet understandably sparked excitement. A new study challenges this finding, but also outlines how the telescope might verify the presence of the life-produced gas. Source:University of California - Riverside. २० वैशाख, काठमाडौं । वेस्टइन्डिज ‘ए’ले चौथो टी २० खेलमा घरेलु टोली नेपाललाई २८ रनले हराउँदै एक खेल अगावै सिरिज जितेको छ ।पाँच खेलको सिरिजमा वेस्टइन्डिज ३–१ ले अघि छ । पहिलो खेल हारे पनि त्यसपछि लगातार तीन खेलमा जितका साथ सिरिज जितेको हो । ११० रनको लक्ष्य पछ्याएको नेपाल २० ओभरमा १८१ रनमा अल आउट भएको छ । नेपालका कप्तान रोहित पौडेलले ४७ बलमा ७ चौका र ५ छक्कासहित सर्वाधिक ८२ रन बनाए पनि जित्नलाई पर्याप्त भएन । रोहितले पहिलो खेलमा ११२, दोस्रो खेलमा ७१ रन र तेस्रो खेलमा विश्राम गरेका थिए । रोहित बाहेक आजको खेलमा पनि अन्य ब्याटरले राम्रो प्रदर्शन गर्न सकेनन् । स्पिनर हेडन वाल्स जुनियरको १४औं ओभरको तेस्रो बलमा रोहित लङअफमा म्याथ्यु फर्डबाट क्याच आउट भएपछि शतकको अवसर गुमाएका थिए । ओपनर आसिफ शेख शून्य र कुशल भुर्तेल १ रनमा आउट भएका थिए । कुशल मल्ल ४, सन्दीप जोरा, दीपेन्द्रसिंह ऐरी र गुलशन झा १९–१९ रनमा आउट भए । अभिनाश बोहरा ८ बलमा २ चौका र १ छक्कासहित १७ र सोमपाल कामी १० रनमा आउट भए । Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪!'''

```
In this implementation, we will be using `utf-8` encode for some advatanges. lets map all `str` to `UTF-8` format and then convert it into a list of integers representing the bytes of the UTF-8 encoded string.

```python

tokens = raw_demo_text.encode('utf-8')
tokens = list(map(int, tokens))

print("====================================")
print(f"Text length: {len(raw_demo_text)}\n")
print(raw_demo_text[:100])
print("====================================")
print(f"Token length: {len(tokens)}")
print(tokens[:50])

```
Output:
```python
====================================
Text length: 1349

😄 0123456789 Webb telescope probably didn't find life on an exoplanet -- yet Claims of biosignature 
====================================
Token length: 2846
[240, 159, 152, 132, 32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 32, 87, 101, 98, 98, 32, 116, 101, 108, 101, 115, 99, 111, 112, 101, 32, 112, 114, 111, 98, 97, 98, 108, 121, 32, 100, 105, 100, 110, 39, 116, 32, 102, 105, 110]

```

Lets create a function that creates a consecutive pairs of chracters present in the dataset.

```python 
def get_pair_data(ids):
    '''
    @params: list[int]
    returns: {(pairs: tuples): int(occurance) }
    '''
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts
```
example usage: 
```python
pair_stats = get_pair_data(tokens)
print(pair_stats)
```
output:
```python 
{(240, 159): 15, (159, 152): 1, (152, 132): 1, (132, 32): 1, (32, 48): 1, (48, 49): 1, (49, 50): 1, (50, 51): 1, (51, 52): 1, (52, 53): 1}
```
lets sort them by value (occurance) and pick top k.
```python
print(sorted(((v,k) for k,v in pair_stats.items()), reverse=True))
[(531, (224, 164)), (177, (224, 165)), (171, (32, 224)), (60, (164, 190)), (50, (164, 176)), (40, (176, 224)), (40, (164, 178))]
```
Lets understand the token merge and vocabulary extension in BPE. For instance, 

vacob: ```{"a", "b", "c", "d"} ```

input_text (dataset): `"aaabdaaabac"`

pair tokens: ```{"aa": 3, "ab": 1, "bd": 1, "da": 2, "ac": 1}```

Iteration 1:

Most frequent pair: "aa" (frequency = 3)

Merge `"a"` and `"a"` to create a new subword token `"Z"`.

Updated vocabulary: ```{"a", "b", "c", "d", "Z"}```, and updated text ```ZabdZabac```

new pair tokens: ``` {"ab":2, "bd":1,"dZ":1, "ac":1 } ```

Iteration 2:

Most frequent pair: "ab" (frequency = 2)

Merge `"a"` and `"b"` to create a new subword token `"Y"`.

Updated vocabulary: ```{"a", "b", "c", "d", "Z", "Y"}```, and updated text ```ZYdZYac```

new pair tokens: ``` {"ZY":2, "Yd":1, "ac":1} ```


Iteration 3:

Most frequent pair: `"ZY"` (frequency = 2)

Merge `"Z"` and `"Y"` to create a new subword token `"X"`.

Updated vocabulary: ```{"a", "b", "c", "d", "Z", "Y", "X"}```, and updated text ```XdXac```

**Here we have compressed the sequence length from 11 to 5, but increased the vocab size from 4 to 7.**

Lets see this in code:
```python 
def merge(ids, pair, idx):
    # list of ints (ids), replace all consecutive occurances of pair with new token idx 
    newids = []
    i = 0
    while i <len (ids):
        # if we are not at very last position AND the pair matches, replace it 
        if i < len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

```
example usage:

```python 
print(f"demo: {merge([1,2,2,3,5], (2,3), 999)}")
```
output:  ```demo: [1, 2, 999, 5]``` pair `(2,3)` is replaced with `999`.

Here, we create the vocab size of 333 (a hyperparameter), and merge tokens. Here, we are using utf-8 that is 256 tokens of raw bytes. The merged token will have IDs from 256. Following codes runs 77 (333-256) iterations and creates new merged tokens.

```python 
voacab_size = 333
required_merges = voacab_size - 256
# new copy of ids 
ids = list(tokens)

merges = {}
for i in range(required_merges):
    pair_stats = get_pair_data(ids)
    pair = max(pair_stats, key=pair_stats.get)
    idx = 256 + i
    print(f"merging {pair} into new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx
```
Output:
```
merging (224, 164) into new token 256
merging (224, 165) into new token 257
merging (32, 256) into new token 258
merging (256, 190) into new token 259
merging (257, 141) into new token 260
merging (260, 256) into new token 261
merging (256, 191) into new token 262
.....................................
.....................................
merging (116, 101) into new token 332
```
Compression result:
```python
print(f"toekn length: {len(tokens)}, ids length: {len(ids)} \ncompression ratio: {(len(tokens) / len(ids)):.2f}X")
```
```toekn length: 2846, ids length: 1094  compression ratio: 2.60X```. We have compressed the text by 2.6 X. 
 
create the vocab. it stores the byte information for given IDs.
```python 
vocab = { idx: bytes([idx]) for idx in range(256) }
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
```

Here, we will create a encoder and decoder block. The encoder returns the token IDs `list[int]` for a given text, and decoder returns the text for given `list[int]`.

```python 
# given a text, return token (list of integers)
def encode(text:str)->list[int]:
    # extract raw bytes, converted to list of integers
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        pair_stats = get_pair_data(tokens)
        pair = min(pair_stats, key=lambda p: merges.get(p, float('inf')))
        if pair not in merges: break # nothing can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens    

# given list of integers, returns python string
def decode(ids:list[int])->str:
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text
    
```
example usage:
```python
enc = encode("I traveled to Nepal to explore the breathtaking Himalayan mountains.")
dec = decode(enc)
print(f"encoded: {enc}\ndecoded:{dec}")
```
```encoded: [73, 32, 116, 114, 97, 118, 101, 108, 101, 100, 32, 116, 111, 32, 78, 101, 112, 97, 108, 32, 116, 111, 32, 101, 120, 112, 108, 111, 114, 277, 116, 104, 277, 98, 114, 101, 97, 116, 104, 116, 97, 107, 304, 103, 32, 72, 105, 109, 97, 108, 97, 121, 317, 32, 109, 111, 117, 110, 116, 97, 304, 115, 46]```

```decoded:I traveled to Nepal to explore the breathtaking Himalayan mountains.```

Now, lets create simple BPE version 1 tokenizer. This tokenizer will first take dataset i.e. text to construct vocabulary and merges with highly frequent pairs until desired vocab size. The instance then can encode and decode the test and IDs repectively.


```python
class BPEV1:
    def __init__(self, dataset:str, vocab_size:int) -> None:
        self.vocabulary = {}
        self.merges = {}
        self.vocab_size = vocab_size
        self.tokens = self.__parse_text(dataset)
        self.__create_vocab()
        del self.tokens

    
    def encode(self, text:str)->list[int]:
        tokens = self.__parse_text(text)
        while len(tokens) >= 2:
            pair_stats = self.__get_pair_data(tokens)
            pair = min(pair_stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges: break # nothing can be merged
            idx = self.merges[pair]
            tokens = self.__merge(tokens, pair, idx)
        return tokens  
        
        
    def decode(self, ids:list[int])->str:
        tokens = b"".join(self.vocabulary[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    
    def __parse_text(self, text:str)->list[int]:
        tokens = text.encode("utf-8")
        return list(map(int, tokens))
    
    def __get_pair_data(self, ids:list[int]):
        '''
        @params: list[int]
        returns: {(pairs: tuples): int(occurance) }
        '''
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def __merge(self, ids:list[int], pair:tuple, idx:int):
        newids = []
        i = 0
        while i <len (ids):
            # if we are not at very last position AND the pair matches, replace it 
            if i < len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    def __create_vocab(self):
        new_vocabs = self.vocab_size - 256
        assert new_vocabs >= 1, "vocab size must be greater than 256"
        self.vocabulary = { idx: bytes([idx]) for idx in range(256)}
        # merging
        for i in range(new_vocabs):
            pair_stats = get_pair_data(self.tokens)
            pair = max(pair_stats, key=pair_stats.get)
            idx = 256 + i
            self.tokens = self.__merge(self.tokens, pair, idx)
            self.merges[pair] = idx
        
        for (p0, p1), idx in self.merges.items():
            self.vocabulary[idx] = self.vocabulary[p0] + self.vocabulary[p1]
    
```
example usage:
```python

bpev1 = BPEV1(dataset=raw_demo_text, vocab_size=333)

enc = bpev1.encode("I traveled to Nepal to explore the breathtaking Himalayan mountains.")
dec = bpev1.decode(enc)
print(f"encoded: {enc}\ndecoded:{dec}")

```
Output: look at previous section.

```python 

# validation
test_txt = ''' म रोयल नेपाल गल्फ क्लबमा खेल्थें । उहाँले सोही समय गल्फ खेल्न सुरु गर्नु भएको हो । उहाँले मलाई सानो भाइको रुपमा माया गर्नुहुन्थ्यो ।'''
val_txt = bpev1.decode(bpev1.encode(txt))
print(test_txt == val_txt)
```
output: `True`


Finally, we have implemented the Byte pair encoder and decoder on small text dataset. However, this can be extended to larger dataset as well. It is robust and capable to accomodate most of the unseen texts.

Take a look at the following, example python script:
```python
tt = '''
import tiktoken
sample_text = "आइएमइ अध्यक्ष चन्द्र प्रसाद ढकालले बाह्रखरीसँग गरेको कुराकानीमा आधारित"
#GPT2 
enc = tiktoken.get_encoding("gpt2")
print(enc.encode(sample_text))

#GPT4 (merge space)
enc = tiktoken.get_encoding("cl100k_base")
gpt4tokens = enc.encode(sample_text)
print(gpt4tokens)'''

enc = bpev1.encode(tt)
bpev1.decode(enc)

```
Output:
```'\nimport tiktoken\nsample_text = "आइएमइ अध्यक्ष चन्द्र प्रसाद ढकालले बाह्रखरीसँग गरेको कुराकानीमा आधारित"\n#GPT2 \nenc = tiktoken.get_encoding("gpt2")\nprint(enc.encode(sample_text))\n\n#GPT4 (merge space)\nenc = tiktoken.get_encoding("cl100k_base")\ngpt4tokens = enc.encode(sample_text)\nprint(gpt4tokens)'```


Aweesome !!


## Conclusion
We have discussed embedding and tkonization techniques primarily what, why, and how? Every NLP tasks need to be tokenized and embedded in order to train, fine-tune, and generate text from deep learning models such as BERT, GPT, llama, mistral and etc. Embedding is extended to encode any non numerical data such as signals and vision.

***

## Citations
- Thanks to GenAI for content writing. 
- A <a href="https://www.youtube.com/watch?v=zduSFxRajkE&t=6359s">tutorial</a> by A. Karpathy.

