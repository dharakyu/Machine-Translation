# Machine-Translation
As my project for the quarter, I sought to replicate the approach described in 
[Towards Neural Machine Translation with Partially Aligned Corpora](https://www.aclweb.org/anthology/I17-1039.pdf).

If you'd like to try it on your machine (although I cannot guarantee good results...), 
clone on to your machine using `git clone https://github.com/dharakyu/Machine-Translation`. Make sure you have
the dependencies installed (NLTK and PyTorch should cover everything) and run `python phrase_based_model.py`. I ran it
as is on the slowest Google Cloud GPU and it took about an hour.

A bit about how this repo is organized: the `tutorials` folder contains the code from the PyTorch NLP tutorials,
and the `project` folder contains the source code from my project. Within `project` there is a `data` folder which
contains a few different datasets. 

`dev.en` and `dev.es` are small parallel English-Spanish datasets from the [CS 224N](http://web.stanford.edu/class/cs224n/) 
machine translation assignment, for the phrase matching portion of the translator.

`en_unligned` and `es_unaligned` are monolingual corpora used for training. They're samples from the massive [Europarl
dataset](https://www.statmt.org/europarl/).

The folders `en` and `es` contain the entirety of the Europarl dataset in English and Spanish, broken up into multiple .txt
files for space reasons.

If you want to create another dataset, use `head -n number_of_lines data/file.txt > sample.txt` in Terminal.

### Next steps:

I would like to perform a few more tests with and without the modified attention mechanism, as I'm not sure that it's working effectively. Also should modify the loss function. Additionally, I might try different preprocessing methods for Spanish words
(i.e. keep as is vs remove accents vs remove all accented characters).
