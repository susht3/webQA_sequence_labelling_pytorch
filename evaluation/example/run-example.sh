#!/bin/bash
set -x
# strict matching evaluation
python ../evaluate-tagging-result.py \
  ./raw_prediction_example.txt \
  ./test_file_example.json.gz \
  -o strict-matching.ref

# fuzzy matching evaluation
python ../evaluate-tagging-result.py -f \
  ./raw_prediction_example.txt \
  ./test_file_example.json.gz \
  -o fuzzy-matching.ref

# strict matching evaluation
python ../evaluate-voting-result.py \
  ./raw_prediction_example.txt \
  ./test_file_example.json.gz \
  -o vote-strict-matching.ref

# fuzzy matching evaluation
python ../evaluate-voting-result.py -f \
  ./raw_prediction_example.txt \
  ./test_file_example.json.gz \
  -o vote-fuzzy-matching.ref
