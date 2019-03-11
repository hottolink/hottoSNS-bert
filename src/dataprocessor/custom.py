#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, csv

from .preset import DataProcessor, InputExample
import tokenization


# dataset processor for Twitter日本語評判分析データセット [Suzuki+, 2017]
class PublicTwitterSentimentProcessor(DataProcessor):
  """
  Processor for the Twitter日本語評判分析データセット .
  refer to: http://bigdata.naist.jp/~ysuzuki/data/twitter/

  """

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["pos", "neg", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type in ["train","dev"]:
        text_a = tokenization.convert_to_unicode(line[7])
        label = tokenization.convert_to_unicode(line[3])
      elif set_type == "test":
        text_a = tokenization.convert_to_unicode(line[7])
        label = tokenization.convert_to_unicode(line[3])
      else:
        raise NotImplementedError(f"unsupported set type: {set_type}")

      if label in self.get_labels():
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples
