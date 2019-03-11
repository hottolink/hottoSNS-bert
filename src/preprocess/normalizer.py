#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os, sys, io

import unicodedata
import regex
import html

# String Normalizer Module
# Usage: from normalizer import twitter_normalizer

##charFilter: urlFilter
url_regex = "https?://[-_.!~*\'()a-z0-9;/?:@&=+$,%#]+"
url_pattern = regex.compile(url_regex, regex.IGNORECASE)

##charFilter: partialurlFilter
partial_url_regex = "(((https|http)(.{1,3})?)|(htt|ht))$"
partial_url_pattern = regex.compile(partial_url_regex, regex.IGNORECASE)

##charFilter: retweetflagFilter
rt_regex = "rt (?=\@)"
rt_pattern = regex.compile(rt_regex, regex.IGNORECASE)

##charFilter: screennameFilter
scname_regex = "\@[a-z0-9\_]+:?"
scname_pattern = regex.compile(scname_regex, regex.IGNORECASE)

##charFilter: truncationFilter
truncation_regex = "…$" # NFKC:"...$"
truncation_pattern = regex.compile(truncation_regex, regex.IGNORECASE)

##charFilter: hashtagFilter
hashtag_regex = r"\#\S+"
hashtag_pattern = regex.compile(hashtag_regex, regex.IGNORECASE)

##charFilter: whitespaceNormalizer
ws_regex = "\p{Zs}"
ws_pattern = regex.compile(ws_regex, regex.IGNORECASE)

##charFilter: controlcodeFilter
cc_regex = "\p{Cc}"
cc_pattern = regex.compile(cc_regex, regex.IGNORECASE)

##charFilter: singlequestionFilter
sq_regex = "\?{1,}"
sq_pattern = regex.compile(sq_regex, regex.IGNORECASE)

SPECIAL_TOKENS = {
    "url":"<url>",
    "screen_name":"<mention>"
}


def twitter_normalizer(str_):
    # processing order is crucial.

    #unescape html entities
    str_ = html.unescape(str_)
    #charFilter: strip
    str_ = str_.strip()
    #charFilter: truncationFilter
    str_ = truncation_pattern.sub("", str_)
    #charFilter: icuNormalizer(NKFC)
    str_ = unicodedata.normalize('NFKC', str_)
    #charFilter: caseNormalizer
    str_ = str_.lower()
    #charFilter: retweetflagFilter
    str_ = rt_pattern.sub("", str_)
    ##charFilter: partialurlFilter
    str_ = partial_url_pattern.sub("", str_)
    ##charFilter: screennameFilter
    str_ = scname_pattern.sub(SPECIAL_TOKENS["screen_name"], str_)
    ##charFilter: urlFilter
    str_ = url_pattern.sub(SPECIAL_TOKENS["url"], str_)
    ##charFilter: strip(once again)
    str_ = str_.strip()

    return str_

def question_remover(str_: str):
    return sq_pattern.sub(" ", str_)

def whitespace_normalizer(str_: str):
    str_ = ws_pattern.sub("　", str_)
    return str_

def control_code_remover(str_: str):
    str_ = cc_pattern.sub(" ", str_)
    return str_

def twitter_normalizer_for_bert_encoder(str_):
    # normalizer that is specialized to Twitter BERT Encoder.

    #unescape html entities
    str_ = html.unescape(str_)
    #charFilter: question mark
    str_ = sq_pattern.sub(" ", str_)
    #charFilter: strip
    str_ = str_.strip()
    #charFilter: truncationFilter
    str_ = truncation_pattern.sub("", str_)
    #charFilter: icuNormalizer(NKFC)
    str_ = unicodedata.normalize('NFKC', str_)
    #charFilter: caseNormalizer
    # str_ = str_.lower()
    #charFilter: retweetflagFilter
    str_ = rt_pattern.sub("", str_)
    #charFilter: partialurlFilter
    str_ = partial_url_pattern.sub("", str_)
    #charFilter: screennameFilter
    str_ = scname_pattern.sub(SPECIAL_TOKENS["screen_name"], str_)
    #charFilter: urlFilter
    str_ = url_pattern.sub(SPECIAL_TOKENS["url"], str_)
    #charFilter: control code such as newline
    str_ = cc_pattern.sub(" ", str_)
    #charFilter: strip(once again)
    str_ = str_.strip()

    return str_
