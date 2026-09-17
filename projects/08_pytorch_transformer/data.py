"""Reuse stage 07's corpus split, tokenizer, and shifted windows unchanged."""

from importlib import import_module

prepare_corpus = import_module("projects.07_char_transformer.data").prepare_corpus
