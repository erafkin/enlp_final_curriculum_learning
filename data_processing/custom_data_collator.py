#!/usr/bin/env python3
"""
Custom Data Collator with POS and USAS Semantic Tag-Based Masking

This module provides a custom DataCollatorForLanguageModeling that uses spaCy
for POS tagging and PyMUSAS for UCREL Semantic Analysis System (USAS) tagging,
applying different masking probabilities based on these tags.
Uses word_ids for accurate alignment between spaCy tokens and tokenizer subwords.
"""

import random
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import threading

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import spacy
from spacy.language import Language

class UsasTagAwareDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator using POS/USAS tags and word_ids for alignment.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        pos_tag_weights: Optional[Dict[str, float]] = None,
        sem_tag_weights: Optional[Dict[str, float]] = None,
        spacy_model: str = "en_core_web_sm",
        pad_to_multiple_of: Optional[int] = None,
        cache_tags: bool = True,
    ):
        super().__init__(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of
        )
        self.pos_tag_weights = pos_tag_weights or {}
        self.sem_tag_weights = sem_tag_weights or {}
        self.cache_tags = cache_tags
        self.tag_cache = {}
        self.tag_cache_lock = threading.Lock()
        self.nlp = None
        self.pymusas_available = False
        try:
            self.nlp = spacy.load(spacy_model, exclude=['parser', 'ner'])
            print(f"Loaded spaCy model: {spacy_model}")
            try:
                english_tagger_pipeline = spacy.load('en_dual_none_contextual')
                if 'pymusas_rule_based_tagger' not in self.nlp.pipe_names:
                    self.nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)
                    print("Added PyMUSAS tagger to spaCy pipeline")
                else:
                    print("PyMUSAS tagger already in spaCy pipeline")
                self.pymusas_available = True
            except IOError:
                 warnings.warn("PyMUSAS model 'en_dual_none_contextual' not found. Run 'python -m spacy download en_dual_none_contextual'. USAS tagging disabled.")
                 self.pymusas_available = False
            except Exception as e:
                warnings.warn(f"Failed to load or add PyMUSAS tagger: {e}")
                self.pymusas_available = False
        except IOError:
            warnings.warn(f"spaCy model '{spacy_model}' not found. Run 'python -m spacy download {spacy_model}'. Tagging disabled.")
        except Exception as e:
            warnings.warn(f"Failed to load spaCy model: {e}. Tagging disabled.")

    def get_token_tags_batch(self, batch_texts: List[str]) -> Tuple[List[List[Optional[str]]], List[List[Optional[str]]]]:
        if self.nlp is None:
            num_tokens_est = [len(text.split()) for text in batch_texts]
            empty_tags = [[None] * count for count in num_tokens_est]
            return empty_tags, empty_tags
        
        results_pos = [None] * len(batch_texts)
        results_usas = [None] * len(batch_texts)
        texts_to_process = []
        indices_to_process = []
        text_to_indices_map = defaultdict(list)

        if self.cache_tags:
            with self.tag_cache_lock:
                for i, text in enumerate(batch_texts):
                    if text in self.tag_cache:
                        results_pos[i], results_usas[i] = self.tag_cache[text]
                    else:
                        if text not in text_to_indices_map:
                             texts_to_process.append(text)
                        text_to_indices_map[text].append(i)
                        indices_to_process.append(i) # Keep track of all indices needing processing
        else:
            texts_to_process = batch_texts
            indices_to_process = list(range(len(batch_texts)))
            for i, text in enumerate(batch_texts):
                 text_to_indices_map[text].append(i)

        if texts_to_process:
            docs = self.nlp.pipe(texts_to_process)
            processed_tags = {}
            for doc, original_text in zip(docs, texts_to_process):
                pos_tags = [token.pos_ for token in doc]
                usas_tags = []
                if self.pymusas_available:
                    for token in doc:
                        main_category = None
                        if hasattr(token._, "pymusas_tags") and token._.pymusas_tags:
                            sem_tag = token._.pymusas_tags[0]
                            if sem_tag and len(sem_tag) > 0:
                                main_category = sem_tag[0] if sem_tag[0].isalpha() else None
                        usas_tags.append(main_category)
                else:
                    usas_tags = [None] * len(pos_tags)
                processed_tags[original_text] = (pos_tags, usas_tags)

            with self.tag_cache_lock:
                for text, (pos_tags, usas_tags) in processed_tags.items():
                    # Update cache if needed
                    if self.cache_tags and text not in self.tag_cache:
                        self.tag_cache[text] = (pos_tags, usas_tags)
                    # Fill results for all indices corresponding to this text
                    if text in text_to_indices_map:
                        for idx in text_to_indices_map[text]:
                            results_pos[idx] = pos_tags
                            results_usas[idx] = usas_tags

        # Fallback for any potential misses (shouldn't happen often)
        for i in range(len(batch_texts)):
            if results_pos[i] is None:
                num_tokens_est = len(batch_texts[i].split())
                results_pos[i] = [None] * num_tokens_est
                results_usas[i] = [None] * num_tokens_est

        return results_pos, results_usas

    def calculate_masking_probabilities(
        self,
        batch: Dict[str, torch.Tensor], # Batch containing padded tensors ('input_ids', etc.)
        word_ids_list: List[List[Optional[int]]], # PADDED list of word_ids for each sequence
        batch_pos_tags: List[List[Optional[str]]], # Tags from get_token_tags_batch
        batch_usas_tags: List[List[Optional[str]]] # Tags from get_token_tags_batch
    ) -> torch.Tensor:
        """
        Calculate token-specific masking probabilities using word_ids for alignment.
        """
        input_ids = batch['input_ids']
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Initialize probabilities
        token_probabilities = torch.full_like(input_ids, self.mlm_probability, dtype=torch.float)

        for i in range(batch_size):
            pos_tags = batch_pos_tags[i]
            usas_tags = batch_usas_tags[i]
            word_ids = word_ids_list[i] # Use the padded word_ids list for this sequence

            if pos_tags is None or usas_tags is None: # Check if tagging worked
                 continue # Keep base probabilities, special tokens handled below

            num_spacy_tags = len(pos_tags)

            for j in range(seq_length):
                word_idx = word_ids[j] # word_ids are already aligned with input_ids by padding

                # Skip special tokens (word_idx is None) or padding tokens
                if word_idx is None:
                    token_probabilities[i, j] = 0.0
                    continue

                # Check if word_idx is within the bounds of the obtained tags
                if word_idx < num_spacy_tags:
                    current_prob = self.mlm_probability # Start with base

                    # Apply POS weight
                    pos_tag = pos_tags[word_idx]
                    if pos_tag and pos_tag in self.pos_tag_weights:
                        current_prob = self.pos_tag_weights[pos_tag]

                    # Apply USAS weight
                    if self.pymusas_available:
                        usas_tag = usas_tags[word_idx]
                        if usas_tag and usas_tag in self.sem_tag_weights:
                            current_prob = self.sem_tag_weights[usas_tag]

                    # Assign the calculated probability (clamped)
                    token_probabilities[i, j] = max(0.0, min(1.0, current_prob))
                else:
                    # Word index out of bounds for tags (e.g., due to truncation differences)
                    # Keep base probability (already set)
                    pass

        return token_probabilities

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Override __call__ to handle text/word_ids, calculate probabilities, and perform masking.
        """
        # 1. Extract text and unpadded word_ids
        texts_list = [f['text'] for f in features]
        word_ids_list_unpadded = [f.get('word_ids') for f in features]

        # 2. Pad the features using the default tokenizer.pad logic
        # Exclude 'text' as it's not tensorizable for padding.
        # 'word_ids' might also cause issues if not handled correctly by tokenizer.pad.
        # Let's pad input_ids, attention_mask etc., first.
        features_for_padding = []
        for f in features:
            # Keep only keys that are typically padded (like input_ids, attention_mask)
            # Exclude 'text', 'word_ids' for now.
            padded_f = {k: v for k, v in f.items() if k in self.tokenizer.model_input_names or k == 'attention_mask'}
            features_for_padding.append(padded_f)

        batch = self.tokenizer.pad(
            features_for_padding,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        # Manually pad word_ids to the same length as input_ids
        max_length = batch['input_ids'].shape[1]
        padded_word_ids_list = []
        for word_ids in word_ids_list_unpadded:
            if word_ids is None:
                 padded_word_ids_list.append([None] * max_length)
                 continue
            diff = max_length - len(word_ids)
            padded_word_ids_list.append(word_ids + [None] * diff)
        # Add padded word_ids to the batch if needed elsewhere, though we use the list directly
        # batch['word_ids'] = torch.tensor(padded_word_ids_list) # This might fail if None is present

        # If MLM is disabled, return the padded batch
        if not self.mlm:
            return batch

        # 3. Get spaCy tags using the original texts
        batch_pos_tags, batch_usas_tags = self.get_token_tags_batch(texts_list)

        # 4. Calculate token-specific masking probabilities using padded word_ids list
        token_probabilities = self.calculate_masking_probabilities(
            batch, padded_word_ids_list, batch_pos_tags, batch_usas_tags
        )

        # 5. Perform masking based on calculated probabilities
        inputs = batch['input_ids'].clone()
        labels = inputs.clone()

        # Sample mask based on token probabilities
        probability_matrix = torch.rand_like(inputs, dtype=torch.float)
        masked_indices = probability_matrix < token_probabilities

        # Set labels for non-masked tokens to -100
        labels[~masked_indices] = -100

        # Apply masking strategy (80% MASK, 10% random, 10% original)
        indices_mask = torch.bernoulli(torch.full_like(inputs, 0.8, dtype=torch.float)).bool() & masked_indices
        inputs[indices_mask] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full_like(inputs, 0.5, dtype=torch.float)).bool() & masked_indices & ~indices_mask
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=inputs.device)
        inputs[indices_random] = random_words[indices_random]

        # Update batch with masked inputs and labels
        batch['input_ids'] = inputs
        batch['labels'] = labels

        return batch