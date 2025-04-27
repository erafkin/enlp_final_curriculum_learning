#!/usr/bin/env python3
"""
Custom Data Collator with POS and USAS Semantic Tag-Based Masking

This module provides a custom DataCollatorForLanguageModeling that uses spaCy
for POS tagging and PyMUSAS for UCREL Semantic Analysis System (USAS) tagging,
applying different masking probabilities based on these tags.
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
    Data collator that applies different masking probabilities based on POS and USAS semantic tags.
    
    This extends the standard DataCollatorForLanguageModeling to provide token-specific
    masking probabilities based on POS and USAS semantic tagging using PyMUSAS.
    Includes optimizations like batch processing and caching.
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
        """
        Initialize the POS and USAS semantic tag-aware data collator.
        
        Args:
            tokenizer: The tokenizer used for encoding the data.
            mlm: Whether to use masked language modeling loss.
            mlm_probability: The base probability of masking a token.
            pos_tag_weights: Dict mapping POS tags to their weight multipliers.
            sem_tag_weights: Dict mapping USAS semantic tags to their weight multipliers.
            spacy_model: The spaCy model to use for POS tagging and as base for USAS tagging.
            pad_to_multiple_of: If set, will pad the sequence to a multiple of the provided value.
            cache_tags: Whether to cache tags for texts to avoid redundant processing.
        """
        super().__init__(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of
        )
        
        # Use weights passed from outside, or default to empty dictionaries
        self.pos_tag_weights = pos_tag_weights or {}
        self.sem_tag_weights = sem_tag_weights or {}
        
        # Settings for optimized processing
        self.cache_tags = cache_tags
        self.tag_cache = {}  # Cache for storing processed text tags
        self.tag_cache_lock = threading.Lock()  # Thread safety for cache
        
        # Initialize spaCy only once during init
        self.nlp = None
        self.pymusas_available = False
        try:
            # Optimize spaCy loading by disabling components we don't need
            self.nlp = spacy.load(spacy_model, exclude=['parser', 'ner'])
            print(f"Loaded spaCy model: {spacy_model}")
            
            # Add PyMUSAS tagger using the recommended approach
            try:
                # Load the English PyMUSAS rule-based tagger in a separate spaCy pipeline
                english_tagger_pipeline = spacy.load('en_dual_none_contextual')
                # Adds the English PyMUSAS rule-based tagger to the main spaCy pipeline
                if 'pymusas_rule_based_tagger' not in self.nlp.pipe_names:
                    self.nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)
                    print("Added PyMUSAS tagger to spaCy pipeline")
                else:
                    print("PyMUSAS tagger already in spaCy pipeline")
                self.pymusas_available = True
            except IOError:
                 warnings.warn("PyMUSAS model 'en_dual_none_contextual' not found. Please run 'python -m spacy download en_dual_none_contextual'. USAS tagging disabled.")
                 self.pymusas_available = False
            except Exception as e:
                warnings.warn(f"Failed to load or add PyMUSAS tagger: {e}")
                self.pymusas_available = False
                
        except IOError:
            warnings.warn(f"spaCy model '{spacy_model}' not found. Please run 'python -m spacy download {spacy_model}'. Tagging disabled.")
        except Exception as e:
            warnings.warn(f"Failed to load spaCy model: {e}. Tagging disabled.")
    
    def get_token_tags_batch(self, batch_texts: List[str]) -> Tuple[List[List[Optional[str]]], List[List[Optional[str]]]]:
        """
        Process a batch of texts to get POS and USAS tags efficiently using nlp.pipe and caching.
        
        Args:
            batch_texts: List of text strings from the batch
            
        Returns:
            Tuple of (pos_tags_list, usas_tags_list) for all texts in the batch
        """
        if self.nlp is None:
            # Estimate token count crudely if NLP model failed to load
            num_tokens_est = [len(text.split()) for text in batch_texts]
            empty_tags = [[None] * count for count in num_tokens_est]
            return empty_tags, empty_tags
        
        results_pos = [None] * len(batch_texts)
        results_usas = [None] * len(batch_texts)
        texts_to_process = []
        indices_to_process = []

        # Check cache first
        if self.cache_tags:
            with self.tag_cache_lock:
                for i, text in enumerate(batch_texts):
                    if text in self.tag_cache:
                        results_pos[i], results_usas[i] = self.tag_cache[text]
                    else:
                        if text not in texts_to_process:
                             texts_to_process.append(text)
                        indices_to_process.append(i)
        else:
            texts_to_process = batch_texts
            indices_to_process = list(range(len(batch_texts)))

        # Process texts not found in cache using nlp.pipe
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

            # Fill results and update cache
            with self.tag_cache_lock:
                for i in indices_to_process:
                     text = batch_texts[i]
                     if text in processed_tags:
                         pos_tags, usas_tags = processed_tags[text]
                         results_pos[i] = pos_tags
                         results_usas[i] = usas_tags
                         if self.cache_tags and text not in self.tag_cache:
                             self.tag_cache[text] = (pos_tags, usas_tags)

        # Ensure all results are filled (fallback if any processing failed unexpectedly)
        for i in range(len(batch_texts)):
            if results_pos[i] is None:
                num_tokens_est = len(batch_texts[i].split())
                results_pos[i] = [None] * num_tokens_est
                results_usas[i] = [None] * num_tokens_est

        return results_pos, results_usas

    def get_token_specific_probabilities(
        self, 
        batch_texts: List[str], 
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate token-specific masking probabilities based on POS and USAS semantic tags.
        Uses optimized batch tagging with caching.
        """
        batch_size, seq_length = input_ids.shape
        
        # Initialize with base probability
        token_probabilities = torch.full_like(input_ids, self.mlm_probability, dtype=torch.float)
        
        # Get POS and semantic tags for batch (optimized batch processing + caching)
        batch_pos_tags, batch_usas_tags = self.get_token_tags_batch(batch_texts)
        
        cpu_input_ids = input_ids.cpu().numpy()
        cpu_token_probabilities = token_probabilities.cpu().numpy()

        special_token_ids = {self.tokenizer.cls_token_id, 
                             self.tokenizer.sep_token_id, 
                             self.tokenizer.pad_token_id}

        for i in range(batch_size):
            pos_tags = batch_pos_tags[i]
            usas_tags = batch_usas_tags[i]
            
            if pos_tags is None or usas_tags is None:
                continue

            tag_cursor = 0
            num_tags = len(pos_tags)

            for j in range(seq_length):
                token_id = cpu_input_ids[i, j]

                if token_id in special_token_ids:
                    cpu_token_probabilities[i, j] = 0.0
                    continue
                
                if tag_cursor >= num_tags:
                    break 

                current_prob = self.mlm_probability
                
                pos_tag = pos_tags[tag_cursor]
                if pos_tag and pos_tag in self.pos_tag_weights:
                    current_prob = self.pos_tag_weights[pos_tag]
                
                if self.pymusas_available:
                    usas_tag = usas_tags[tag_cursor]
                    if usas_tag and usas_tag in self.sem_tag_weights:
                        current_prob = self.sem_tag_weights[usas_tag]
                
                cpu_token_probabilities[i, j] = max(0.0, min(1.0, current_prob))

                tag_cursor += 1
        
        token_probabilities = torch.tensor(cpu_token_probabilities, dtype=torch.float, device=input_ids.device)
        
        return token_probabilities
    
    def torch_mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling using token-specific probabilities.
        """
        batch_texts = [self.tokenizer.decode(ids.tolist(), skip_special_tokens=True) 
                     for ids in inputs]
        
        labels = inputs.clone()
        
        token_probabilities = self.get_token_specific_probabilities(batch_texts, inputs)
        
        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor([
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ], dtype=torch.bool, device=inputs.device)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        
        if self.tokenizer.pad_token_id is not None:
            padding_mask = inputs.eq(self.tokenizer.pad_token_id)
        else:
            padding_mask = torch.zeros_like(inputs, dtype=torch.bool)

        non_maskable_mask = special_tokens_mask | padding_mask
        
        probability_matrix = torch.rand_like(inputs, dtype=torch.float)
        masked_indices = probability_matrix < token_probabilities
        
        masked_indices.masked_fill_(non_maskable_mask, value=False)
        
        labels[~masked_indices] = -100
        
        indices_mask = torch.bernoulli(torch.full_like(inputs, 0.8, dtype=torch.float)).bool() & masked_indices
        inputs[indices_mask] = self.tokenizer.mask_token_id
        
        indices_random = torch.bernoulli(torch.full_like(inputs, 0.5, dtype=torch.float)).bool() & masked_indices & ~indices_mask
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=inputs.device)
        inputs[indices_random] = random_words[indices_random]
        
        return inputs, labels