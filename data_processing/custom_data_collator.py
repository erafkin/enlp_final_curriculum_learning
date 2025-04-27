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
        """
        super().__init__(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of
        )
        
        # Default weights for POS tags if not provided
        self.pos_tag_weights = pos_tag_weights or {
            "NOUN": 0.15,     # Nouns are masked more frequently
            "VERB": 0.15,     # Verbs are masked more frequently
            "ADJ": 0.15,      # Adjectives are masked more frequently
            "ADV": 0.15,      # Adverbs are masked more frequently
            "PRON": 0.15,     # Pronouns are masked less frequently
            "DET": 0.15,      # Determiners are masked less frequently
            "ADP": 0.15,      # Prepositions are masked less frequently
            "CCONJ": 0.15,    # Conjunctions are masked less frequently
            "PUNCT": 0.15,    # Punctuation is rarely masked
        }
        
        # Default weights for USAS semantic tags if not provided
        # These are based on USAS major categories
        self.sem_tag_weights = sem_tag_weights or {
            "A": 0.15,    # General and abstract terms (most conceptual)
            "B": 0.15,    # The body and the individual
            "C": 0.15,    # Arts and crafts
            "E": 0.15,    # Emotion
            "F": 0.15,    # Food and farming
            "G": 0.15,    # Government and public
            "H": 0.15,    # Architecture, housing and the home
            "I": 0.15,    # Money and commerce
            "K": 0.15,    # Entertainment, sports and games
            "L": 0.15,    # Life and living things
            "M": 0.15,    # Movement, location, travel and transport
            "N": 0.15,    # Numbers and measurement
            "O": 0.15,    # Substances, materials, objects and equipment
            "P": 0.15,    # Education
            "Q": 0.15,    # Language and communication
            "S": 0.15,    # Social actions, states and processes
            "T": 0.15,    # Time
            "W": 0.15,    # World and environment
            "X": 0.15,    # Psychological actions, states and processes
            "Y": 0.15,    # Science and technology
            "Z": 0.15,    # Names and grammar (least conceptual)
        }
        
        # Initialize spaCy and PyMUSAS following official documentation
        try:
            # Load the base spaCy model excluding components we don't need
            self.nlp = spacy.load(spacy_model, exclude=['parser', 'ner'])
            print(f"Loaded spaCy model: {spacy_model}")
            
            # Add PyMUSAS tagger using the recommended approach
            try:
                # Load the English PyMUSAS rule-based tagger in a separate spaCy pipeline
                english_tagger_pipeline = spacy.load('en_dual_none_contextual')
                # Adds the English PyMUSAS rule-based tagger to the main spaCy pipeline
                self.nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)
                print("Added PyMUSAS tagger to spaCy pipeline")
                self.pymusas_available = True
            except Exception as e:
                warnings.warn(f"Failed to load PyMUSAS tagger: {e}")
                self.pymusas_available = False
                
        except Exception as e:
            warnings.warn(f"Failed to load spaCy model: {e}")
            self.nlp = None
            self.pymusas_available = False
    
    def get_token_pos_tags(self, batch_texts: List[str]) -> List[List[str]]:
        """
        Get POS tags for each token in the batch of texts.
        
        Args:
            batch_texts: List of text strings from the batch
            
        Returns:
            List of lists containing POS tags for each token
        """
        if self.nlp is None:
            return [[None] * len(text.split()) for text in batch_texts]
        
        batch_pos_tags = []
        
        for text in batch_texts:
            doc = self.nlp(text)
            pos_tags = [token.pos_ for token in doc]
            batch_pos_tags.append(pos_tags)
            
        return batch_pos_tags
    
    def get_token_usas_tags(self, batch_texts: List[str]) -> List[List[str]]:
        """
        Get USAS semantic tags for each token in the batch of texts using PyMUSAS.
        
        Args:
            batch_texts: List of text strings from the batch
            
        Returns:
            List of lists containing USAS semantic tags for each token
        """
        if self.nlp is None or not self.pymusas_available:
            return [[None] * len(text.split()) for text in batch_texts]
        
        batch_usas_tags = []
        
        for text in batch_texts:
            # Process text with spaCy and PyMUSAS
            try:
                doc = self.nlp(text)
                
                # Extract primary semantic tags (first tag from each token)
                usas_tags = []
                for token in doc:
                    # PyMUSAS stores the semantic tags in token._.pymusas_tags attribute
                    main_category = None
                    if hasattr(token._, "pymusas_tags") and token._.pymusas_tags:
                        sem_tag = token._.pymusas_tags[0]
                        # Extract just the main category (first letter before any separator)
                        if sem_tag and len(sem_tag) > 0:
                            # USAS tags often have format like "A1.1.1" or "Z99"
                            # We just want the first letter (main category)
                            main_category = sem_tag[0] if sem_tag[0].isalpha() else None
                    
                    usas_tags.append(main_category)
                
                batch_usas_tags.append(usas_tags)
            except Exception as e:
                warnings.warn(f"Error getting USAS tags: {e}")
                # If tagging fails, return None for all tokens
                batch_usas_tags.append([None] * len(text.split()))
            
        return batch_usas_tags
    
    def get_token_specific_probabilities(
        self, 
        batch_texts: List[str], 
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate token-specific masking probabilities based on POS and USAS semantic tags.
        
        Args:
            batch_texts: List of text strings
            input_ids: Tensor of token IDs
            
        Returns:
            Tensor containing token-specific masking probabilities
        """
        batch_size, seq_length = input_ids.shape
        
        # Initialize with base probability
        token_probabilities = torch.ones_like(input_ids, dtype=torch.float) * self.mlm_probability
        
        # Get POS and semantic tags for batch
        batch_pos_tags = self.get_token_pos_tags(batch_texts)
        batch_usas_tags = self.get_token_usas_tags(batch_texts)
        # print(f"Batch POS tags: {batch_pos_tags}")
        # print(f"Batch USAS tags: {batch_usas_tags}")
        # input("Press Enter to continue...")
        
        # Convert token IDs back to tokens to align with spaCy tokenization
        batch_tokens = []
        for ids in input_ids:
            tokens = self.tokenizer.convert_ids_to_tokens(ids.tolist())
            batch_tokens.append(tokens)
        
        # This is a simplified approach - in practice, you'd need to handle
        # alignment between tokenizer's tokens and spaCy/PyMUSAS tokens
        for i in range(batch_size):
            text = batch_texts[i]
            doc = self.nlp(text) if self.nlp else None
            
            if doc:
                # Simple alignment strategy (will need refinement for real use)
                pos_cursor = 0
                for j in range(seq_length):
                    # Skip special tokens
                    if input_ids[i, j] in [self.tokenizer.cls_token_id, 
                                         self.tokenizer.sep_token_id, 
                                         self.tokenizer.pad_token_id]:
                        token_probabilities[i, j] = 0.0  # Don't mask special tokens
                        continue
                    
                    # Apply POS-based weights when possible
                    if pos_cursor < len(batch_pos_tags[i]):
                        pos_tag = batch_pos_tags[i][pos_cursor]
                        if pos_tag in self.pos_tag_weights:
                            token_probabilities[i, j] = self.pos_tag_weights[pos_tag]
                    
                    # Apply USAS semantic tag-based weights when possible
                    if pos_cursor < len(batch_usas_tags[i]):
                        usas_tag = batch_usas_tags[i][pos_cursor]
                        if usas_tag and usas_tag in self.sem_tag_weights:
                            token_probabilities[i, j] = self.sem_tag_weights[usas_tag]
                    
                    pos_cursor += 1
        
        # Clip probabilities to be between 0 and 1
        token_probabilities = torch.clamp(token_probabilities, 0.0, 1.0)
        
        return token_probabilities
    
    def torch_mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 
        80% MASK, 10% random, 10% original.
        This overrides the parent class method to use token-specific probabilities.
        """
        # Retrieve batch texts to get POS and USAS tags
        batch_texts = [self.tokenizer.decode(ids.tolist(), skip_special_tokens=True) 
                     for ids in inputs]
        
        labels = inputs.clone()
        
        # Get token-specific probabilities
        token_probabilities = self.get_token_specific_probabilities(batch_texts, inputs)
        
        # Create special tokens mask if not provided
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        
        # Create probability mask for each token
        probability_matrix = torch.rand_like(inputs, dtype=torch.float)
        probability_matrix = probability_matrix < token_probabilities
        
        # Don't mask special tokens
        probability_matrix.masked_fill_(special_tokens_mask, value=False)
        
        # Don't mask [PAD] tokens
        if self.tokenizer.pad_token_id is not None:
            padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=False)
        
        # Mask tokens
        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # The rest of the time (10% of the time) keep the masked input tokens unchanged
        return inputs, labels