#!/usr/bin/env python3
"""
Aggregate Dataset Script

This script reads the data files in the train, test, and dev folders and creates 
a new aggregated file for each directory containing all the cleaned separate data files.
"""

import os
import glob
import logging
import argparse
import re
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_line(line):
    """
    Clean a line of text based on the file type and format.
    
    Args:
        line: The line to clean
    
    Returns:
        Cleaned line of text
    """
    # First, strip leading/trailing whitespace
    line = line.strip()
    
    # Skip empty lines
    if not line:
        return ""
    
    # Remove speaker indicators (childes format: *XXX:)
    if line.startswith('*') and ':' in line[:10]:
        line = re.sub(r'^\*\w+:\s*', '', line)
    
    # Remove speaker indicators (switchboard format: A:, B:, etc.)
    if re.match(r'^[A-Z]:\s', line):
        line = re.sub(r'^[A-Z]:\s*', '', line)
    
    # Remove chapter headers and formatting (gutenberg format: *XXXX*)
    if line.startswith('*') and line.endswith('*'):
        return ""  # Skip chapter headers
    if line.startswith('=') and line.endswith('='):
        return ""  # Skip chapter headers
    
    # Remove inline formatting like *s* in gutenberg texts
    line = re.sub(r'\*[a-zA-Z]+\*', '', line)
    
    # Remove subtitle formatting (e.g., "- Text" at the beginning)
    line = re.sub(r'^-\s+', '', line)
    
    # Remove any remaining special characters
    line = re.sub(r'[^\w\s.,!?;:\'"\-]', ' ', line)
    
    # Replace multiple spaces with a single space
    line = re.sub(r'\s+', ' ', line)

    # Remove lines with less than 3 words
    if len(line.split()) < 3:
        return ""
    
    # Final trim
    return line.strip()

def aggregate_files(input_dir, output_file, sample_only=False):
    """
    Aggregate all files in the input directory into a single output file.
    
    Args:
        input_dir: Directory containing the files to aggregate
        output_file: Path to the output aggregated file
        sample_only: If True, only process a sample of the data (for testing)
    """
    logger.info(f"Processing files in {input_dir}")
    
    # Get all files in the directory
    files = glob.glob(os.path.join(input_dir, '*'))
    
    if not files:
        logger.warning(f"No files found in {input_dir}")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in sorted(files):
            if not os.path.isfile(file_path):
                continue
                
            file_name = os.path.basename(file_path)
            logger.info(f"Processing {file_name}")
            
            lines_processed = 0
            
            # For large files, we don't want to read everything into memory at once
            try:
                with open(file_path, 'r', encoding='utf-8') as in_f:
                    for line_num, line in enumerate(in_f, 1):
                        # Clean the line based on the file type
                        cleaned_line = clean_line(line)
                        
                        # Skip empty lines
                        if not cleaned_line:
                            continue
                            
                        # Write the cleaned line to the output file
                        out_f.write(cleaned_line + '\n')
                        
                        lines_processed += 1
                        
                        # If sample_only, limit to first 1000 lines per file
                        if sample_only and lines_processed >= 1000:
                            logger.info(f"Sample limit reached for {file_name}, processed {lines_processed} lines")
                            break
                        
                        # Log progress for large files
                        if lines_processed % 100000 == 0:
                            logger.info(f"Processed {lines_processed} lines from {file_name}")
                
                total_lines += lines_processed
                logger.info(f"Completed {file_name}: {lines_processed} lines processed")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info(f"Successfully aggregated {len(files)} files into {output_file}")
    logger.info(f"Total lines processed: {total_lines}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Aggregate dataset files from train, test, and dev directories.')
    parser.add_argument('--data-dir', default='data', help='Path to the data directory')
    parser.add_argument('--output-dir', default='data/aggregated', help='Output directory for aggregated files')
    parser.add_argument('--sample', action='store_true', help='Process only a sample of the data (for testing)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each subdirectory
    for split in ['train', 'dev', 'test']:
        input_dir = data_dir / split
        output_file = output_dir / f"{split}.{split}"
        
        if os.path.isdir(input_dir):
            logger.info(f"Processing {split} directory")
            success = aggregate_files(input_dir, output_file, args.sample)
            
            if success:
                logger.info(f"Successfully created {output_file}")
            else:
                logger.warning(f"Failed to process {split} directory")
        else:
            logger.warning(f"Directory not found: {input_dir}")

if __name__ == "__main__":
    main()