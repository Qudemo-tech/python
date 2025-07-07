#!/usr/bin/env python3
"""
Standalone script for asking questions about company video content
Usage: python ask_question.py <company_name> <question>
"""

import sys
import logging
from dotenv import load_dotenv
from main import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

def main():
    if len(sys.argv) < 3:
        print("Usage: python ask_question.py <company_name> <question>")
        sys.exit(1)
    
    company_name = sys.argv[1]
    question = " ".join(sys.argv[2:])
    
    logger.info(f"Company: {company_name}")
    logger.info(f"Question: {question}")
    
    try:
        result = answer_question(company_name, question)
        
        if "error" in result:
            logger.error(f"❌ Error: {result['error']}")
            sys.exit(1)
        
        print("\n" + "="*50)
        print("ANSWER:")
        print("="*50)
        print(result["answer"])
        print("\n" + "="*50)
        print("SOURCES:")
        print("="*50)
        for i, source in enumerate(result["sources"], 1):
            print(f"{i}. {source}")
        print("\n" + "="*50)
        print(f"PRIMARY VIDEO: {result['video_url']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 