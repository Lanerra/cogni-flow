# Paragraph-Based Dynamic Sampling for Ollama LLMs
# -----------------------------------------------
# This implementation enables dynamic sampling for Ollama models by breaking generation
# into paragraph-sized chunks, analyzing each paragraph, and adjusting sampling
# parameters for the next paragraph.

import re
import time
import requests
import json
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from transformers import pipeline

class OllamaDynamicSampling:
    def __init__(
        self,
        model: str = "llama3.3:70b-instruct-q3_K_M",
        ollama_base_url: str = "http://192.168.64.1:11434",
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        base_temperature: float = 0.7,
        base_top_p: float = 0.9,
    ):
        """Initialize with base sampling parameters, Ollama endpoint, and sentiment analyzer."""
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.api_url = f"{ollama_base_url}/api/generate"
        
        self.base_params = {
            "temperature": base_temperature,
            "top_p": base_top_p,
            "top_k": 40,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        self.current_params = self.base_params.copy()
        
        # Load sentiment analysis model
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
        
        # Define content analyzers
        self.analyzers = {
            "sentiment": self._analyze_sentiment,
            "complexity": self._analyze_complexity,
            "creativity": self._analyze_creativity,
            "formality": self._analyze_formality,
        }
        
        # Sampling parameter adjustment strategies
        self.adjustment_strategies = {
            "positive_sentiment": {"temperature": 0.8, "top_p": 0.92},
            "negative_sentiment": {"temperature": 0.6, "top_p": 0.85},
            "high_complexity": {"temperature": 0.5, "top_p": 0.8},
            "low_complexity": {"temperature": 0.75, "top_p": 0.9},
            "high_creativity": {"temperature": 0.9, "top_p": 0.95, "frequency_penalty": 0.2},
            "low_creativity": {"temperature": 0.4, "top_p": 0.8, "frequency_penalty": 0.0},
            "formal": {"temperature": 0.5, "top_p": 0.85},
            "informal": {"temperature": 0.8, "top_p": 0.9},
        }
        
        # Check if the model exists in Ollama
        self._verify_model()
    
    def _verify_model(self):
        """Verify that the requested model is available in Ollama."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model.get("name") for model in models]
                if self.model not in available_models:
                    print(f"Warning: Model '{self.model}' not found in Ollama. Available models: {available_models}")
            else:
                print(f"Warning: Could not verify models. Ollama returned status code {response.status_code}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
    
    def generate_text(self, prompt: str, max_paragraphs: int = 5) -> str:
        """Generate text with dynamically adjusted sampling parameters by paragraph."""
        full_response = ""
        current_context = prompt
        
        print(f"Starting generation with model: {self.model}")
        print(f"Initial parameters: {self.current_params}")
        
        for i in range(max_paragraphs):
            print(f"\nGenerating paragraph {i+1}...")
            # Generate the next paragraph with current parameters
            paragraph = self._generate_paragraph(current_context)
            
            if not paragraph.strip():
                print("Received empty paragraph, ending generation.")
                break  # End generation if empty paragraph
            
            full_response += paragraph
            
            # Analyze the paragraph and adjust parameters for the next one
            print(f"Analyzing paragraph {i+1}...")
            analysis_results = self._analyze_paragraph(paragraph)
            self._adjust_parameters(analysis_results)
            
            # Update context for next paragraph generation
            current_context = prompt + full_response
            
            # Check if the paragraph seems to conclude the response
            if self._is_conclusion(paragraph):
                print("Detected conclusion, ending generation.")
                break
        
        return full_response
    
    def _generate_paragraph(self, context: str) -> str:
        """Generate a single paragraph with current parameters using Ollama."""
        try:
            payload = {
                "model": self.model,
                "prompt": context,
                "system": "You are continuing a response. Write the next paragraph only. Do not repeat information.",
                "stream": False,
                "options": self.current_params
            }
            
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Get just the next paragraph by finding the double newline
                generated_text = result.get("response", "")
                
                # Handle the case where Ollama might not respect the paragraph break
                paragraphs = generated_text.split("\n\n")
                if len(paragraphs) > 1:
                    return paragraphs[0] + "\n\n"
                else:
                    # If no clear paragraph break, find a good stopping point
                    sentences = re.split(r'(?<=[.!?])\s+', generated_text)
                    if len(sentences) > 3:
                        # Take first 3 sentences to keep it paragraph-sized
                        return " ".join(sentences[:3]) + "\n\n"
                    else:
                        return generated_text + "\n\n"
            else:
                print(f"Error: Ollama returned status code {response.status_code}")
                print(response.text)
                return ""
        except Exception as e:
            print(f"Error generating paragraph: {e}")
            return ""
    
    def _analyze_paragraph(self, paragraph: str) -> Dict[str, Dict]:
        """Analyze the paragraph using multiple dimensions."""
        analysis = {}
        
        for analyzer_name, analyzer_func in self.analyzers.items():
            analysis[analyzer_name] = analyzer_func(paragraph)
        
        print(f"Paragraph analysis: {analysis}")  # For debugging
        return analysis
    
    def _adjust_parameters(self, analysis: Dict[str, Dict]) -> None:
        """Adjust sampling parameters based on paragraph analysis."""
        # Start with base parameters
        new_params = self.base_params.copy()
        
        # Apply adjustments based on sentiment
        sentiment = analysis["sentiment"]
        if sentiment["label"] == "POSITIVE" and sentiment["score"] > 0.7:
            print("Adjusting for positive sentiment")
            new_params.update(self.adjustment_strategies["positive_sentiment"])
        elif sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.7:
            print("Adjusting for negative sentiment")
            new_params.update(self.adjustment_strategies["negative_sentiment"])
        
        # Apply adjustments based on complexity
        complexity = analysis["complexity"]
        if complexity["score"] > 0.7:
            print("Adjusting for high complexity")
            new_params.update(self.adjustment_strategies["high_complexity"])
        elif complexity["score"] < 0.3:
            print("Adjusting for low complexity")
            new_params.update(self.adjustment_strategies["low_complexity"])
        
        # Apply adjustments based on creativity
        creativity = analysis["creativity"]
        if creativity["score"] > 0.7:
            print("Adjusting for high creativity")
            new_params.update(self.adjustment_strategies["high_creativity"])
        elif creativity["score"] < 0.3:
            print("Adjusting for low creativity")
            new_params.update(self.adjustment_strategies["low_creativity"])
        
        # Apply adjustments based on formality
        formality = analysis["formality"]
        if formality["score"] > 0.7:
            print("Adjusting for formal tone")
            new_params.update(self.adjustment_strategies["formal"])
        elif formality["score"] < 0.3:
            print("Adjusting for informal tone")
            new_params.update(self.adjustment_strategies["informal"])
        
        # Update current parameters
        self.current_params = new_params
        print(f"Adjusted parameters: {self.current_params}")
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of the text."""
        try:
            result = self.sentiment_analyzer(text)[0]
            return result
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {"label": "NEUTRAL", "score": 0.5}
    
    def _analyze_complexity(self, text: str) -> Dict:
        """Analyze linguistic complexity of the text."""
        # Simple implementation using average word length and sentence length
        words = text.split()
        if not words:
            return {"score": 0.5, "label": "MEDIUM"}
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences if sentence) / max(1, len([s for s in sentences if s]))
        
        # Normalize scores between 0 and 1
        word_length_score = min(1.0, avg_word_length / 8.0)
        sentence_length_score = min(1.0, avg_sentence_length / 25.0)
        
        complexity_score = (word_length_score + sentence_length_score) / 2.0
        
        if complexity_score > 0.7:
            label = "HIGH"
        elif complexity_score < 0.3:
            label = "LOW"
        else:
            label = "MEDIUM"
            
        return {"score": complexity_score, "label": label}
    
    def _analyze_creativity(self, text: str) -> Dict:
        """Analyze creativity/uniqueness of the text."""
        # Simple implementation using presence of figurative language and rare words
        figurative_indicators = [
            "like", "as", "metaphor", "imagine", "resembles", "similar to",
            "comparison", "symbolizes", "represents", "analogous"
        ]
        
        uncommon_words = [
            "ephemeral", "serendipity", "mellifluous", "idiosyncratic", "quixotic",
            "ethereal", "esoteric", "quintessential", "ineffable", "obfuscate",
            "juxtapose", "paradigm", "paradox", "synthesis", "innovative"
        ]
        
        text_lower = text.lower()
        figurative_count = sum(1 for word in figurative_indicators if word in text_lower)
        uncommon_count = sum(1 for word in uncommon_words if word in text_lower)
        
        # Normalize score between
        word_count = len(text.split())
        creativity_score = min(1.0, (figurative_count + uncommon_count * 2) / max(10, word_count / 5))
        
        if creativity_score > 0.7:
            label = "HIGH"
        elif creativity_score < 0.3:
            label = "LOW"
        else:
            label = "MEDIUM"
            
        return {"score": creativity_score, "label": label}
    
    def _analyze_formality(self, text: str) -> Dict:
        """Analyze formality level of the text."""
        formal_indicators = [
            "moreover", "therefore", "consequently", "however", "thus",
            "nevertheless", "furthermore", "regarding", "concerning",
            "hereby", "herein", "aforementioned", "subsequent", "pursuant"
        ]
        
        informal_indicators = [
            "kind of", "sort of", "like", "you know", "I mean", "stuff",
            "things", "cool", "awesome", "yeah", "hey", "gonna", "wanna",
            "gotta", "dunno", "ain't", "folks", "okay", "ok", "pretty"
        ]
        
        contractions = ["'s", "'re", "'ll", "'ve", "'d", "n't"]
        
        text_lower = text.lower()
        formal_count = sum(1 for word in formal_indicators if word in text_lower)
        informal_count = sum(1 for word in informal_indicators if word in text_lower)
        contraction_count = sum(1 for contraction in contractions if contraction in text_lower)
        
        # Calculate formality score
        formality_score = (formal_count - informal_count - contraction_count / 2) / max(5, len(text.split()) / 20)
        formality_score = max(0.0, min(1.0, formality_score + 0.5))  # Normalize between 0 and 1
        
        if formality_score > 0.7:
            label = "FORMAL"
        elif formality_score < 0.3:
            label = "INFORMAL"
        else:
            label = "NEUTRAL"
            
        return {"score": formality_score, "label": label}
    
    def _is_conclusion(self, paragraph: str) -> bool:
        """Determine if the paragraph appears to be a conclusion."""
        conclusion_indicators = [
            "in conclusion", "to summarize", "in summary", "finally", 
            "in closing", "to conclude", "ultimately", "in the end",
            "altogether", "overall", "to sum up"
        ]
        
        paragraph_lower = paragraph.lower()
        
        # Check if paragraph has conclusion markers
        has_conclusion_marker = any(indicator in paragraph_lower for indicator in conclusion_indicators)
        
        # Check if paragraph is short and ends with a period (potential final statement)
        is_short_final = len(paragraph.split()) < 20 and paragraph.strip().endswith(".")
        
        return has_conclusion_marker or is_short_final


# Example usage
if __name__ == "__main__":
    # Initialize with your Ollama model of choice
    sampler = OllamaDynamicSampling(
        model="llama3.3:70b-instruct-q3_K_M",  # Change to any model you have in Ollama
        ollama_base_url="http://192.168.64.1:11434"
    )
    
    prompt = "Write a story that begins with a formal tone but shifts into increasingly emotionally manic tones of exposition."
    
    print("Generating story with dynamic sampling...")
    response = sampler.generate_text(prompt, max_paragraphs=7)
    print("\nFinal response:")
    print(response)
