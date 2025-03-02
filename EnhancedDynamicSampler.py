"""
Enhanced Dynamic Sampling for LLMs
----------------------------------
This improved implementation provides more nuanced and effective dynamic sampling
with broader parameter ranges, better content analysis, and a weighted approach
to parameter selection.
"""

import re
import time
import requests
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from transformers import pipeline
from collections import Counter

class EnhancedDynamicSampling:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        ollama_base_url: str = "http://localhost:11434",
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
    ):
        """Initialize with Ollama endpoint and sentiment analyzer."""
        self.model = model
        self.ollama_base_url = ollama_base_url
        self.api_url = f"{ollama_base_url}/api/generate"
        
        # Starting parameters - we'll keep track of the narrative progression
        self.current_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        
        # For tracking progression through the narrative
        self.paragraph_count = 0
        self.narrative_phase = "introduction"  # introduction, development, climax, resolution
        self.narrative_tension = 0.2  # 0.0 to 1.0
        self.emotional_intensity = 0.2  # 0.0 to 1.0
        
        # Load sentiment analysis model
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
        
        # Define content analyzers with improved detection algorithms
        self.analyzers = {
            "sentiment": self._analyze_sentiment,
            "complexity": self._analyze_complexity,
            "creativity": self._analyze_creativity,
            "formality": self._analyze_formality,
            "tension": self._analyze_narrative_tension,
            "dialogue": self._analyze_dialogue_content,
            "emotionality": self._analyze_emotional_intensity
        }
        
        # Parameter adjustment profiles - expanded with wider ranges
        self.parameter_profiles = {
            # Narrative phases
            "introduction": {"temperature": 0.6, "top_p": 0.9, "frequency_penalty": 0.0},
            "development": {"temperature": 0.7, "top_p": 0.9, "frequency_penalty": 0.1},
            "climax": {"temperature": 0.85, "top_p": 0.95, "frequency_penalty": 0.2},
            "resolution": {"temperature": 0.65, "top_p": 0.85, "frequency_penalty": 0.1},
            
            # Content types
            "factual": {"temperature": 0.3, "top_p": 0.7, "top_k": 20, "frequency_penalty": 0.0},
            "creative": {"temperature": 1.1, "top_p": 0.95, "top_k": 60, "frequency_penalty": 0.3, "presence_penalty": 0.1},
            "dialogue_heavy": {"temperature": 0.8, "top_p": 0.9, "frequency_penalty": 0.2, "presence_penalty": 0.1},
            "descriptive": {"temperature": 0.7, "top_p": 0.8, "top_k": 30, "frequency_penalty": 0.1},
            
            # Emotional tones
            "highly_emotional": {"temperature": 0.9, "top_p": 0.92, "presence_penalty": 0.1},
            "neutral_emotion": {"temperature": 0.6, "top_p": 0.85, "presence_penalty": 0.0},
            
            # Sentiment
            "positive": {"temperature": 0.75, "top_p": 0.9},
            "negative": {"temperature": 0.65, "top_p": 0.85},
            
            # Complexity
            "high_complexity": {"temperature": 0.5, "top_p": 0.8, "top_k": 30},
            "low_complexity": {"temperature": 0.85, "top_p": 0.92, "top_k": 50},
            
            # Formality
            "formal": {"temperature": 0.5, "top_p": 0.8, "frequency_penalty": 0.0},
            "informal": {"temperature": 0.9, "top_p": 0.92, "frequency_penalty": 0.15, "presence_penalty": 0.05},
        }
        
        # Verify model availability
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
    
    def generate_text(self, prompt: str, narrative_style: str = "standard", max_paragraphs: int = 5) -> str:
        """
        Generate text with dynamically adjusted sampling parameters by paragraph.
        
        narrative_style: One of "standard", "building_tension", "emotional_rollercoaster", "factual_to_creative"
        """
        full_response = ""
        current_context = prompt
        
        # Set initial narrative trajectory based on style
        if narrative_style == "building_tension":
            self.narrative_tension = 0.1
            self.emotional_intensity = 0.2
            tension_increment = 0.15
            emotion_increment = 0.1
        elif narrative_style == "emotional_rollercoaster":
            self.narrative_tension = 0.3
            self.emotional_intensity = 0.2
            tension_increment = 0.05
            emotion_increment = 0.2
        elif narrative_style == "factual_to_creative":
            self.narrative_tension = 0.1
            self.emotional_intensity = 0.1
            tension_increment = 0.05
            emotion_increment = 0.15
            # Start more factual
            self._apply_profile_weighted("factual", 0.8)
        else:  # standard
            self.narrative_tension = 0.2
            self.emotional_intensity = 0.2
            tension_increment = 0.1
            emotion_increment = 0.1
        
        print(f"Starting generation with model: {self.model}")
        print(f"Narrative style: {narrative_style}")
        print(f"Initial parameters: {self.current_params}")
        
        for i in range(max_paragraphs):
            self.paragraph_count = i + 1
            print(f"\nGenerating paragraph {self.paragraph_count}...")
            
            # Advance narrative progression
            if self.paragraph_count < max_paragraphs * 0.25:
                self.narrative_phase = "introduction"
            elif self.paragraph_count < max_paragraphs * 0.6:
                self.narrative_phase = "development"
            elif self.paragraph_count < max_paragraphs * 0.8:
                self.narrative_phase = "climax"
            else:
                self.narrative_phase = "resolution"
                
            # Apply narrative phase profile
            self._apply_profile_weighted(self.narrative_phase, 0.3)
            
            # Increase tension and emotion based on narrative style
            self.narrative_tension = min(1.0, self.narrative_tension + tension_increment)
            self.emotional_intensity = min(1.0, self.emotional_intensity + emotion_increment)
            
            # Generate the next paragraph with current parameters
            paragraph = self._generate_paragraph(current_context)
            
            if not paragraph.strip():
                print("Received empty paragraph, ending generation.")
                break
            
            full_response += paragraph
            
            # Analyze the paragraph and adjust parameters for the next one
            print(f"Analyzing paragraph {self.paragraph_count}...")
            analysis_results = self._analyze_paragraph(paragraph)
            self._adjust_parameters_advanced(analysis_results)
            
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
                "system": f"You are continuing a response in the {self.narrative_phase} phase. Write the next paragraph only with appropriate {self.emotional_intensity:.1f} emotional intensity. Do not repeat information.",
                "stream": False,
                "options": self.current_params
            }
            
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                # Handle paragraph breaks properly
                paragraphs = generated_text.split("\n\n")
                if len(paragraphs) > 1:
                    return paragraphs[0] + "\n\n"
                else:
                    # If no clear paragraph break, find a good stopping point
                    sentences = re.split(r'(?<=[.!?])\s+', generated_text)
                    if len(sentences) > 3:
                        # Take first 3-5 sentences to keep it paragraph-sized
                        return " ".join(sentences[:min(5, len(sentences))]) + "\n\n"
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
        
        print(f"Paragraph analysis: {analysis}")
        return analysis
    
    def _adjust_parameters_advanced(self, analysis: Dict[str, Dict]) -> None:
        """
        Adjust sampling parameters based on paragraph analysis using a weighted approach.
        This creates more nuanced parameter adjustments based on multiple factors.
        """
        # Start fresh with default parameters
        new_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        
        # Track which profiles we're applying and their weights
        applied_profiles = []
        
        # Apply sentiment-based profile
        sentiment = analysis["sentiment"]
        if sentiment["label"] == "POSITIVE" and sentiment["score"] > 0.7:
            applied_profiles.append(("positive", min(sentiment["score"], 0.9)))
        elif sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.7:
            applied_profiles.append(("negative", min(sentiment["score"], 0.9)))
            
        # Apply complexity-based profile
        complexity = analysis["complexity"]
        if complexity["score"] > 0.7:
            applied_profiles.append(("high_complexity", min(complexity["score"], 0.8)))
        elif complexity["score"] < 0.4:
            applied_profiles.append(("low_complexity", 1 - complexity["score"]))
            
        # Apply creativity-based profile
        creativity = analysis["creativity"]
        if creativity["score"] > 0.5:  # Lowered threshold from previous implementation
            applied_profiles.append(("creative", creativity["score"]))
        elif creativity["score"] < 0.3:
            # For very uncreative text, apply some factual profile aspects
            applied_profiles.append(("factual", 0.7))
            
        # Apply formality-based profile
        formality = analysis["formality"]
        if formality["score"] > 0.7:
            applied_profiles.append(("formal", formality["score"]))
        elif formality["score"] < 0.3:
            applied_profiles.append(("informal", 1 - formality["score"]))
            
        # Apply dialogue detection
        dialogue = analysis["dialogue"]
        if dialogue["score"] > 0.5:
            applied_profiles.append(("dialogue_heavy", dialogue["score"]))
            
        # Apply emotionality profile
        emotionality = analysis["emotionality"]
        if emotionality["score"] > 0.6:
            applied_profiles.append(("highly_emotional", emotionality["score"]))
        elif emotionality["score"] < 0.3:
            applied_profiles.append(("neutral_emotion", 1 - emotionality["score"]))
            
        # Apply narrative tension profile - increases as story progresses
        tension = analysis["tension"]
        self.narrative_tension = max(self.narrative_tension, tension["score"])
        if self.narrative_tension > 0.7:
            # High tension correlates with climax
            applied_profiles.append(("climax", self.narrative_tension))
        
        # Apply all profiles with weighting
        for profile_name, weight in applied_profiles:
            self._apply_profile_weighted(profile_name, weight)
            
        print(f"Applied profiles: {applied_profiles}")
        print(f"Adjusted parameters: {self.current_params}")
    
    def _apply_profile_weighted(self, profile_name: str, weight: float) -> None:
        """Apply a parameter profile with weighting for smoother transitions."""
        if profile_name in self.parameter_profiles:
            profile = self.parameter_profiles[profile_name]
            print(f"Applying {profile_name} profile with weight {weight}")
            for param, value in profile.items():
                if param in self.current_params:
                    # Weighted blend of current and profile values
                    current = self.current_params[param]
                    self.current_params[param] = current * (1 - weight) + value * weight
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of the text."""
        try:
            result = self.sentiment_analyzer(text)[0]
            return result
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {"label": "NEUTRAL", "score": 0.5}
    
    def _analyze_complexity(self, text: str) -> Dict:
        """
        Analyze linguistic complexity with improved metrics:
        - Average word length
        - Sentence length variation
        - Vocabulary richness
        - Structural complexity (commas, semicolons, etc.)
        """
        words = text.split()
        if not words:
            return {"score": 0.5, "label": "MEDIUM"}
        
        # Word length and variation
        word_lengths = [len(word) for word in words]
        avg_word_length = sum(word_lengths) / len(words)
        word_length_std = np.std(word_lengths) if len(words) > 1 else 0
        
        # Sentence analysis
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return {"score": 0.5, "label": "MEDIUM"}
            
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = sum(sentence_lengths) / len(sentences) if sentences else 0
        sentence_length_std = np.std(sentence_lengths) if len(sentences) > 1 else 0
        
        # Vocabulary richness (unique words ratio)
        unique_words = len(set([w.lower() for w in words]))
        vocabulary_richness = unique_words / len(words)
        
        # Structural complexity
        structural_markers = re.findall(r'[,;:\(\)\[\]\{\}]', text)
        structural_complexity = len(structural_markers) / len(words) if words else 0
        
        # Calculate complexity score (weighted factors)
        word_length_score = min(1.0, avg_word_length / 8.0) * 0.25
        sentence_length_score = min(1.0, avg_sentence_length / 25.0) * 0.25
        variation_score = min(1.0, (word_length_std + sentence_length_std) / 10.0) * 0.2
        vocabulary_score = vocabulary_richness * 0.15
        structural_score = min(1.0, structural_complexity * 5) * 0.15
        
        complexity_score = word_length_score + sentence_length_score + variation_score + vocabulary_score + structural_score
        complexity_score = min(1.0, complexity_score)
        
        if complexity_score > 0.7:
            label = "HIGH"
        elif complexity_score < 0.4:
            label = "LOW"
        else:
            label = "MEDIUM"
            
        return {"score": complexity_score, "label": label, "details": {
            "avg_word_length": avg_word_length,
            "vocabulary_richness": vocabulary_richness,
            "avg_sentence_length": avg_sentence_length
        }}
    
    def _analyze_creativity(self, text: str) -> Dict:
        """
        Enhanced creativity analysis looking at:
        - Figurative language
        - Uncommon words
        - Imagery
        - Novelty of phrasing
        - Sentence structure variation
        """
        text_lower = text.lower()
        words = text.split()
        if not words:
            return {"score": 0.5, "label": "MEDIUM"}
        
        # Figurative language markers
        figurative_markers = [
            "like a", "as if", "as though", "metaphor", "symbol", "resemble", 
            "similar to", "comparison", "analogy", "allegory", "personif",
            "imagin", "seems to", "appeared to"
        ]
        
        # Vocabulary unusualness
        common_words = set([
            "the", "a", "an", "and", "but", "or", "for", "nor", "yet", "so",
            "in", "on", "at", "to", "with", "by", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below",
            "from", "up", "down", "of", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "that", "then",
            "these", "this", "those", "too", "very", "can", "will", "just",
            "should", "now", "also", "however", "thus", "indeed"
        ])
        
        # Count figurative language
        figurative_count = sum(1 for marker in figurative_markers if marker in text_lower)
        
        # Check for imagery and sensory language
        sensory_words = [
            "saw", "see", "seen", "look", "gaze", "glimpse", "glance",
            "hear", "sound", "noise", "listen", "melody", "whisper", "scream",
            "touch", "feel", "felt", "smooth", "rough", "texture", "hard", "soft",
            "smell", "scent", "aroma", "fragrance", "odor", "perfume",
            "taste", "flavor", "sweet", "sour", "bitter", "spicy", "delicious"
        ]
        sensory_count = sum(1 for word in sensory_words if word in text_lower)
        
        # Check for rich descriptors (adjectives and adverbs)
        rich_descriptors = [
            "ly ", "beautiful", "gorgeous", "stunning", "magnificent", "elegant",
            "graceful", "delicate", "vibrant", "vivid", "lush", "spectacular",
            "breathtaking", "extraordinary", "remarkable", "impressive", "striking",
            "dazzling", "glittering", "shimmering", "radiant", "gleaming", "glowing"
        ]
        descriptor_count = sum(1 for desc in rich_descriptors if desc in text_lower)
        
        # Check for unusual words (not in common set)
        word_list = [w.lower() for w in words if len(w) > 3]  # Only consider words > 3 chars
        unusual_word_ratio = sum(1 for word in word_list if word not in common_words) / max(1, len(word_list))
        
        # Check for exclamation, question marks, and unusual punctuation
        unusual_punct_count = len(re.findall(r'[!?…]', text))
        
        # Check for all caps words (intensity)
        all_caps_count = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        # Sentence structure variation
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        first_words = [s.split()[0].lower() if s.split() else "" for s in sentences]
        first_word_variety = len(set(first_words)) / max(1, len(first_words))
        
        # Calculate creativity score with weighted components
        figurative_score = min(1.0, figurative_count / max(5, len(words) / 20)) * 0.25
        sensory_score = min(1.0, sensory_count / max(5, len(words) / 30)) * 0.2
        descriptor_score = min(1.0, descriptor_count / max(5, len(words) / 25)) * 0.15
        unusual_word_score = unusual_word_ratio * 0.2
        punctuation_score = min(1.0, unusual_punct_count / max(3, len(sentences))) * 0.1
        caps_score = min(1.0, all_caps_count * 0.5) * 0.05
        variety_score = first_word_variety * 0.05
        
        creativity_score = (
            figurative_score + 
            sensory_score + 
            descriptor_score + 
            unusual_word_score + 
            punctuation_score + 
            caps_score + 
            variety_score
        )
        
        if creativity_score > 0.6:  # Lower threshold from previous version
            label = "HIGH"
        elif creativity_score < 0.3:
            label = "LOW"
        else:
            label = "MEDIUM"
            
        return {
            "score": creativity_score, 
            "label": label,
            "details": {
                "figurative": figurative_count,
                "sensory": sensory_count,
                "unusual_words": unusual_word_ratio,
                "descriptors": descriptor_count
            }
        }
    
    def _analyze_formality(self, text: str) -> Dict:
        """Enhanced formality analysis."""
        text_lower = text.lower()
        
        formal_indicators = [
            "moreover", "therefore", "consequently", "however", "thus",
            "nevertheless", "furthermore", "regarding", "concerning",
            "hereby", "herein", "aforementioned", "subsequent", "pursuant",
            "accordingly", "hence", "wherein", "whereby", "notwithstanding",
            "albeit", "whereas", "hitherto", "thence", "therein", "thereafter"
        ]
        
        informal_indicators = [
            "kind of", "sort of", "like", "you know", "I mean", "stuff",
            "things", "cool", "awesome", "yeah", "hey", "gonna", "wanna",
            "gotta", "dunno", "ain't", "folks", "okay", "ok", "pretty",
            "super", "totally", "basically", "actually", "literally", "just",
            "whatever", "anyways", "btw", "lol", "omg", "wow", "yep", "nope"
        ]
        
        contractions = ["'s", "'re", "'ll", "'ve", "'d", "n't"]
        
        # Count indicators
        formal_count = sum(1 for word in formal_indicators if word in text_lower)
        informal_count = sum(1 for word in informal_indicators if word in text_lower)
        contraction_count = sum(1 for contraction in contractions if contraction in text_lower)
        
        # Check for first/second person pronouns (informal)
        first_second_person = len(re.findall(r'\b(I|me|my|mine|myself|we|us|our|ours|ourselves|you|your|yours|yourself|yourselves)\b', text_lower))
        
        # Check for exclamation marks (informal)
        exclamations = text.count('!')
        
        # Word length (longer = more formal)
        words = text_lower.split()
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        
        # Calculate formality score
        formality_base = (formal_count * 2 - informal_count - contraction_count / 2) / max(5, len(words) / 20)
        pronoun_penalty = -0.1 * (first_second_person / max(5, len(words) / 10))
        exclamation_penalty = -0.1 * (exclamations / max(2, len(words) / 50))
        word_length_bonus = 0.1 * (avg_word_length / 6)  # Bonus for longer words
        
        formality_score = 0.5 + formality_base + pronoun_penalty + exclamation_penalty + word_length_bonus
        formality_score = max(0.0, min(1.0, formality_score))  # Normalize between 0 and 1
        
        if formality_score > 0.7:
            label = "FORMAL"
        elif formality_score < 0.3:
            label = "INFORMAL"
        else:
            label = "NEUTRAL"
            
        return {"score": formality_score, "label": label}
    
    def _analyze_narrative_tension(self, text: str) -> Dict:
        """Analyze narrative tension and conflict in the text."""
        text_lower = text.lower()
        
        # Tension indicators
        tension_markers = [
            "sudden", "abrupt", "unexpected", "surprised", "shocked", "startled",
            "fear", "afraid", "terrified", "dread", "horror", "panic", "alarm",
            "urgent", "desperate", "frantic", "anxious", "worried", "concerned",
            "danger", "threat", "risk", "peril", "hazard", "emergency", "crisis",
            "conflict", "clash", "battle", "fight", "struggle", "confrontation",
            "challenge", "obstacle", "problem", "difficulty", "trouble", "dilemma"
        ]
        
        # Intensity markers
        intensity_markers = [
            "extremely", "intensely", "incredibly", "overwhelmingly", "absolutely",
            "completely", "totally", "utterly", "entirely", "wholly", "fully",
            "very", "really", "truly", "highly", "exceedingly", "exceptionally",
            "remarkably", "notably", "significantly", "substantially", "considerably"
        ]
        
        # Count tension indicators
        tension_count = sum(1 for word in tension_markers if word in text_lower)
        intensity_count = sum(1 for word in intensity_markers if word in text_lower)
        
        # Check for exclamation marks, question marks, and ellipses
        exclamations = text.count('!')
        questions = text.count('?')
        ellipses = text.count('...')
        
        # Check for short sentences (can indicate tension)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        short_sentences = sum(1 for s in sentences if len(s.split()) < 8)
        
        # Count ALL CAPS words
        all_caps = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        # Calculate tension score
        words = text_lower.split()
        word_count = max(1, len(words))
        
        tension_base = tension_count / max(5, word_count / 15)
        intensity_modifier = intensity_count / max(5, word_count / 25)
        punctuation_factor = (exclamations + questions + ellipses * 0.5) / max(2, len(sentences))
        short_sentence_factor = short_sentences / max(1, len(sentences))
        caps_factor = all_caps / max(1, word_count / 50)
        
        tension_score = (
            tension_base * 0.4 +
            intensity_modifier * 0.2 +
            punctuation_factor * 0.15 +
            short_sentence_factor * 0.15 +
            caps_factor * 0.1
        )
        
        tension_score = max(0.0, min(1.0, tension_score))
        
        if tension_score > 0.7:
            label = "HIGH_TENSION"
        elif tension_score < 0.3:
            label = "LOW_TENSION"
        else:
            label = "MODERATE_TENSION"
            
        # Update overall narrative tension tracker
        self.narrative_tension = max(self.narrative_tension, tension_score)
            
        return {"score": tension_score, "label": label}
    
    def _analyze_dialogue_content(self, text: str) -> Dict:
        """Analyze if the content contains dialogue."""
        # Count quotation marks
        quote_count = text.count('"') + text.count("'")
        
        # Look for dialogue tags
        dialogue_tags = re.findall(r'(said|asked|replied|whispered|shouted|exclaimed|muttered|responded)', text.lower())
        
        # Check for turn-taking indicators
        turns = max(1, len(re.split(r'"[^"]*"', text)) - 1)
        
        # Calculate dialogue score
        has_quotes = quote_count >= 4  # At least two pairs of quotes
        has_tags = len(dialogue_tags) > 0
        has_turns = turns > 1
        
        if has_quotes and (has_tags or has_turns):
            base_score = 0.7
        elif has_quotes:
            base_score = 0.5
        elif has_tags:
            base_score = 0.3
        else:
            base_score = 0.0
            
        # Adjust based on dialogue density
        words = text.split()
        quote_density = quote_count / max(20, len(words))
        tag_density = len(dialogue_tags) / max(20, len(words))
        
        dialogue_score = base_score + quote_density * 0.5 + tag_density * 0.3
        dialogue_score = max(0.0, min(1.0, dialogue_score))
        
        if dialogue_score > 0.6:
            label = "DIALOGUE_HEAVY"
        elif dialogue_score > 0.3:
            label = "SOME_DIALOGUE"
        else:
            label = "MINIMAL_DIALOGUE"
            
        return {"score": dialogue_score, "label": label}
    
    def _analyze_emotional_intensity(self, text: str) -> Dict:
        """Analyze the emotional intensity of the text."""
        text_lower = text.lower()
        
        # Emotion words lists
        emotion_words = {
            "joy": ["happy", "joy", "delight", "thrill", "ecstatic", "elated", "overjoyed", "bliss", "jubilant"],
            "sadness": ["sad", "sorrow", "grief", "misery", "despair", "heartbroken", "dejected", "depressed", "melancholy"],
            "anger": ["angry", "fury", "rage", "outrage", "wrath", "indignation", "irritated", "enraged", "hostile"],
            "fear": ["afraid", "fear", "terror", "dread", "horror", "panic", "anxiety", "frightened", "terrified"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned", "startled", "dumbfounded", "bewildered"],
            "disgust": ["disgust", "revulsion", "repugnance", "abhorrence", "loathing", "nauseated", "repelled"],
            "anticipation": ["anticipation", "expectation", "eagerness", "excitement", "suspense", "waiting", "looking forward"],
            "trust": ["trust", "confidence", "faith", "belief", "reliance", "assurance", "conviction", "dependence"]
        }
        
        # Count emotion words
        emotion_counts = {}
        total_emotion_words = 0
        
        for emotion, word_list in emotion_words.items():
            count = sum(1 for word in word_list if word in text_lower)
            emotion_counts[emotion] = count
            total_emotion_words += count
        
        # Check for intensifiers
        intensifiers = [
            "very", "extremely", "incredibly", "absolutely", "completely", 
            "totally", "utterly", "entirely", "deeply", "profoundly",
            "intensely", "overwhelmingly", "tremendously", "exceedingly"
        ]
        intensifier_count = sum(1 for word in intensifiers if word in text_lower)
        
        # Check for punctuation indicators
        exclamations = text.count('!')
        all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
        
        # Calculate emotion intensity
        words = text.split()
        word_count = max(1, len(words))
        
        # Base emotional intensity from emotion word density
        emotion_density = total_emotion_words / max(10, word_count / 5)
        # Intensifier contribution
        intensifier_factor = intensifier_count / max(5, word_count / 10)
        # Punctuation contribution
        punctuation_factor = (exclamations + all_caps_words) / max(2, word_count / 20)
        
        # Calculate dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
        
        # Overall emotional intensity
        emotional_intensity = min(1.0, emotion_density * 0.6 + intensifier_factor * 0.25 + punctuation_factor * 0.15)
        
        # Update overall emotional intensity tracker
        self.emotional_intensity = max(self.emotional_intensity, emotional_intensity)
        
        if emotional_intensity > 0.7:
            label = "HIGH_EMOTIONAL"
        elif emotional_intensity < 0.3:
            label = "LOW_EMOTIONAL"
        else:
            label = "MODERATE_EMOTIONAL"
            
        return {
            "score": emotional_intensity, 
            "label": label, 
            "dominant_emotion": dominant_emotion,
            "emotions": emotion_counts
        }
    
    def _is_conclusion(self, paragraph: str) -> bool:
        """Determine if the paragraph appears to be a conclusion."""
        conclusion_indicators = [
            "in conclusion", "to summarize", "in summary", "finally", 
            "in closing", "to conclude", "ultimately", "in the end",
            "altogether", "overall", "to sum up", "in essence",
            "to conclude", "wrapping up", "in retrospect"
        ]
        
        paragraph_lower = paragraph.lower()
        
        # Check if paragraph has conclusion markers
        has_conclusion_marker = any(indicator in paragraph_lower for indicator in conclusion_indicators)
        
        # Check if paragraph is short and ends with a period (potential final statement)
        is_short_final = len(paragraph.split()) < 20 and paragraph.strip().endswith(".")
        
        # Check if we're in the resolution phase and the paragraph feels concluding
        in_resolution_phase = self.narrative_phase == "resolution"
        has_resolution_feel = in_resolution_phase and (
            paragraph.strip().endswith(".") or 
            paragraph.strip().endswith("!") or 
            paragraph.strip().endswith("?")
        )
        
        return has_conclusion_marker or (is_short_final and in_resolution_phase) or has_resolution_feel


# Example usage
if __name__ == "__main__":
    sampler = EnhancedDynamicSampling(
        model="qwen2.5:7b",  # Change to your available model
        ollama_base_url="http://localhost:11434"
    )
    
    prompt = "Write a story that begins with a formal tone but shifts into increasingly emotionally manic tones of exposition."
    
    print("Generating story with dynamic sampling...")
    response = sampler.generate_text(
        prompt, 
        narrative_style="building_tension",  # Try different styles: "standard", "building_tension", "emotional_rollercoaster", "factual_to_creative"
        max_paragraphs=7
    )
    print("\nFinal response:")
    print(response)
