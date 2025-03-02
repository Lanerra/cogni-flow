# Cogni-Flow
Dynamic Sampling for LLMs: A Meta-Cognitive Approach with Working Memory

## Executive Summary

This outlines a novel approach to enhancing the coherence, adaptability, and contextual awareness of Large Language Models (LLMs) through dynamic parameter adjustment. By implementing a system that combines working memory and meta-cognitive control mechanisms, we demonstrate how LLMs can adapt their generation parameters in real-time based on content analysis, effectively mimicking aspects of human cognitive flexibility. Our experimental results show marked improvements in sustained coherence, emotional progression, and information retention across extended text generation tasks.

## 1. Introduction

### 1.1 The Problem: Static Parameters in Dynamic Contexts

Large Language Models have revolutionized text generation, yet they typically operate with static sampling parameters throughout inference. This static approach constrains their ability to adapt to evolving content requirements within a single generation session. Human cognition, by contrast, constantly adjusts its "parameters" based on context, goals, and feedback.

Current approaches to LLM text generation suffer from several limitations:

- **Parameter rigidity**: Temperature, top-p, and other sampling parameters remain fixed regardless of content shifts
- **Context amnesia**: Difficulty maintaining coherence over long-form content generation
- **Emotional flatness**: Challenges in producing natural progression of tone, style, and emotional intensity
- **Task inflexibility**: Inability to smoothly transition between different cognitive modes (creative vs. analytical)

### 1.2 Our Approach: Dynamic Sampling with Meta-Cognitive Control

We present a system that addresses these limitations by:

1. Implementing a **Working Memory** component that maintains key information within a limited capacity buffer
2. Creating a **Meta-Cognitive Controller** that evaluates generation quality and goal alignment
3. Developing **Content Analyzers** that assess multiple dimensions of generated text
4. Designing **Parameter Adjustment Strategies** that modify sampling parameters based on content and goals

Our approach enables LLMs to transition smoothly between different cognitive states, maintain coherence across extended generations, and adapt to evolving narrative or informational requirements—all without requiring specialized training or model modifications.

## 2. Theoretical Framework

### 2.1 Biological Analogues in Human Cognition

Our system draws inspiration from several cognitive mechanisms in human psychology:

- **Prefrontal Inhibitory Control**: Regulates competing responses and filters inappropriate outputs
- **Working Memory**: Maintains limited but crucial information in an active, accessible state
- **Cognitive State Transitions**: Shifts between explorative and exploitative thinking modes
- **Metacognition**: Monitors and evaluates one's own thought processes

The dynamic sampling approach aims to simulate these mechanisms by adjusting LLM sampling parameters in ways that parallel human cognitive adaptations.

### 2.2 Sampling Parameters as Cognitive Controls

We map LLM sampling parameters to cognitive control mechanisms:

| LLM Parameter | Human Cognitive Analogue | Effect on Generation |
|---------------|--------------------------|----------------------|
| Temperature   | Cognitive arousal        | Controls randomness/creativity vs. deterministic responses |
| Top-p/Top-k   | Attention filtering      | Limits consideration to most relevant possibilities |
| Frequency penalty | Habituation/novelty seeking | Reduces repetition and encourages diverse vocabulary |
| Presence penalty | Lateral inhibition    | Discourages fixation on specific concepts |

By dynamically adjusting these parameters, we can simulate different cognitive states:

- **Exploratory**: Higher temperature, higher top-p, higher frequency penalty
- **Focused**: Lower temperature, lower top-p, lower frequency penalty
- **Transitional**: Balanced parameters with presence penalty to shift focus
- **Resetting**: Moderate adjustments to parameters to break out of loops

### 2.3 Working Memory as Content Integration Mechanism

Human working memory typically holds 7±2 items, providing a short-term store of information crucial for task completion. Our system implements a similar capacity-limited buffer that maintains:

- Key entities and characters
- Dominant emotions and themes
- Important concepts and relationships
- Narrative or technical milestones

This working memory implementation addresses the context amnesia problem by actively maintaining and refreshing important information across paragraph boundaries.

## 3. System Architecture

### 3.1 High-Level Overview

The dynamic sampling system comprises five main components:

1. **Text Generator**: Interface with the underlying LLM API
2. **Content Analyzers**: Evaluate multiple dimensions of generated text
3. **Working Memory**: Store and maintain key information with attention mechanisms
4. **Meta-Cognitive Controller**: Evaluate quality and alignment with goals
5. **Parameter Adjuster**: Modify sampling parameters based on system state

The system operates in a feedback loop, with each generated paragraph analyzed to inform parameter adjustments for subsequent generation.

![System Architecture](https://i.imgur.com/gKIQHvF.png)

### 3.2 Working Memory Implementation

The `WorkingMemory` class implements a capacity-limited memory store with the following key features:

- **Limited Capacity Buffer**: Maintains ~7 items for cognitive plausibility
- **Recency-Based Decay**: Older items become less accessible over time
- **Salience-Based Filtering**: More important items remain longer in memory
- **Attention Mechanisms**: Weights items based on current context relevance

Items in working memory include entities, themes, emotional states, and key concepts extracted from the generated text. The system can retrieve these items based on contextual relevance and use them to influence future generation.

### 3.3 Meta-Cognitive Controller

The `MetaCognitiveController` evaluates generation quality and goal alignment by:

- Tracking generation goals (creativity, formality, emotionality, etc.)
- Evaluating generated content against these goals
- Determining appropriate cognitive states based on evaluation
- Recommending parameter adjustments based on cognitive state

Key functions include:

- `evaluate_output()`: Assesses alignment with goals and generation quality
- `_update_cognitive_state()`: Determines which cognitive state is appropriate
- `_calculate_parameter_adjustments()`: Computes specific parameter changes

### 3.4 Content Analysis Framework

Our system analyzes generated text across multiple dimensions:

- **Sentiment**: Evaluates emotional valence (positive/negative)
- **Complexity**: Assesses linguistic sophistication through word length, sentence structure, etc.
- **Creativity**: Measures use of figurative language, unusual words, and descriptive richness
- **Formality**: Evaluates adherence to formal vs. informal language patterns
- **Tension**: Identifies narrative tension and conflict markers
- **Dialogue**: Detects and evaluates conversation presence and quality
- **Emotionality**: Assesses emotional intensity and specific emotion types

These analyses inform both working memory updates and parameter adjustments.

### 3.5 Parameter Adjustment Strategies

The system implements several parameter adjustment strategies:

- **Profile-Based**: Applies predefined parameter profiles for specific content types
- **Weighted Blending**: Combines multiple profiles based on analysis weights
- **Meta-Cognitive Overrides**: Allows direct parameter adjustments based on cognitive state
- **Narrative Progression**: Adjusts parameters based on position in narrative arc

## 4. Implementation Details

### 4.1 Core Components

```python
class WorkingMemory:
    """Simulates limited capacity working memory with attention mechanisms."""
    
    def __init__(self, capacity: int = 5):
        """Initialize working memory with specified capacity."""
        self.capacity = capacity
        self.elements = []  # Content elements (key concepts, entities, themes)
        self.importance = []  # Salience scores for each element
        self.recency = []  # Recency counters for each element (lower = more recent)
        
    def update(self, new_elements: List[Dict], max_age: int = 10):
        """Update working memory with new elements, maintaining limited capacity."""
        # Implementation details omitted for brevity
        
    def get_active_elements(self, context: Dict = None, threshold: float = 0.0) -> List[Dict]:
        """Retrieve currently active elements, optionally filtered by relevance to context."""
        # Implementation details omitted for brevity


class MetaCognitiveController:
    """
    Simulates meta-cognitive processes for dynamic parameter adjustment.
    Monitors generation quality, goal alignment, and adjusts parameters accordingly.
    """
    
    def __init__(self):
        """Initialize the meta-cognitive controller."""
        # Generation goals and tracking
        self.generation_goals = {}
        self.generation_history = []
        
        # Current progress evaluation
        self.goal_alignment = 1.0  # How well we're progressing toward goals (0-1)
        self.content_quality = 1.0  # Perceived quality of generated content (0-1)
        self.cognitive_state = "neutral"  # Current cognitive state (explore, focus, etc.)
        
        # Parameter adjustment strategies
        self.adjustment_strategies = {
            "explore": {"temperature": 0.3, "top_p": 0.1, "top_k": 20, "frequency_penalty": 0.1},
            "focus": {"temperature": -0.3, "top_p": -0.1, "top_k": -10, "frequency_penalty": -0.1},
            "shift": {"temperature": 0.2, "presence_penalty": 0.1},
            "reset": {"temperature": 0.1, "top_p": 0.05, "presence_penalty": 0.1},
        }
        
    def evaluate_output(self, output: Dict, analysis: Dict, target_state: Dict) -> Dict:
        """Evaluate generated output against goals and current cognitive state."""
        # Implementation details omitted for brevity


class EnhancedMetaCognitiveSystem:
    """
    Combined system with working memory and meta-cognitive control
    for dynamic sampling parameter adjustment.
    """
    
    def __init__(self, model: str = "llama3", ollama_base_url: str = "http://localhost:11434",
                sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize the enhanced meta-cognitive system."""
        # Implementation details omitted for brevity
        
    def generate_text(self, prompt: str, narrative_style: str = "standard", max_paragraphs: int = 5) -> str:
        """Generate text with dynamically adjusted sampling parameters by paragraph."""
        # Implementation details omitted for brevity
```

### 4.2 Content Analysis Algorithms

The system employs several sophisticated analysis algorithms to evaluate text across multiple dimensions:

#### 4.2.1 Complexity Analysis

```python
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
```

#### 4.2.2 Creativity Analysis

```python
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
        # Additional common words omitted for brevity
    ])
    
    # Count figurative language
    figurative_count = sum(1 for marker in figurative_markers if marker in text_lower)
    
    # Check for imagery and sensory language
    sensory_words = [
        "saw", "see", "seen", "look", "gaze", "glimpse", "glance",
        "hear", "sound", "noise", "listen", "melody", "whisper", "scream",
        # Additional sensory words omitted for brevity
    ]
    sensory_count = sum(1 for word in sensory_words if word in text_lower)
    
    # Check for rich descriptors (adjectives and adverbs)
    rich_descriptors = [
        "ly ", "beautiful", "gorgeous", "stunning", "magnificent", "elegant",
        # Additional descriptors omitted for brevity
    ]
    descriptor_count = sum(1 for desc in rich_descriptors if desc in text_lower)
    
    # Check for unusual words (not in common set)
    word_list = [w.lower() for w in words if len(w) > 3]  # Only consider words > 3 chars
    unusual_word_ratio = sum(1 for word in word_list if word not in common_words) / max(1, len(word_list))
    
    # Calculate creativity score with weighted components
    # Implementation details omitted for brevity
    
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
```

### 4.3 Parameter Adjustment Logic

```python
def _adjust_parameters_advanced(self, analysis: Dict[str, Dict]) -> None:
    """
    Adjust sampling parameters based on paragraph analysis using a weighted approach.
    This creates more nuanced parameter adjustments based on multiple factors.
    """
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
```

## 5. Experimental Results

### 5.1 Narrative Generation

We tested the system's ability to generate coherent, emotionally progressing narratives across multiple models and narrative styles. Key findings included:

- **Extended coherence**: Successfully maintained character and plot consistency across 24+ paragraphs
- **Emotional progression**: Demonstrated gradual emotional intensity increases in "building_tension" narrative style
- **Parameter evolution**: Showed appropriate adjustment of parameters based on content analysis
- **Working memory effectiveness**: Successfully maintained and referenced key entities throughout generation

#### Sample Parameter Evolution - Narrative Task (70B Model)

| Paragraph | Temperature | Top-p | Frequency Penalty | Narrative Phase | Dominant Profile |
|-----------|-------------|-------|------------------|----------------|-----------------|
| 1         | 0.70        | 0.90  | 0.00             | Introduction   | positive        |
| 3         | 0.62        | 0.85  | 0.12             | Development    | informal        |
| 5         | 0.61        | 0.85  | 0.22             | Development    | creative        |
| 7         | 0.59        | 0.85  | 0.21             | Climax         | neutral_emotion |
| 8         | 0.60        | 0.85  | 0.18             | Resolution     | high_complexity |

### 5.2 Technical Explanation

The system demonstrated strong capabilities in generating technical explanations that:

- Started with factual, formal language for core concepts
- Gradually transitioned to more creative analogies and applications
- Maintained high complexity throughout while adjusting creativity
- Successfully tracked key concepts in working memory for reference

#### Sample Parameter Evolution - Technical Task (14B Model)

| Paragraph | Temperature | Top-p | Frequency Penalty | Content Type | Dominant Profile |
|-----------|-------------|-------|------------------|--------------|-----------------|
| 1         | 0.36        | 0.73  | 0.03             | Factual      | high_complexity |
| 6         | 0.60        | 0.85  | 0.00             | Factual      | neutral_emotion |
| 11        | 0.60        | 0.85  | 0.11             | Informative  | informal        |
| 18        | 0.59        | 0.85  | 0.09             | Creative     | high_complexity |
| 24        | 0.61        | 0.85  | 0.14             | Concluding   | informal        |

### 5.3 Meta-Cognitive State Analysis

The meta-cognitive controller primarily remained in "neutral" state across generations, indicating that parameter adjustments through profile application were sufficient to maintain generation quality. Key observations:

- Goal alignment generally decreased over time (0.91→0.64), indicating drift from initial targets
- Content quality assessments remained relatively stable (0.65-0.80)
- First cognitive state transition to "shift" typically occurred only near conclusion
- Repetition was observed in only a few cases, suggesting the need for detection mechanisms

## 6. Limitations and Future Work

### 6.1 Current Limitations

1. **Conservative State Transitions**: The meta-cognitive controller rarely shifts from neutral state
2. **Limited Emotional Range**: Tension and emotional intensity detection need refinement
3. **Parameter Sensitivity**: Some LLM models respond more dramatically to parameter changes than others
4. **Repetition Handling**: No specific mechanism to detect and address repetitive content
5. **Computational Overhead**: Content analysis adds latency between generation steps

### 6.2 Future Improvements

1. **Neural Parameter Controller**: Train a model to directly predict optimal parameters based on content
2. **Enhanced Emotion Detection**: Implement more sophisticated emotion recognition
3. **Adaptive Memory Mechanisms**: Develop more nuanced working memory with tiered storage
4. **Cross-Model Calibration**: Create model-specific parameter profiles for consistent effects
5. **Repetition Detection**: Add mechanisms to identify and address redundant content
6. **Latency Optimization**: Batch analyzer operations to reduce generation pauses

## 7. Conclusion

### 7.1 Key Contributions

1. **Dynamic Sampling Framework**: A novel approach to real-time LLM parameter adjustment
2. **Working Memory Integration**: An effective solution to the context amnesia problem
3. **Meta-Cognitive Control System**: A mechanism for evaluating and adjusting generation strategy
4. **Content Analysis Suite**: Multi-dimensional text analysis for informed parameter adjustment
5. **Self-Annotating Dataset**: Generation logs provide valuable training data for future enhancements

### 7.2 Practical Applications

1. **Long-form Content Generation**: Stories, articles, reports with coherent progression
2. **Educational Content**: Technical explanations that build concepts progressively
3. **Creative Writing**: Narratives with emotional arcs and character consistency
4. **Conversational AI**: More natural dialogue progression with emotional awareness
5. **Technical Documentation**: Complex explanations that transition from concepts to applications

The dynamic sampling approach appears to represent a significant advancement in LLM text generation, addressing key limitations of static parameter settings while mimicking aspects of human cognitive flexibility. By enabling models to adapt their "cognitive state" based on content and goals, this system paves the way for more natural, coherent, and contextually appropriate text generation.
