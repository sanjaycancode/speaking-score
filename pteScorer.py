import whisper
import librosa
import numpy as np
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from typing import Dict, List
import re
from datetime import datetime
import json

EspeakWrapper.set_library('C:\Program Files\eSpeak NG\libespeak-ng.dll')

class PTESpeakingScorer:
    """
    A comprehensive PTE Speaking scorer that evaluates pronunciation, fluency, 
    and content accuracy using speech recognition and phonetic analysis.
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the PTE Speaking Scorer.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model(model_size)
        print("PTE Speaking Scorer initialized successfully!")
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Convert audio speech to detailed, word-by-word, timestamped transcript using Whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing transcript and word-level timestamps
        """
        print("Transcribing audio...")
        result = self.whisper_model.transcribe(
            audio_path, 
            word_timestamps=True,
            language='en'
        )
        
        # Extract word-level information
        words_with_timestamps = []
        if 'segments' in result:
            for segment in result['segments']:
                if 'words' in segment:
                    for word_info in segment['words']:
                        words_with_timestamps.append({
                            'word': word_info['word'].strip(),
                            'start': word_info['start'],
                            'end': word_info['end'],
                            'confidence': word_info.get('probability', 0.0)
                        })
        
        return {
            'full_text': result['text'],
            'words': words_with_timestamps,
            'language': result['language']
        }
    
    def convert_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text script into a sequence of phonemes for analysis.
        
        Args:
            text: Input text to convert
            
        Returns:
            List of phonemes
        """
        print("Converting text to phonemes...")
        try:
            # Clean the text
            cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
            
            # Convert to phonemes using espeak backend
            phonemes = phonemize(
                cleaned_text,
                language='en-us',
                backend='espeak',
                strip=True,
                preserve_punctuation=False,
                with_stress=True
            )
            
            # Split into individual phonemes
            phoneme_list = phonemes.split()
            return phoneme_list
            
        except Exception as e:
            print(f"Error in phoneme conversion: {e}")
            return []
    
    def get_audio_features(self, audio_path: str) -> Dict:
        """
        Extract audio features for additional analysis.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio features dictionary
        """
        print("Extracting audio features...")
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)
            
            # Extract features
            features = {
                'duration': len(audio) / sr,
                'rms_energy': float(np.mean(librosa.feature.rms(y=audio))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio))),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return {'duration': 0, 'rms_energy': 0, 'zero_crossing_rate': 0, 'spectral_centroid': 0}
    
    def calculate_pronunciation_score(self, spoken_phonemes: List[str], 
                                    reference_phonemes: List[str]) -> float:
        """
        Calculate pronunciation accuracy by comparing phonemes.
        
        Args:
            spoken_phonemes: Phonemes from spoken audio
            reference_phonemes: Phonemes from reference text
            
        Returns:
            Pronunciation score (0-100)
        """
        if not spoken_phonemes or not reference_phonemes:
            return 0.0
        
        # Simple Levenshtein distance-based scoring
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(spoken_phonemes, reference_phonemes)
        max_length = max(len(spoken_phonemes), len(reference_phonemes))
        
        if max_length == 0:
            return 100.0
        
        accuracy = (1 - distance / max_length) * 100
        return max(0, accuracy)
    
    def calculate_fluency_score(self, word_timestamps: List[Dict]) -> Dict:
        """
        Calculate fluency metrics including speech rate and pause analysis.
        
        Args:
            word_timestamps: List of words with timing information
            
        Returns:
            Dictionary with fluency metrics
        """
        if not word_timestamps:
            return {'fluency_score': 0, 'speech_rate': 0, 'pause_analysis': {}}
        
        # Calculate speech rate (words per minute)
        total_duration = word_timestamps[-1]['end'] - word_timestamps[0]['start']
        total_words = len(word_timestamps)
        speech_rate = (total_words / total_duration) * 60 if total_duration > 0 else 0
        
        # Analyze pauses
        pauses = []
        for i in range(1, len(word_timestamps)):
            pause_duration = word_timestamps[i]['start'] - word_timestamps[i-1]['end']
            if pause_duration > 0.1:  # Consider pauses > 0.1 seconds
                pauses.append(pause_duration)
        
        avg_pause = np.mean(pauses) if pauses else 0
        long_pauses = len([p for p in pauses if p > 0.5])
        
        # Calculate fluency score (0-100)
        # Ideal speech rate is around 150-180 WPM
        rate_score = min(100, (speech_rate / 165) * 100) if speech_rate <= 165 else max(0, 100 - (speech_rate - 165) * 2)
        
        # Penalty for long pauses
        pause_penalty = min(50, long_pauses * 10)
        fluency_score = max(0, rate_score - pause_penalty)
        
        return {
            'fluency_score': fluency_score,
            'speech_rate': speech_rate,
            'pause_analysis': {
                'total_pauses': len(pauses),
                'average_pause': avg_pause,
                'long_pauses': long_pauses
            }
        }
    
    def calculate_content_score(self, spoken_text: str, reference_text: str) -> float:
        """
        Calculate content accuracy score.
        
        Args:
            spoken_text: Transcribed spoken text
            reference_text: Reference text
            
        Returns:
            Content accuracy score (0-100)
        """
        spoken_words = set(spoken_text.lower().split())
        reference_words = set(reference_text.lower().split())
        
        if not reference_words:
            return 0.0
        
        intersection = spoken_words.intersection(reference_words)
        content_score = (len(intersection) / len(reference_words)) * 100
        
        return content_score
    
    def score_speaking_task(self, audio_path: str, reference_text: str = None, 
                          task_type: str = "read_aloud") -> Dict:
        """
        Complete scoring pipeline for PTE speaking tasks.
        
        Args:
            audio_path: Path to the audio file
            reference_text: Reference text (for read aloud tasks)
            task_type: Type of speaking task
            
        Returns:
            Comprehensive scoring results
        """
        print(f"Scoring {task_type} task...")
        
        # Step 1: Transcribe audio
        transcription = self.transcribe_audio(audio_path)
        spoken_text = transcription['full_text']
        word_timestamps = transcription['words']
        
        # Step 1.5: Extract audio features
        audio_features = self.get_audio_features(audio_path)
        
        results = {
            'task_type': task_type,
            'spoken_text': spoken_text,
            'timestamp': datetime.now().isoformat(),
            'audio_file': audio_path,
            'audio_features': audio_features
        }
        
        # Step 2: Calculate fluency score
        fluency_metrics = self.calculate_fluency_score(word_timestamps)
        results.update(fluency_metrics)
        
        if reference_text:
            # Step 3: Convert texts to phonemes
            spoken_phonemes = self.convert_to_phonemes(spoken_text)
            reference_phonemes = self.convert_to_phonemes(reference_text)
            
            # Step 4: Calculate pronunciation score
            pronunciation_score = self.calculate_pronunciation_score(
                spoken_phonemes, reference_phonemes
            )
            
            # Step 5: Calculate content score
            content_score = self.calculate_content_score(spoken_text, reference_text)
            
            results.update({
                'reference_text': reference_text,
                'pronunciation_score': pronunciation_score,
                'content_score': content_score,
                'spoken_phonemes': spoken_phonemes[:10],  # First 10 for preview
                'reference_phonemes': reference_phonemes[:10]
            })
        
        # Calculate overall score
        if reference_text:
            overall_score = (
                results['pronunciation_score'] * 0.4 +
                results['fluency_score'] * 0.3 +
                results['content_score'] * 0.3
            )
        else:
            overall_score = results['fluency_score']
        
        results['overall_score'] = overall_score
        
        return results
    
    def generate_feedback(self, scoring_results: Dict) -> str:
        """
        Generate detailed feedback based on scoring results.
        
        Args:
            scoring_results: Results from score_speaking_task
            
        Returns:
            Detailed feedback string
        """
        feedback = []
        feedback.append(f"=== PTE Speaking Task Analysis ===")
        feedback.append(f"Task Type: {scoring_results['task_type']}")
        feedback.append(f"Overall Score: {scoring_results['overall_score']:.1f}/100")
        feedback.append("")
        
        # Fluency feedback
        fluency_score = scoring_results['fluency_score']
        speech_rate = scoring_results['speech_rate']
        feedback.append(f"Fluency Analysis:")
        feedback.append(f"  Score: {fluency_score:.1f}/100")
        feedback.append(f"  Speech Rate: {speech_rate:.1f} words/minute")
        
        if speech_rate < 120:
            feedback.append("  → Try to speak a bit faster for better fluency")
        elif speech_rate > 200:
            feedback.append("  → Try to slow down slightly for clearer pronunciation")
        else:
            feedback.append("  → Good speech rate!")
        
        pause_info = scoring_results.get('pause_analysis', {})
        if pause_info.get('long_pauses', 0) > 3:
            feedback.append("  → Reduce long pauses for better fluency")
        
        # Pronunciation feedback (if available)
        if 'pronunciation_score' in scoring_results:
            pron_score = scoring_results['pronunciation_score']
            feedback.append(f"\nPronunciation Analysis:")
            feedback.append(f"  Score: {pron_score:.1f}/100")
            
            if pron_score < 60:
                feedback.append("  → Focus on pronunciation accuracy")
            elif pron_score < 80:
                feedback.append("  → Good pronunciation, minor improvements needed")
            else:
                feedback.append("  → Excellent pronunciation!")
        
        # Content feedback (if available)
        if 'content_score' in scoring_results:
            content_score = scoring_results['content_score']
            feedback.append(f"\nContent Analysis:")
            feedback.append(f"  Score: {content_score:.1f}/100")
            
            if content_score < 70:
                feedback.append("  → Make sure to include all key words from the text")
            else:
                feedback.append("  → Good content coverage!")
        
        return "\n".join(feedback)


# Example usage
def example_usage():
    """
    Example of how to use the PTE Speaking Scorer
    """
    # Initialize scorer
    scorer = PTESpeakingScorer(model_size="tiny")
    
    # Example for Read Aloud task
    reference_text = open("C:/NCC/PTE/assets/monologue.txt", "r").read().strip()
    audio_file = r"C:/NCC/PTE/assets/monologue.mp3"  # Replace with actual audio file
    
    try:
        # Score the speaking task
        results = scorer.score_speaking_task(
            audio_path=audio_file,
            reference_text=reference_text,
            task_type="read_aloud"
        )
        
        # Generate feedback
        feedback = scorer.generate_feedback(results)
        print(feedback)

        # Save results to JSON
        with open('scoring_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    except FileNotFoundError:
        print("Audio file not found. Please provide a valid audio file path.")
    except Exception as e:
        print(f"Error during scoring: {e}")

if __name__ == "__main__":
    example_usage()