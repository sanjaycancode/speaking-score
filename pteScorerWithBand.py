import whisper, json, re, warnings, difflib
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from typing import Dict, List, Tuple
from urllib.parse import urlparse
import requests
import tempfile
import os

warnings.filterwarnings("ignore")
EspeakWrapper.set_library(r'C:\Program Files\eSpeak NG\libespeak-ng.dll')


SUGGESTIONS = {
    "content": {
        5: "Perfect coverage - keep this accuracy!",
        4: "Minor omissions - double-check plurals & articles",
        3: "Missed a few key words - slow down on long sentences",
        2: "2/3 correct - practice shadow-reading daily",
        1: "Over half the words lost - read aloud daily with transcript",
        0: "Content missing - re-record or re-read the prompt"
    },
    "fluency": {
        5: "Smooth rhythm - no action needed",
        4: "One tiny hesitation - rehearse tricky phrases",
        3: "Uneven speed - practice 150-170 WPM pacing",
        2: "Several hesitations - record & self-correct every 2 days",
        1: "Frequent pauses - chunk text into 3-word groups",
        0: "Choppy delivery - slow, shadow-read short sentences"
    },
    "pronunciation": {
        5: "Native-like - keep practising stress patterns",
        4: "Minor distortions - isolate problem phonemes",
        3: "Some unclear sounds - focus on vowel clarity",
        2: "1/3 unclear - record & mimic native clips",
        1: "Strong accent - drill minimal pairs daily",
        0: "Mostly unintelligible - slow phoneme drills + shadowing"
    }
}

def pte_band(value: float, band_map: dict) -> int:
    for th in sorted(band_map.keys(), reverse=True):
        if value >= th:
            return band_map[th]
    return min(band_map.values())

class PTESpeakingScorer:
    def __init__(self, model_size: str = "base"):
        print("Initializing PTE Speaking Scorer...")
        self.whisper = whisper.load_model(model_size)
        print("Whisper model loaded!")

    def is_url(self, path: str) -> bool:
        """Check if the given path is a URL."""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
    
    def download_audio(self, audio_url: str) -> str:
        try:
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()  # Raise an error for bad status
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp.flush()
            return tmp.name
        except requests.RequestException as e:
            print(f"Failed to download audio: {e}")
            return None

    # ----------  ASR  ----------
    def transcribe_audio(self, audio_src: str) -> Dict:
        print("Transcribing audio with Whisper...")
        audio_path = ''

        if self.is_url(audio_src):
            print("Downloading audio from URL...")
            audio_path = self.download_audio(audio_src)
            if not audio_path:
                return {}  # Return empty dict if download failed

            result = self.whisper.transcribe(audio_path, word_timestamps=True, language='en')

              # Clean up the temporary file if it was downloaded
            if bool(audio_path):
                os.remove(audio_path)
        else:
            result = self.whisper.transcribe(audio_src, word_timestamps=True, language='en')

        words = [
            {"word": w["word"].strip(), "start": w["start"], "end": w["end"]}
            for seg in result["segments"] for w in seg.get("words", [])
        ]
        
        return {"text": result["text"], "words": words}

    # ----------  PHONEMES  ----------
    def convert_to_phonemes(self, text: str) -> List[str]:
        if not text.strip():
            return []
        phones = phonemize(
            re.sub(r"[^\w\s]", "", text.lower()),
            language="en-us", 
            backend="espeak",
            strip=True, preserve_punctuation=False, with_stress=True
        )
        phoneme_list = phones.split()
        return phoneme_list
    

      # ----------  PRONUNCIATION ----------
    def pronunciation_score(self, spoken_phoneme: List[str], ref_phoneme: List[str]) -> int:
        if not spoken_phoneme or not ref_phoneme:
            return 0
        def lev(a, b):
            dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
            for i in range(len(a)+1): dp[i][0]=i
            for j in range(len(b)+1): dp[0][j]=j
            for i in range(1,len(a)+1):
                for j in range(1,len(b)+1):
                    dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1,
                                 dp[i-1][j-1]+(a[i-1]!=b[j-1]))
            return dp[len(a)][len(b)]
        dist = lev(spoken_phoneme, ref_phoneme)
        acc = 1 - dist / max(len(spoken_phoneme), len(ref_phoneme))
        band = pte_band(acc*100, {95:5,80:4,60:3,40:2,20:1,0:0})
        return band

    # ----------  FLUENCY ----------
    def fluency_score(self, words: List[Dict]) -> int:
        if not words:
            return 0
        dur = words[-1]["end"] - words[0]["start"]
        wpm = (len(words)/max(dur,1e-6))*60
        pauses=[w["start"]-words[i-1]["end"]
                for i,w in enumerate(words) if i and w["start"]-words[i-1]["end"]>0.15]
        long=len([p for p in pauses if p>1])
        reps=sum(1 for i in range(1,len(words))
                 if words[i]["word"].lower()==words[i-1]["word"].lower())

        if 140<=wpm<=180 and long==0 and reps==0: band=5
        elif 120<=wpm<=200 and long<=1 and reps<=1: band=4
        elif 100<=wpm<=220 and long<=1 and reps<=3: band=3
        elif len(words)>=6 and long<=1: band=2
        else: band = 1 if len(words)>=3 else 0
        return band


    # ----------  CONTENT  ----------
    def _tokens(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())
    
    def generate_alignment_feedback(self, ref_tokens: List[str], skp_tokens: List[str], alignment: List[Tuple[str, int, int, int, int]]) -> List[Dict]:
        """
        Returns feedback as a list of dicts for easier frontend rendering.
        Each dict contains:
            - type: 'correct', 'replace', 'missing', 'extra'
            - ref: reference words (if applicable)
            - spoken: spoken words (if applicable)
            - indices: positions in reference and spoken
        """
        feedback = []
        for tag, i1, i2, j1, j2 in alignment:
            match tag:
                case 'equal':
                    feedback.append({
                        "type": "correct",
                        "spoken": skp_tokens[j1:j2],
                        "ref": ref_tokens[i1:i2],
                        # "ref_indices": [i1, i2],
                        # "spoken_indices": [j1, j2]
                    })
                case 'replace':
                    feedback.append({
                        "type": "replace",
                        "spoken": skp_tokens[j1:j2],
                        "ref": ref_tokens[i1:i2],
                        # "ref_indices": [i1, i2],
                        # "spoken_indices": [j1, j2]
                    })
                case 'delete':
                    feedback.append({
                        "type": "missing",
                        "spoken": skp_tokens[j1:j2],
                        "ref": ref_tokens[i1:i2],
                        # "ref_indices": [i1, i2],
                        # "spoken_indices": [j1, j2]
                    })
                case 'insert':
                    feedback.append({
                        "type": "extra",
                        "spoken": skp_tokens[j1:j2],
                        "ref": ref_tokens[i1: i2],
                        # "ref_indices": [i1, i2],
                        # "spoken_indices": [j1, j2]
                    })
        return feedback

    def content_read_aloud(self, spoken: str, ref: str) -> Tuple[int, List[Dict]]:
        ref_tokens = self._tokens(ref)
        skp_tokens = self._tokens(spoken)
        matcher = difflib.SequenceMatcher(None, ref_tokens, skp_tokens)
        errors = sum(max(i2 - i1, j2 - j1)
                    for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")
        raw = max(0, len(ref_tokens) - errors)
        acc = raw / max(1, len(ref_tokens))
        band = pte_band(acc * 100, {95: 5, 80: 4, 60: 3, 40: 2, 20: 1, 0: 0})

        # Use aligned words to provide more detailed feedback
        alignment = matcher.get_opcodes()
        alignment_feedback = self.generate_alignment_feedback(ref_tokens, skp_tokens, alignment)

        return band, alignment_feedback

    def content_repeat_sentence(self, spoken_text: str, ref_text: str) -> int:
        ref_tokens = self._tokens(ref_text)
        hyp_t = self._tokens(spoken_text)
        matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_t)
        correct = sum(b.size for b in matcher.get_matching_blocks())
        pct = (correct / max(1, len(ref_tokens))) * 100
        band = pte_band(pct, {100: 3, 50: 2, 1: 1, 0: 0})
        return band
    
    def generate_overall_remarks(self, content_band: int, pronunciation_band: int, fluency_band: int) -> str:
        # Generate a one-liner summary
        if content_band == 0:
            overall_remark = "Content score is 0, significantly impacting the overall score."
        elif content_band == 5 and pronunciation_band == 5 and fluency_band == 5:
            overall_remark = "Excellent performance in all areas!"
        elif content_band >= 4 and pronunciation_band >= 4 and fluency_band >= 4:
            overall_remark = "Very good performance with minor areas for improvement."
        elif content_band >= 3 and pronunciation_band >= 3 and fluency_band >= 3:
            overall_remark = "Good performance, but some areas need attention."
        else:
            overall_remark = "Needs improvement in multiple areas."

        return overall_remark

    # ----------  6.  FULL TASK ----------
    def score_speaking_task(self, audio_src: str, reference_text: str=None,
                            task_type: str="read_aloud") -> Dict:
        trans = self.transcribe_audio(audio_src)
        spoken_text, words = trans["text"], trans["words"]

        fluency_band = self.fluency_score(words)
        content_band = pronunciation_band = None
        note = ''
        alignment_feedback = None

        if reference_text:
            match task_type:
                case 'read_aloud':
                    content_band, alignment_feedback = self.content_read_aloud(spoken_text, reference_text)
                case "repeat_sentence":
                    content_band = self.content_repeat_sentence(spoken_text, reference_text)

            ref_phoneme = self.convert_to_phonemes(reference_text)
            spoken_phoneme = self.convert_to_phonemes(spoken_text)
            pronunciation_band = self.pronunciation_score(spoken_phoneme, ref_phoneme)

        # PTE rule: Content = 0 → item = 0
        if content_band == 0:
            overall_score = 0
            note = "Content band = 0 => overall 0 (PTE rule)"
        elif content_band is not None:
            overall_score = int(round(((content_band + pronunciation_band + fluency_band) / 3) * 18))
        else:
            overall_score = int(round((fluency_band / 5) * 90))
            note = "Fluency only (no reference provided)"

        overall_remarks = self.generate_overall_remarks(content_band, pronunciation_band, fluency_band)

        return {
            "task_type": task_type,
            "audio_src": audio_src,
            "reference_text": reference_text,
            "spoken_text": spoken_text,
            "score_distribution": {
                "content": {
                    'score': content_band,
                    'remark': SUGGESTIONS["content"][content_band]
                },
                "pronunciation": {
                    'score': pronunciation_band,
                    'remark':  SUGGESTIONS["pronunciation"][pronunciation_band]
                },
                'fluency': {
                    'score': fluency_band,
                    'remark': SUGGESTIONS["fluency"][fluency_band]
                },
            },
            "item_score": min(90, max(0, overall_score)),
            "overall_remarks": overall_remarks,
            'alignment_feedback': alignment_feedback or None,
            "note": note
        }

# ----------  QUICK DEMO  ----------
if __name__ == "__main__":
    scorer = PTESpeakingScorer("base")
    # audio_path = r"C:\NCC\PTE\assets\qbf.mp3"
    audio_path = r"C:\NCC\PTE\assets\monologue.mp3"
    reference_text = open("C:/NCC/PTE/assets/monologue.txt", "r").read().strip()
    # audio_url = "https://edux-demo.sgp1.digitaloceanspaces.com/649bc4d19627855fb6c58ab1/637e024c2519524733a812ea/client/1755426415500.mp3"
    # reference_text = "The quick brown fox jumped over the lazy dog"

    result = scorer.score_speaking_task(
        audio_src=audio_path,
        reference_text=reference_text,
        task_type="read_aloud"
    )
    print(json.dumps(result, indent=2))