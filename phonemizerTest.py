from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library('C:\Program Files\eSpeak NG\libespeak-ng.dll')

text = "Hello world"
phonemes = phonemize(text, language="en-us")

print(phonemes)
