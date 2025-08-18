import whisper

model = whisper.load_model("tiny")

# load audio and pad/trim it to fit 30 seconds
# path = r"C:/Users/Suvam/Desktop/projects/webPageChecker/audio4.mp3"
path = r"C:/NCC/PTE/monologue.mp3"

audio = whisper.load_audio(path)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)
# print(result.)

# print the recognized text
print(result.text)