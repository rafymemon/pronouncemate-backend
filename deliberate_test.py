# save as deliberate_test.py
# Run this after starting the server
import requests, base64, wave

def read_wav_as_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# First login to get token
login = requests.post("http://127.0.0.1:8000/api/login/",
    json={"username": "testuser", "password": "Test@1234"})
token = login.json()["token"]
headers = {"Authorization": f"Token {token}"}

# Test cases — record each WAV separately and test
test_cases = [
    {
        "script":  "think about the schedule carefully",
        "wav":     "test_think.wav",
        "note":    "Say: 'tink about de skedule carefully'"
    },
    {
        "script":  "the weather is very nice today",
        "wav":     "test_weather.wav",
        "note":    "Say: 'de weather is wery nice today'"
    },
]

for tc in test_cases:
    print(f"\nScript : {tc['script']}")
    print(f"You said: {tc['note']}")

    with open(tc['wav'], 'rb') as audio_file:
        resp = requests.post(
            "http://127.0.0.1:8000/api/analyze/",
            headers=headers,
            files={"audio": audio_file},
            data={"script_text": tc['script']}
        )

    result = resp.json()
    print(f"Score  : {result['score']}%  Grade: {result['grade']}")
    print(f"Words:")
    for w in result['words']:
        marker = "✓" if w['type'] == 'correct' else f"✗ {w['type']}"
        print(f"  {w['word']:<15} {marker:<20} {w['per_word_score']}%")
    if result['feedback']:
        print("Feedback:")
        for f in result['feedback']:
            print(f"  [{f['word']}] {f['explanation']}")