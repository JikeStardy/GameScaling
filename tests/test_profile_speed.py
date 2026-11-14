from pathlib import Path

from gamescaling.server import decode_image
from tools.profile_speed import make_request


def image_urls(request):
    return [part.image_url.url for message in request.messages
            if isinstance(message.content, list) for part in message.content
            if part.type == "image_url"]


def test_rolling_cache_fixture_preserves_pixels_and_previous_frame_identities():
    root = Path(__file__).resolve().parents[1]
    first = image_urls(make_request(root, 2, 100))
    second = image_urls(make_request(root, 2, 101))
    original = image_urls(make_request(root, 0))[0]
    assert len(first) == 3 and len(set(first)) == 3
    assert first[1:] == second[:-1]
    pixels = decode_image(original).tobytes()
    assert all(decode_image(url).tobytes() == pixels for url in first + second)
