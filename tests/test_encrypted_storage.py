from cryptography.fernet import Fernet
import pytest

from geologic_memory.db import decrypt_session_payload, encrypt_session_payload


def test_session_payload_round_trip_is_encrypted():
    key = Fernet.generate_key().decode("ascii")
    plaintext = b"private memory payload"
    ciphertext = encrypt_session_payload(plaintext, key)
    assert ciphertext != plaintext
    assert plaintext not in ciphertext
    assert decrypt_session_payload(ciphertext, key) == plaintext


def test_wrong_storage_key_is_rejected():
    first = Fernet.generate_key().decode("ascii")
    second = Fernet.generate_key().decode("ascii")
    ciphertext = encrypt_session_payload(b"secret", first)
    with pytest.raises(ValueError, match="authenticated or decrypted"):
        decrypt_session_payload(ciphertext, second)
