import os
import torch
import numpy as np
from Crypto.Cipher import AES
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

# Hardcoded 256-bit AES key (must be identical on both nodes)
# AES_KEY = b"0123456789abcdef0123456789abcdef"  # 32 bytes
SCALE = 1e4  # quantization scale


def aes_encrypt_tensor_quantized(tensor: torch.Tensor, key: bytes) -> tuple:
    """
    Quantizes and encrypts a tensor using AES-GCM.
    Returns ciphertext, nonce, tag, shape, dtype.
    """
    import time

    start = time.time()
    nonce = os.urandom(12)
    arr = (tensor.detach().cpu().float().numpy() * SCALE).round().astype(np.int32)
    data = arr.tobytes()
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    print("[Encryption]")
    elapsed = time.time() - start
    with open("encryption_latency.txt", "a") as f:
        f.write(f"ENCRYPT {elapsed:.6f}\n")
    dtype_name = str(tensor.dtype).split(".")[-1]
    return ciphertext, nonce, tag, tensor.shape, dtype_name


def aes_decrypt_tensor_quantized(
    ciphertext: bytes, nonce: bytes, tag: bytes, shape, dtype, key: bytes, device: str = "cpu"
) -> torch.Tensor:
    """
    Decrypts and dequantizes a tensor using AES-GCM.
    """
    import time

    start = time.time()
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    decrypted = cipher.decrypt_and_verify(ciphertext, tag)
    arr = np.frombuffer(decrypted, dtype=np.int32).reshape(shape)
    tensor = torch.tensor(arr, dtype=torch.float32, device=device) / SCALE
    elapsed = time.time() - start
    with open("encryption_latency.txt", "a") as f:
        f.write(f"DECRYPT {elapsed:.6f}\n")
    return tensor.to(dtype)


def generate_ecdh_keypair():
    """Generate ECDH private and public key."""
    private_key = ec.generate_private_key(ec.SECP384R1(), default_backend())
    public_key = private_key.public_key()
    return private_key, public_key


def serialize_public_key(public_key):
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def deserialize_public_key(public_bytes):
    return serialization.load_pem_public_key(public_bytes, backend=default_backend())


def derive_shared_key(private_key, peer_public_key):
    shared_secret = private_key.exchange(ec.ECDH(), peer_public_key)
    # Derive a 32-byte AES key from the shared secret
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"mdi-llm-ecdh",
        backend=default_backend(),
    ).derive(shared_secret)
    return derived_key