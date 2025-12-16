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
# SCALE = 1e4  # quantization scale
import torch
import numpy as np
from Crypto.Cipher import AES
import os

def aes_encrypt_tensor_quantized(tensor: torch.Tensor, key: bytes) -> tuple:
    """
    Encrypts tensor data preserving exact bits (Bit-Perfect for bfloat16).
    """
    import time
    start = time.time()
    nonce = os.urandom(12)

    # 1. Move to CPU. DO NOT call .float() or .numpy() on the tensor directly!
    # casting to float32 adds noise. .numpy() crashes on bfloat16.
    cpu_tensor = tensor.detach().cpu()
    
    # 2. View as RAW BYTES (uint8). 
    # This treats the numbers as generic data, preserving perfect precision.
    data = cpu_tensor.view(torch.uint8).numpy().tobytes()

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    
    # Logging
    # print("[Encryption]") # Commented out for speed
    elapsed = time.time() - start
    with open("encryption_latency.txt", "a") as f:
        f.write(f"ENCRYPT {elapsed:.6f}\n")
    
    # Send dtype name so receiver knows how to interpret the bytes
    dtype_name = str(tensor.dtype).split(".")[-1] 
    return ciphertext, nonce, tag, tensor.shape, dtype_name


def aes_decrypt_tensor_quantized(
    ciphertext: bytes, nonce: bytes, tag: bytes, shape, dtype, key: bytes, device: str = "cpu"
) -> torch.Tensor:
    """
    Decrypts bytes and reconstructs tensor with correct dtype.
    """
    import time
    start = time.time()
    
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    decrypted_bytes = cipher.decrypt_and_verify(ciphertext, tag)
    
    # 1. Reconstruct Tensor from Raw Bytes
    # torch.frombuffer is efficient and handles bfloat16 bytes natively.
    # We create a copy to ensure the memory is writable/movable.
    tensor = torch.frombuffer(bytearray(decrypted_bytes), dtype=dtype).reshape(shape)
    
    # 2. Move to GPU
    final_tensor = tensor.to(device=device)

    elapsed = time.time() - start
    with open("encryption_latency.txt", "a") as f:
        f.write(f"DECRYPT {elapsed:.6f}\n")
        
    return final_tensor


def generate_ecdh_keypair():
    """Generate ECDH private and public key."""
    private_key = ec.generate_private_key(ec.SECP384R1(), default_backend())    #curve P-384 = SECP384R1, NIST assumed ok for yy 2031+
    public_key = private_key.public_key()
    return private_key, public_key


def serialize_public_key(public_key):
    """Prepare the public key for transmission over the network (HTTP)."""
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def deserialize_public_key(public_bytes):
    return serialization.load_pem_public_key(public_bytes, backend=default_backend())


def derive_shared_key(private_key, peer_public_key):
    """Performs the Elliptic Curve multiplication to calculate the Shared Secret"""
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