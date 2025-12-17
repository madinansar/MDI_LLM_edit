
import os
import time
import torch
import numpy as np
from Crypto.Cipher import AES
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

# Performance optimization: disable latency logging in production
ENABLE_LATENCY_LOGGING = os.environ.get('ENABLE_ENCRYPTION_LOGGING', '0') == '1'
_latency_buffer = []
_BUFFER_SIZE = 100

def _flush_latency_buffer():
    """Flush buffered latency measurements to disk."""
    if _latency_buffer:
        with open("encryption_latency.txt", "a") as f:
            f.writelines(_latency_buffer)
        _latency_buffer.clear()

# Hardcoded 256-bit AES key (must be identical on both nodes)
# AES_KEY = b"0123456789abcdef0123456789abcdef"  # 32 bytes

# Mapping torch dtypes to numpy dtypes for encryption/decryption
TORCH_TO_NUMPY_DTYPE = {
    'float32': np.float32,
    'float64': np.float64,
    'float16': np.float16,
    'bfloat16': np.uint16,  # bfloat16 needs special handling
    'int32': np.int32,
    'int64': np.int64,
    'int16': np.int16,
    'int8': np.int8,
}


def aes_encrypt_tensor_quantized(tensor: torch.Tensor, key: bytes) -> tuple:
    """
    Encrypts a tensor using AES-GCM (no quantization - direct byte encryption).
    Returns ciphertext, nonce, tag, shape, dtype.
    """
    start = time.time() if ENABLE_LATENCY_LOGGING else None
    nonce = os.urandom(12)
    # Get tensor data as bytes directly (no quantization)
    dtype_name = str(tensor.dtype).split(".")[-1]
    tensor_cpu = tensor.detach().cpu()
    # Special handling for bfloat16 (not directly supported by numpy)
    if dtype_name == 'bfloat16':
        # Convert bfloat16 to uint16 view for byte serialization
        data = tensor_cpu.view(torch.uint16).numpy().tobytes()
    else:
        data = tensor_cpu.numpy().tobytes()
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    print("[Encryption]")
    # Optimized latency logging: buffer writes to reduce I/O overhead
    if ENABLE_LATENCY_LOGGING:
        elapsed = time.time() - start
        _latency_buffer.append(f"ENCRYPT {elapsed:.6f}\n")
        if len(_latency_buffer) >= _BUFFER_SIZE:
            _flush_latency_buffer()
    return ciphertext, nonce, tag, tensor.shape, dtype_name


def aes_decrypt_tensor_quantized(
    ciphertext: bytes, nonce: bytes, tag: bytes, shape, dtype, key: bytes, device: str = "cpu"
) -> torch.Tensor:
    """
    Decrypts a tensor using AES-GCM (no quantization - direct byte decryption).
    """
    start = time.time() if ENABLE_LATENCY_LOGGING else None
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    decrypted = cipher.decrypt_and_verify(ciphertext, tag)
    # Ensure dtype is a string
    if isinstance(dtype, torch.dtype):
        dtype = str(dtype).split(".")[-1]
    elif not isinstance(dtype, str):
        dtype = str(dtype)
    # Get torch dtype object
    dtype_obj = getattr(torch, dtype)
    # Special handling for bfloat16
    if dtype == 'bfloat16':
        # Reconstruct from uint16 view
        arr = np.frombuffer(decrypted, dtype=np.uint16).reshape(shape)
        tensor = torch.from_numpy(arr).view(torch.bfloat16).to(device)
    else:
        # Get corresponding numpy dtype
        np_dtype = TORCH_TO_NUMPY_DTYPE.get(dtype, np.float32)
        arr = np.frombuffer(decrypted, dtype=np_dtype).reshape(shape)
        tensor = torch.from_numpy(arr).to(dtype_obj).to(device)
    # Optimized latency logging: buffer writes to reduce I/O overhead
    if ENABLE_LATENCY_LOGGING:
        elapsed = time.time() - start
        _latency_buffer.append(f"DECRYPT {elapsed:.6f}\n")
        if len(_latency_buffer) >= _BUFFER_SIZE:
            _flush_latency_buffer()
    return tensor


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


def serialize_private_key(private_key):
    """Serialize private key to PEM format"""
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )


def deserialize_private_key(private_bytes):
    """Deserialize private key from PEM format"""
    return serialization.load_pem_private_key(
        private_bytes, 
        password=None, 
        backend=default_backend()
    )


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


def aes_encrypt_string(plaintext: str, key: bytes) -> tuple:
    """
    Encrypt a string using AES-GCM.
    Returns (ciphertext, nonce, tag)
    """
    nonce = os.urandom(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode('utf-8'))
    return ciphertext, nonce, tag


def aes_decrypt_string(ciphertext: bytes, nonce: bytes, tag: bytes, key: bytes) -> str:
    """
    Decrypt a string using AES-GCM.
    Returns the plaintext string.
    """
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    return plaintext.decode('utf-8')