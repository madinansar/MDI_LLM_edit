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

# Mapping torch dtypes to numpy dtypes for direct encryption/decryption
TORCH_TO_NUMPY_DTYPE = {
    'float32': np.float32,
    'float64': np.float64,
    'float16': np.float16,
    'bfloat16': np.uint16,  # bfloat16 stored as uint16 bit pattern
    'int32': np.int32,
    'int64': np.int64,
    'int16': np.int16,
    'int8': np.int8,
}


def aes_encrypt_tensor_quantized(tensor: torch.Tensor, key: bytes) -> tuple:
    """
    Encrypts a tensor directly using AES-GCM (no quantization).
    Returns ciphertext, nonce, tag, shape, dtype.
    """
    import time

    start = time.time()
    nonce = os.urandom(12)
    
    # Get tensor dtype
    dtype_name = str(tensor.dtype).split(".")[-1]
    
    # Optimize: Chain operations instead of intermediate variables
    # Special handling for bfloat16 (not directly supported by numpy)
    if dtype_name == 'bfloat16':
        # Convert bfloat16 to uint16 view for byte serialization (chained)
        data = tensor.detach().cpu().view(torch.uint16).numpy().tobytes()
    else:
        # Direct byte conversion for other dtypes (chained)
        data = tensor.detach().cpu().numpy().tobytes()
    
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    print("[Encryption]")
    elapsed = time.time() - start
    with open("encryption_latency.txt", "a") as f:
        f.write(f"ENCRYPT {elapsed:.6f}\n")
    
    return ciphertext, nonce, tag, tensor.shape, dtype_name


def aes_decrypt_tensor_quantized(
    ciphertext: bytes, nonce: bytes, tag: bytes, shape, dtype, key: bytes, device: str = "cpu"
) -> torch.Tensor:
    """
    Decrypts a tensor directly using AES-GCM (no quantization).
    """
    import time

    start = time.time()
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    decrypted = cipher.decrypt_and_verify(ciphertext, tag)
    
    # Ensure dtype is a string
    if isinstance(dtype, torch.dtype):
        dtype_str = str(dtype).split(".")[-1]
    else:
        dtype_str = str(dtype)
    
    # Get torch dtype object
    dtype_obj = getattr(torch, dtype_str)
    
    # Special handling for bfloat16
    if dtype_str == 'bfloat16':
        # Reconstruct from uint16 view
        arr = np.frombuffer(decrypted, dtype=np.uint16).reshape(shape)
        tensor = torch.from_numpy(arr.copy()).view(torch.bfloat16).to(device)
    else:
        # Direct reconstruction for other dtypes
        np_dtype = TORCH_TO_NUMPY_DTYPE.get(dtype_str, np.float32)
        arr = np.frombuffer(decrypted, dtype=np_dtype).reshape(shape)
        tensor = torch.from_numpy(arr.copy()).to(dtype_obj).to(device)
    
    elapsed = time.time() - start
    with open("encryption_latency.txt", "a") as f:
        f.write(f"DECRYPT {elapsed:.6f}\n")
    
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