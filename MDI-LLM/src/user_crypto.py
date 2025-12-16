#!/usr/bin/env python3
"""
User-side encryption utilities for MDI-LLM

This script provides utilities for users to:
1. Generate their own ECDH keypair
2. Encrypt prompts before sending to the system
3. Decrypt results received from the system

Usage:
    # Generate user keypair (one time)
    python3 user_crypto.py generate-key --output user_key.pem
    
    # Get starter's public key
    python3 user_crypto.py get-starter-key --url http://localhost:8088 --output starter_key.pem
    
    # Encrypt a prompt
    python3 user_crypto.py encrypt-prompt --prompt "What is the capital of Spain?" --user-key user_key.pem --starter-key starter_key.pem --output encrypted_prompt.bin
    
    # Decrypt response
    python3 user_crypto.py decrypt-response --encrypted encrypted_response.bin --user-key user_key.pem --starter-key starter_key.pem
"""

import argparse
import sys
import requests
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from sub.utils.encryption import (
    generate_ecdh_keypair,
    serialize_public_key,
    serialize_private_key,
    deserialize_public_key,
    deserialize_private_key,
    derive_shared_key,
    aes_encrypt_string,
    aes_decrypt_string
)


def generate_keypair(output_file: str):
    """Generate user's ECDH keypair and save to file"""
    print("[INFO] Generating user ECDH keypair...")
    private_key, public_key = generate_ecdh_keypair()
    
    # Serialize both keys
    private_bytes = serialize_private_key(private_key)
    public_bytes = serialize_public_key(public_key)
    
    # Save to file (both keys together)
    output_path = Path(output_file)
    with open(output_path, 'wb') as f:
        # Write lengths first, then data
        f.write(len(private_bytes).to_bytes(4, 'big'))
        f.write(private_bytes)
        f.write(len(public_bytes).to_bytes(4, 'big'))
        f.write(public_bytes)
    
    print(f"[SUCCESS] Keypair saved to: {output_path}")
    print(f"[INFO] Keep this file secure! It contains your private key.")
    return output_path


def load_keypair(key_file: str):
    """Load user's keypair from file"""
    with open(key_file, 'rb') as f:
        # Read private key
        priv_len = int.from_bytes(f.read(4), 'big')
        priv_bytes = f.read(priv_len)
        private_key = deserialize_private_key(priv_bytes)
        
        # Read public key
        pub_len = int.from_bytes(f.read(4), 'big')
        pub_bytes = f.read(pub_len)
        public_key = deserialize_public_key(pub_bytes)
    
    return private_key, public_key


def get_starter_public_key(url: str, output_file: str):
    """Fetch starter's public key from the server"""
    print(f"[INFO] Fetching starter's public key from {url}/public_key...")
    
    try:
        response = requests.get(f"{url}/public_key", timeout=5)
        if response.status_code == 200:
            starter_pub_key = response.content
            
            # Save to file
            output_path = Path(output_file)
            with open(output_path, 'wb') as f:
                f.write(starter_pub_key)
            
            print(f"[SUCCESS] Starter's public key saved to: {output_path}")
            return output_path
        else:
            print(f"[ERROR] Failed to fetch key. Status: {response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to connect to starter: {e}")
        sys.exit(1)


def encrypt_prompt(prompt: str, user_key_file: str, starter_key_file: str, output_file: str):
    """Encrypt a prompt for the starter"""
    print("[INFO] Encrypting prompt...")
    
    # Load user's keypair
    user_private_key, user_public_key = load_keypair(user_key_file)
    
    # Load starter's public key
    with open(starter_key_file, 'rb') as f:
        starter_pub_bytes = f.read()
    starter_public_key = deserialize_public_key(starter_pub_bytes)
    
    # Derive shared key
    shared_key = derive_shared_key(user_private_key, starter_public_key)
    
    # Encrypt the prompt
    ciphertext, nonce, tag = aes_encrypt_string(prompt, shared_key)
    
    # Save user's public key + encrypted data
    output_path = Path(output_file)
    with open(output_path, 'wb') as f:
        # Write user's public key first (so starter can derive shared key)
        user_pub_bytes = serialize_public_key(user_public_key)
        f.write(len(user_pub_bytes).to_bytes(4, 'big'))
        f.write(user_pub_bytes)
        
        # Write encrypted data
        f.write(len(nonce).to_bytes(4, 'big'))
        f.write(nonce)
        f.write(len(tag).to_bytes(4, 'big'))
        f.write(tag)
        f.write(len(ciphertext).to_bytes(4, 'big'))
        f.write(ciphertext)
    
    print(f"[SUCCESS] Encrypted prompt saved to: {output_path}")
    print(f"[INFO] Ciphertext preview: {ciphertext[:32].hex()}...")
    return output_path


def decrypt_response(encrypted_file: str, user_key_file: str, starter_key_file: str):
    """Decrypt response from the starter"""
    print("[INFO] Decrypting response...")
    
    # Load user's keypair
    user_private_key, _ = load_keypair(user_key_file)
    
    # Load starter's public key
    with open(starter_key_file, 'rb') as f:
        starter_pub_bytes = f.read()
    starter_public_key = deserialize_public_key(starter_pub_bytes)
    
    # Derive shared key
    shared_key = derive_shared_key(user_private_key, starter_public_key)
    
    # Load encrypted data
    with open(encrypted_file, 'rb') as f:
        # Read nonce
        nonce_len = int.from_bytes(f.read(4), 'big')
        nonce = f.read(nonce_len)
        
        # Read tag
        tag_len = int.from_bytes(f.read(4), 'big')
        tag = f.read(tag_len)
        
        # Read ciphertext
        ct_len = int.from_bytes(f.read(4), 'big')
        ciphertext = f.read(ct_len)
    
    # Decrypt
    plaintext = aes_decrypt_string(ciphertext, nonce, tag, shared_key)
    
    print(f"[SUCCESS] Decrypted response:")
    print("=" * 80)
    print(plaintext)
    print("=" * 80)
    return plaintext


def main():
    parser = argparse.ArgumentParser(
        description="User-side encryption utilities for MDI-LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Generate key command
    gen_parser = subparsers.add_parser('generate-key', help='Generate user ECDH keypair')
    gen_parser.add_argument('--output', '-o', required=True, help='Output file for keypair')
    
    # Get starter key command
    get_parser = subparsers.add_parser('get-starter-key', help='Fetch starter public key')
    get_parser.add_argument('--url', required=True, help='Starter URL (e.g., http://localhost:8088)')
    get_parser.add_argument('--output', '-o', required=True, help='Output file for starter public key')
    
    # Encrypt prompt command
    enc_parser = subparsers.add_parser('encrypt-prompt', help='Encrypt a prompt')
    enc_parser.add_argument('--prompt', '-p', required=True, help='Prompt text to encrypt')
    enc_parser.add_argument('--user-key', required=True, help='User keypair file')
    enc_parser.add_argument('--starter-key', required=True, help='Starter public key file')
    enc_parser.add_argument('--output', '-o', required=True, help='Output file for encrypted prompt')
    
    # Decrypt response command
    dec_parser = subparsers.add_parser('decrypt-response', help='Decrypt response')
    dec_parser.add_argument('--encrypted', '-e', required=True, help='Encrypted response file')
    dec_parser.add_argument('--user-key', required=True, help='User keypair file')
    dec_parser.add_argument('--starter-key', required=True, help='Starter public key file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'generate-key':
        generate_keypair(args.output)
    elif args.command == 'get-starter-key':
        get_starter_public_key(args.url, args.output)
    elif args.command == 'encrypt-prompt':
        encrypt_prompt(args.prompt, args.user_key, args.starter_key, args.output)
    elif args.command == 'decrypt-response':
        decrypt_response(args.encrypted, args.user_key, args.starter_key)


if __name__ == "__main__":
    main()
