# Copyright (c) 2024 Davide Macario
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import logging
import os
import pickle
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import cherrypy as cp
import requests
import torch

from sub import PromptStyle
from sub.config import N_LAYERS_NODES
from sub.gptserver import GPTServer
from sub.typing import FileType
from sub.utils import load_from_pt, load_from_hf_direct, split_and_store_with_finisher
from sub.utils.encryption import generate_ecdh_keypair
docstring = """
Distributed implementation of the Llama architecture using Model-Distributed Inference
(with pipeline parallelism).
This implementation allows to run a Llama model (of the ones compatible with LitGPT)
over a network of "nodes" that can be positioned on different physical hosts.

The distributed implementation consists of partitioning the model layers among all the
nodes in the network. Then, each node will work with a subset of layers, receiving the
input from the previous node in the network, and transmitting the output to the next
one.
This allows to reduce the memory usage for each node compared to the memory required to
run the complete model on a single node, allowing to run larger models by increasing the
number of network nodes.

Parallelism is introduced by increasing the batch size, allowing to generate different
independent samples.
Due to the autoregressive nature of LLMs, to generate a new token in the sequence, it is
required to feed back the new token to the model input, hence, if generating a single
piece of text, the nodes that are not currently busy processing their local model chunk
would be idle, waiting for the information to reach them.
If we generate more than one sample, it is possible for different nodes to work on 
different samples concurrently, improving efficiency.
In particular, when the number of samples (batch size) is greater or equal than the
number of nodes, it is possible to ensure every node is always working on a different
sample.
This mechanism, that we call, "recurrent pipelining", allows to achieve a generation
rate (tokens/second) which is higher than sequential generation on a single device.
With this work, we provide a proof of concept for the use of pipeline parallelism for 
transformer architecture inference, resulting in an optimized implementation achieving
competitive performances, with the added bonus of enabling LLM deployment on Edge
devices (the testbed for this project was made up of Nvidia Jetson TX2 modules).

The application architecture is the following.
We define 2 types of nodes: "starter" nodes and "secondary" nodes; unlike the name
suggests, there is no "master-slave" relationship between the 2, the "starter" is just
acting as the "entrypoint"/application interface with the "user".
Starter nodes are used to initialize secondary nodes and contain the first and last
layers of the model. They take the model input (prompt) and collect the output tokens.
As a consequence, they are the ones that "know" the exact number of tokens (and
iterations) to be performed in the current inference run.
Secondary nodes are "activated" by the starting node, and just receive inputs to be 
passed through the local model chunk.
For efficiency reasons, we assume the model chunks are already located on the devices
themselves, but it is also possible to have the starter node split the model layers and
send them to the different devices.

Communication happens over 2 channels.
For coordination and initialization, each node acts as an HTTP server and messages are
sent over HTTP.
For the transmissions of intermediate activations, the nodes use bare Python sockets 
running over TCP/IP. The lack of a fully-fledged application layer protocol allows for
a faster message exchange.
At the application layer, the message only contains a header of fixed length, specifying
the exact message size in bytes, which allows to read the exact amount of bytes to
prevent issues due to message truncation.
Being message transmission a crucial part of the application, as it is necessary to
ensure it does not slow down the overall operation, we implement it through input and 
output message queues running on separate threads.
Once a message is received, it is placed in the input queue, from where the processing
thread (the one performing model forward passes) will extract it (in order), process it,
and place the output message in the output queue.
There, a separate thread will extract it and transmit it.

The application is composed of the following modules:
- GPTDistributed: entrypoint for initializing nodes of any type; for starter nodes, it
provides methods used at initialization.
- GPTServer: core of the application; it creates the HTTP server for coordination and 
sets up the message transmission sockets. It contains the definition of all the
application threads and the processing loop.
- GPT: model definition, based on LitGPT (by Lightning AI, in turn based on NanoGPT).
The actual application uses submodels over with the same architecture (with the same
building blocks).
"""

script_dir = os.path.dirname(__file__)

logger_wp = logging.getLogger("model_dist")
logger_wp.setLevel(logging.ERROR)

MODEL_TYPE = ""
CTX = nullcontext()


class GPTDistributed:
    __doc__ = docstring
    init_msg = {
        "role": "",
        "prev_node": {},
        "next_node": {},
        "model_config": {},
        "n_nodes": 0,
        "n_samples": 0,
        "max_seq_length": None,
    }

    def __init__(
        self,
        node_type: str,
        config_file: FileType,
        *,
        ckpt_dir: Optional[FileType] = None,
        chunk_path: Optional[FileType] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        secondary_index: Optional[int] = None,
        model_seq_length: Optional[int] = None,
        **kwargs,
    ):
        """
        Instantiate a GPTDistributed object, allowing to run a node for
        Model-Distributed Inference.

        Args:
            node_type: role of the node - can be "starter", "secondary", "secondary:<n>"
                if "secondary", make sure to specify `secondary_index`
            config_file: configuration file for the node(s);
                In starters: it is the configuration of the whole network (full config.json)
                In secondary: it _can_ be the full config or the individual node config
            *
            ckpt_dir (optional):
                For starter nodes: checkpoint directory containing the
                model_config.yaml file and either the chunks/ folder or the .pth model.
                In secondary nodes, one between ckpt_dir and chunk_path must be present.
            chunk_path (optional): path of the model chunk for the current node.
                In starters: it can be inferred from ckpt_dir.
                In secondary: if missing, the chunk path will be inferred.
                NOTE: chunk path will always be passed, i.e., the model will always be
                    loaded from disk!
            device (default: None): string indicating the device used to load and run
                the model; if not specified, the application will try to get it from the
                nodes configuration file.
            secondary_index (optional): positional, zero-index of the secondary node;
                not necessary if only 1 secondary node is present in the configuration.
                Not used by starter nodes.
            model_seq_length (optional): maximum sequence length of the model; should be
                less or equal than the one specified in the config (default value)
            Keyword args (optional): allowing to specify verb=VERB and plots=PLOTS
                (bool)
        """
        if isinstance(ckpt_dir, str):
            self.ckpt_dir = Path(ckpt_dir)
        else:
            self.ckpt_dir = ckpt_dir
        if isinstance(chunk_path, str):
            self.chunk_path = Path(chunk_path)
        else:
            # Includes None
            self.chunk_path = chunk_path

        if isinstance(config_file, str):
            config_file = Path(config_file)

        self.torch_device = device if device else None

        global VERB
        VERB = False if "verb" not in kwargs else bool(kwargs["verb"])
        global PLOTS
        PLOTS = False if "plots" not in kwargs else bool(kwargs["plots"])

        self.compile = False if "compile" not in kwargs else bool(kwargs["compile"])
        self.dtype = dtype

        if self.ckpt_dir:
            self.full_model_name = self.ckpt_dir.name
            if VERB:
                print(f"Using model: {self.full_model_name}")

        self.node_type = node_type
        with open(config_file, "r") as f:
            self.node_config = json.load(f)

        if VERB:
            print("Loaded nodes config file")

        if self.node_type == "starter":
            assert self.ckpt_dir, "No model was specified!"
            self.n_secondary = len(self.node_config["nodes"]["secondary"])
            self.n_nodes = 1 + self.n_secondary
            if VERB and self.n_nodes == 1:
                print("Running in standalone mode!")
            self.own_config = self.node_config["nodes"]["starter"]
            self.own_addr = self.own_config["addr"]
            self.own_comm_port = self.own_config["communication"]["port"]
            self.own_inference_port_in = self.own_config["inference"]["port_in"]
            self.own_inference_port_out = self.own_config["inference"]["port_out"]

            # TODO: add support for downloading model as well (extra)
            # Load model config
            if self.chunk_path:  # In standalone mode, chunk path is lit_model.pth
                node_chunks_dir = self.chunk_path.resolve().parent
                self.model_was_split = True
            else:
                #madina
                # node_chunks_dir = self.ckpt_dir / "chunks" / f"{self.n_nodes}nodes"
                # This is the fix. It now looks for ".../3nodes_finisher/"
                node_chunks_dir = self.ckpt_dir / "chunks" / f"{self.n_nodes}nodes_finisher"
                self.model_was_split = node_chunks_dir.is_dir()

            # Check if we have lit_model.pth (converted) or only HF format
            has_lit_model = (self.ckpt_dir / "lit_model.pth").exists()
            
            if not self.model_was_split and self.n_nodes > 1:
                # Load model, split it, store it; the chunks will then be transmitted
                if VERB:
                    print("Chunks not found! Splitting the model")
                if has_lit_model:
                    self.model_config, full_model = load_from_pt(self.ckpt_dir)
                else:
                    # Load directly from HF format
                    if VERB:
                        print("Loading from HF format directly (no lit_model.pth found)")
                    self.model_config, full_model = load_from_hf_direct(self.ckpt_dir)
                assert full_model is not None
                #split and store madina
                node_chunks_dir = split_and_store_with_finisher(
                    full_model, self.n_nodes, self.ckpt_dir
                )
            else:
                # Here if either model was already split or running in standalone mode
                if has_lit_model:
                    self.model_config, _ = load_from_pt(self.ckpt_dir, config_only=True)
                else:
                    # Load config from HF format
                    self.model_config, _ = load_from_hf_direct(self.ckpt_dir, config_only=True)

            self.model_seq_length = None
            if model_seq_length and model_seq_length > self.model_config.block_size:
                raise ValueError(
                    f"The truncated sequence length {model_seq_length} should be lower "
                    "or equal than the model's max sequence length "
                    f"{self.model_config.block_size}"
                )
            else:
                self.model_seq_length = model_seq_length

            # For standalone mode without lit_model.pth, we need to load weights to memory
            model_weights = None
            if not self.chunk_path:
                if self.n_nodes > 1:
                    self.chunk_path = node_chunks_dir / "model_starter.pth"
                else:
                    # Standalone mode
                    if has_lit_model:
                        self.chunk_path = self.ckpt_dir / "lit_model.pth"
                    else:
                        # Load weights from HF format directly to memory
                        if VERB:
                            print("Loading model weights from HF format for standalone mode...")
                        _, model_weights = load_from_hf_direct(self.ckpt_dir)
                        self.chunk_path = None  # Will pass weights directly

            if (not self.chunk_path or not self.model_was_split) and self.n_nodes > 1:
                self.chunk_path = node_chunks_dir / "model_starter.pth"

            self.gpt_serv = GPTServer(
                node_config=self.node_config,
                node_type=self.node_type,
                model_config=self.model_config,
                chunk_path=self.chunk_path,
                model_weights=model_weights,  # Pass weights for HF format standalone
                tokenizer_dir=self.ckpt_dir,
                model_device=self.torch_device,
                dtype=dtype,
                **kwargs,
                model_type=self.full_model_name,
                model_seq_length=self.model_seq_length
            )
            print("[DEBUG GPTDistributed.__init__] GPTServer constructed:")
            print("  _ecdh_private_key:", getattr(self.gpt_serv, '_ecdh_private_key', None))
            print("  _ecdh_public_key:", getattr(self.gpt_serv, '_ecdh_public_key', None))
            print("  _serialize_public_key:", getattr(self.gpt_serv, '_serialize_public_key', None))
            print("  _deserialize_peer_key:", getattr(self.gpt_serv, '_deserialize_peer_key', None))
            print("  _derive_shared_key:", getattr(self.gpt_serv, '_derive_shared_key', None))

        elif "secondary" in self.node_type:
            # FIXME: secondary node may be completely agnostic of the used model and
            # receive the model config (and the chunk) at initialization
            assert (
                self.ckpt_dir or self.chunk_path
            ), "Need to specify at least 1 between the chunk path and the checkpoint directory"

            # Can either pass "secondary:ind" or secondary_index=ind
            split_type = self.node_type.split(":")
            self.secondary_index = (
                secondary_index if len(split_type) < 2 else int(split_type[1])
            )
            assert self.secondary_index is not None

            self.node_type = f"secondary:{self.secondary_index}"
            self.n_nodes = None
            # Initialize secondary node
            if "nodes" in self.node_config:
                # Full config received
                self.own_config = self.node_config["nodes"]["secondary"][
                    self.secondary_index
                ]
                self.n_nodes = 1 + len(self.node_config["nodes"]["secondary"])
            else:
                # Partial config found
                self.own_config = self.node_config
            self.own_addr = self.own_config["addr"]
            self.own_comm_port = self.own_config["communication"]["port"]
            self.own_inference_port_in = self.own_config["inference"]["port_in"]
            self.own_inference_port_out = self.own_config["inference"]["port_out"]

            # NOTE: ckpt path may not be present
            if self.ckpt_dir and self.n_nodes and self.chunk_path is None:
                node_chunks_dir = self.ckpt_dir / "chunks" / f"{self.n_nodes}nodes_finisher"
                self.chunk_path = (
                    node_chunks_dir / f"model_secondary{self.secondary_index}.pth"
                )
            elif not self.chunk_path and not self.n_nodes:
                warnings.warn(
                    "Missing info about total n. of nodes, cannot select correct chunk"
                )

            if self.ckpt_dir:
                # Try HF format first, fall back to LitGPT format
                has_hf_config = (self.ckpt_dir / "config.json").exists()
                if has_hf_config:
                    self.model_config, _ = load_from_hf_direct(self.ckpt_dir, config_only=True)
                else:
                    self.model_config, _ = load_from_pt(self.ckpt_dir, config_only=True)
            else:
                self.model_config = None

            self.gpt_serv = GPTServer(
                node_config=self.node_config,
                node_type=self.node_type,
                model_config=self.model_config,
                chunk_path=self.chunk_path,
                model_device=self.torch_device,
                dtype=dtype,
                **kwargs,
            )

        # Here because if the 'device' arg is None, gpt_serv will infer it
        self.torch_device = self.gpt_serv.model_device

    def start(
        self,
        *,
        n_samples: Optional[int] = None,
        tokens_per_sample: Optional[int] = None,
        prompt: Optional[str] = None,
    ):
        """
        Main class entrypoint.
        Start the application; for the starter node, this triggers the initialization of
        other nodes and launches generation.
        For secondary nodes, this starts an infinite loop where the node will wait to be
        initialized and perform inference.

        Args:
            *,
            n_samples (starter only): number of samples to be generated; NOTE: if the
                number is lower than the number of nodes, the generation will not
                benefit from pipelining.
            tokens_per_sample (starter only): number of samples to be *generated*
                (regardless of the prompt length).
            prompt (starter only): prompt (as received from command line) - can be
                FILE:<...>
        """
        if self.node_type == "starter":
            assert n_samples and tokens_per_sample
            assert self.model_config
            # Init. nodes, launch iterations
            if not self.configure_nodes(n_samples=n_samples):
                raise RuntimeError("Unable to initialize network nodes!")

            try:
                out_text, time_gen = self.gpt_serv.launch_starter(
                    n_samples, tokens_per_sample, prompt
                )
                print("-------------------------------------------------")
                print("Produced output:\n")
                for i, smpl in enumerate(out_text):
                    print("-------------------------------------------------")
                    print(f"Sample {i + 1}:")
                    print(smpl, "\n")
                print("-------------------------------------------------")
                print(f"Total generation time: {time_gen[-1][1]}")

                self.stop_nodes()

                return time_gen
            except KeyboardInterrupt:
                self.gpt_serv.shutdown()
                print("Node was stopped!")

        else:
            try:
                cp.engine.block()  # Same as while True: time.sleep(...)
            except KeyboardInterrupt:
                self.gpt_serv.shutdown()
                print("Node was stopped!")


    # ---------------------------------------------------------------------------------

    def configure_nodes(self, n_samples: int) -> int:
        assert self.node_type == "starter"
        
        # 1. Collect Public Keys (Ring Construction)
        print("[INIT] Phase 1: Collecting Keys...")
        starter_pub = self.gpt_serv._serialize_public_key(self.gpt_serv._ecdh_public_key)
        
        # List of tuples: (Role, PubKeyBytes, Addr, Port)
        network_ring = [("starter", starter_pub, None, None)] 

        # Track public keys of each node for key exchange
        # Index: node index, Value: serialized public key bytes
        node_public_keys = {}
        node_public_keys['starter'] = self.gpt_serv._serialize_public_key(self.gpt_serv._ecdh_public_key)

        # Iterate through secondary nodes
        for i, sec_node in enumerate(self.node_config["nodes"]["secondary"]):
            addr = f"http://{sec_node['addr']}:{sec_node['communication']['port']}/key"
            try:
                # GET request to fetch key
                resp = requests.get(addr, timeout=5)
                if resp.status_code == 200:
                    sec_key = resp.content
                    network_ring.append((f"secondary:{i}", sec_key, sec_node['addr'], sec_node['communication']['port']))
                    print(f"  > Collected key for Secondary {i}")
                else:
                    raise ConnectionError(f"Node {i} failed to return key")
            except Exception as e:
                print(f"Failed to reach node {i}: {e}")
                return 0

        # 2. Distribute Keys to Secondaries
        print(f"[INIT] Phase 2: Distributing Keys...")
        total_nodes = len(network_ring)
        
        for i in range(1, total_nodes):
            curr = network_ring[i]
            prev = network_ring[(i - 1) % total_nodes]
            next = network_ring[(i + 1) % total_nodes]
            
            curr_msg = self.init_msg.copy()
            curr_msg["role"] = curr[0]
            curr_msg["model_config"] = self.model_config.asdict()
            curr_msg["n_nodes"] = self.n_nodes
            curr_msg["n_local_layers"] = N_LAYERS_NODES[self.n_nodes][self.model_config.n_layer]["N_LAYERS_SECONDARY"]
            curr_msg["n_samples"] = n_samples
            
            # Set routing info
            curr_msg["prev_node"] = prev
            curr_msg["next_node"] = next_node_config

            # Attach PREVIOUS node's public key (for establishing key_in)
            # Secondary 0 gets starter's key, Secondary 1 gets Secondary 0's key, etc.
            if i == 0:
                prev_pub_key = node_public_keys['starter']
            else:
                prev_pub_key = node_public_keys[i - 1]
            
            curr_msg['ecdh_public_key'] = prev_pub_key

            if not self.model_was_split:
                chunk_path = node_chunks_dir / f"model_secondary{i}.pth"
                curr_msg["params"] = torch.load(chunk_path, device="cpu")

            # 2. SEND SINGLE REQUEST (HANDSHAKE + CONFIG)
            target_addr = sec_node["addr"]
            target_port = sec_node["communication"]["port"]
            addr = f"http://{target_addr}:{target_port}/init"
            
            # Routing
            if i == 1: curr_msg["prev_node"] = self.own_config
            else: curr_msg["prev_node"] = self.node_config["nodes"]["secondary"][i-2]
                
                if response.status_code == 200:
                    # 3. PROCESS RESPONSE - Receive and store this secondary's public key
                    sec_pubkey_bytes = response.content
                    node_public_keys[i] = sec_pubkey_bytes  # Store for next iteration
                    
                    # Only derive key_out for the FIRST secondary (starter's next node)
                    if i == 0:
                        self.gpt_serv._next_node_public_key = self.gpt_serv._deserialize_peer_key(sec_pubkey_bytes)
                        self.gpt_serv._aes_key_out = self.gpt_serv._derive_shared_key(
                            self.gpt_serv._ecdh_private_key, 
                            self.gpt_serv._next_node_public_key
                        )
                        
                        # Always print this for debugging
                        print(f"> Success! key_out (for encryption) derived for secondary {i}.")
                        print(f"[DEBUG STARTER] key_out = {self.gpt_serv._aes_key_out[:16].hex()}...")
                    else:
                        print(f"> Success! Secondary node {i} initialized.")
                    
                    logger_wp.info(f"Secondary node {i} initialized.")
                else:
                    print(f"> Failed with status {response.status_code}")
                    out = 0
            except Exception as e:
                print(f"Initialization failed: {e}")
                out = 0

            if out == 0: return 0

            # Send
            target_url = f"http://{curr[2]}:{curr[3]}/init"
            requests.post(target_url, data=pickle.dumps(curr_msg), timeout=100)

        # 3. Configure Starter
        print("[INIT] Phase 3: Configuring Starter...")
        last_node = network_ring[-1]
        first_node = network_ring[1]
        
        # Starter Key IN = Last Node Public Key
        self.gpt_serv._key_in = self.gpt_serv._derive_shared_key(
            self.gpt_serv._ecdh_private_key, 
            self.gpt_serv._deserialize_peer_key(last_node[1])
        )
        # Starter Key OUT = First Node Public Key
        self.gpt_serv._key_out = self.gpt_serv._derive_shared_key(
            self.gpt_serv._ecdh_private_key, 
            self.gpt_serv._deserialize_peer_key(first_node[1])
        )
        
        print("[INIT] Ring Established.")
        return 1
    def stop_nodes(self) -> int:
        """
        Send a PUT request to all nodes triggering the application interruption.
        """
        out = 1
        for sec_node in self.node_config["nodes"]["secondary"]:
            target_addr = sec_node["addr"]
            target_port = sec_node["communication"]["port"]

            addr = f"http://{target_addr}:{target_port}/stop"
            out *= self._request_to_node("put", addr, "")
        return out

    def _request_to_node(
        self, req_type: str, addr: str, content: Any, max_n_requests: int = 100
    ) -> int:
        """
        Send an HTTP request containing a json-formatted string to a specified
        target node.

        Args:
            req_type: type of HTTP request, can be "post" or "put"
            addr: full address (http(s)://<ip>:<port>) of the target node
            content: python dict containing the information
            max_n_requests: maximum number of requests before failure

        Returns:
            1 if successful
            0 if failed
        """
        if req_type.lower() == "post":
            req_func = requests.post
        elif req_type.lower() == "put":
            req_func = requests.put
        else:
            raise ValueError(f"Unsupported request type '{req_type}'")
        ret = None
        n_ret = 0
        if VERB:
            print(f"Sending {req_type} request to {addr}")
            print(f"Payload: {len(pickle.dumps(content))} Bytes")
        try:
            # Specify timeout
            ret = req_func(
                addr,
                data=pickle.dumps(content),
                timeout=100,
            )

            if ret.status_code == 413:
                raise ConnectionError(f"Max payload for {req_type} was exceeded!")
            logger_wp.debug(
                f"Successful {req_type} request sent to {addr} - code {ret.status_code}"
            )
        except requests.exceptions.Timeout:
            if VERB:
                print("Connection timed out!")
            logger_wp.warning(f"Request timed out!")
            n_ret += 1
        except:
            logger_wp.warning(f"Unable to submit {req_type} request sent to {addr}")
            n_ret += 1
        while (ret is None or ret.status_code != 200) and n_ret < max_n_requests:
            if VERB:
                print(
                    f"Unable to reach node ({addr}) - retrying in 2s ({n_ret}/{max_n_requests})"
                )
            time.sleep(2)
            try:
                ret = req_func(
                    addr,
                    data=pickle.dumps(content),
                    timeout=10000,
                )
                logger_wp.debug(
                    f"Successful {req_type} request sent to {addr} - code {ret.status_code}"
                )
            except requests.exceptions.Timeout:
                if VERB:
                    print("Connection timed out!")
                logger_wp.warning(f"Request timed out!")
            except:
                logger_wp.warning(f"Unable to submit {req_type} request sent to {addr}")
            n_ret += 1

        if ret is not None and ret.status_code == 200:
            return 1
        return 0
