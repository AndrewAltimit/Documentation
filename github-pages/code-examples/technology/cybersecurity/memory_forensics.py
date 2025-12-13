"""
Advanced Memory Forensics Techniques

Implementation of memory analysis, artifact extraction,
and forensic investigation tools.
"""

import hashlib
import math
import re
import struct
from typing import Dict, List, Optional, Tuple


class MemoryForensics:
    """Advanced memory forensics techniques"""

    def __init__(self, memory_dump):
        self.memory = memory_dump

    def find_processes(self):
        """Find EPROCESS structures in Windows memory"""
        # Simplified EPROCESS signature
        EPROCESS_POOL_TAG = b"\x50\x72\x6f\x63"  # 'Proc'

        processes = []
        offset = 0

        while offset < len(self.memory) - 4:
            if self.memory[offset : offset + 4] == EPROCESS_POOL_TAG:
                # Parse EPROCESS structure
                proc = self.parse_eprocess(offset)
                if proc:
                    processes.append(proc)

            offset += 0x1000  # Page aligned

        return processes

    def parse_eprocess(self, offset):
        """Parse EPROCESS structure"""
        try:
            # Simplified EPROCESS parsing
            eprocess = {
                "offset": offset,
                "pid": struct.unpack(
                    "<I", self.memory[offset + 0x2E0 : offset + 0x2E4]
                )[0],
                "ppid": struct.unpack(
                    "<I", self.memory[offset + 0x2E8 : offset + 0x2EC]
                )[0],
                "image_name": self.memory[offset + 0x450 : offset + 0x460]
                .decode("utf-8", errors="ignore")
                .strip("\x00"),
                "create_time": struct.unpack(
                    "<Q", self.memory[offset + 0x2F0 : offset + 0x2F8]
                )[0],
                "exit_time": struct.unpack(
                    "<Q", self.memory[offset + 0x2F8 : offset + 0x300]
                )[0],
            }

            # Validate structure
            if eprocess["pid"] > 0 and eprocess["pid"] < 65535:
                return eprocess

        except:
            pass

        return None

    def extract_encryption_keys(self):
        """Extract potential encryption keys from memory"""
        keys = []

        # Look for high entropy regions
        window_size = 256

        for offset in range(0, len(self.memory) - window_size, 16):
            window = self.memory[offset : offset + window_size]
            entropy = self.calculate_entropy(window)

            if entropy > 7.5:  # High entropy threshold
                # Check for key schedule patterns
                if self.is_aes_key_schedule(window):
                    keys.append(
                        {
                            "type": "AES",
                            "offset": offset,
                            "key": window[:32],  # 256-bit key
                        }
                    )
                elif self.is_rsa_key(window):
                    keys.append(
                        {
                            "type": "RSA",
                            "offset": offset,
                            "key": self.extract_rsa_key(window),
                        }
                    )

        return keys

    def calculate_entropy(self, data):
        """Calculate Shannon entropy"""
        if not data:
            return 0

        frequency = {}
        for byte in data:
            frequency[byte] = frequency.get(byte, 0) + 1

        entropy = 0
        for count in frequency.values():
            probability = count / len(data)
            entropy -= probability * math.log2(probability)

        return entropy

    def is_aes_key_schedule(self, data):
        """Check if data looks like AES key schedule"""
        # AES key schedule has specific patterns
        # Check for round key expansion artifacts
        if len(data) < 176:  # Minimum for AES-128 expanded key
            return False

        # Check for S-box patterns
        sbox_pattern = 0
        for i in range(len(data) - 1):
            if data[i] ^ data[i + 1] in [
                0x01,
                0x02,
                0x04,
                0x08,
                0x10,
                0x20,
                0x40,
                0x80,
            ]:
                sbox_pattern += 1

        return sbox_pattern > 10

    def is_rsa_key(self, data):
        """Check if data contains RSA key markers"""
        # Look for RSA key headers
        rsa_markers = [
            b"\x30\x82",  # DER sequence
            b"\x02\x01\x00",  # Version
            b"\x00\x00\x00\x07\x73\x73\x68\x2d\x72\x73\x61",  # SSH-RSA
        ]

        for marker in rsa_markers:
            if marker in data:
                return True

        return False

    def extract_rsa_key(self, data):
        """Extract RSA key components"""
        # Simplified RSA key extraction
        key_info = {}

        # Look for DER encoded integers
        offset = 0
        while offset < len(data) - 4:
            if data[offset] == 0x02:  # INTEGER tag
                length = data[offset + 1]
                if length < 128 and offset + 2 + length <= len(data):
                    integer = data[offset + 2 : offset + 2 + length]
                    key_info[f"component_{len(key_info)}"] = integer
                    offset += 2 + length
                else:
                    offset += 1
            else:
                offset += 1

        return key_info

    def find_network_artifacts(self):
        """Find network connections in memory"""
        artifacts = {
            "ipv4_addresses": [],
            "urls": [],
            "email_addresses": [],
            "domains": [],
        }

        # Regular expressions for artifacts
        patterns = {
            "ipv4": rb"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
            "url": rb'https?://[^\s<>"{}|\\^`\[\]]+',
            "email": rb"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "domain": rb"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b",
        }

        # Search memory for patterns
        for pattern_name, pattern in patterns.items():
            matches = re.finditer(pattern, self.memory)
            for match in matches:
                value = match.group(0).decode("utf-8", errors="ignore")
                if pattern_name == "ipv4":
                    # Validate IP address
                    parts = value.split(".")
                    if all(0 <= int(part) <= 255 for part in parts):
                        artifacts["ipv4_addresses"].append(
                            {"address": value, "offset": match.start()}
                        )
                else:
                    artifacts[pattern_name + "s"].append(
                        {"value": value, "offset": match.start()}
                    )

        return artifacts

    def extract_registry_hives(self):
        """Extract Windows registry hives from memory"""
        hives = []

        # Registry hive signatures
        hive_signatures = {
            b"regf": "Registry hive header",
            b"hbin": "Registry hive bin",
        }

        for signature, description in hive_signatures.items():
            offset = 0
            while offset < len(self.memory) - 4:
                if self.memory[offset : offset + 4] == signature:
                    hive = self.parse_registry_hive(offset, signature)
                    if hive:
                        hive["description"] = description
                        hives.append(hive)

                offset += 0x1000  # Page aligned

        return hives

    def parse_registry_hive(self, offset, signature):
        """Parse registry hive structure"""
        try:
            if signature == b"regf":
                # Parse hive header
                hive = {
                    "offset": offset,
                    "signature": signature,
                    "sequence1": struct.unpack(
                        "<I", self.memory[offset + 4 : offset + 8]
                    )[0],
                    "sequence2": struct.unpack(
                        "<I", self.memory[offset + 8 : offset + 12]
                    )[0],
                    "timestamp": struct.unpack(
                        "<Q", self.memory[offset + 12 : offset + 20]
                    )[0],
                    "major_version": struct.unpack(
                        "<I", self.memory[offset + 20 : offset + 24]
                    )[0],
                    "minor_version": struct.unpack(
                        "<I", self.memory[offset + 24 : offset + 28]
                    )[0],
                }

                # Extract hive name
                name_offset = offset + 48
                name_length = 64
                hive["name"] = (
                    self.memory[name_offset : name_offset + name_length]
                    .decode("utf-16le", errors="ignore")
                    .strip("\x00")
                )

                return hive

        except:
            pass

        return None

    def find_code_injection(self):
        """Detect potential code injection in processes"""
        injections = []

        # Common injection patterns
        injection_signatures = [
            b"\x64\xa1\x30\x00\x00\x00",  # mov eax, fs:[30h] - PEB access
            b"\x65\x48\x8b\x04\x25\x60\x00\x00\x00",  # mov rax, gs:[60h] - PEB access (x64)
            b"\x68\x00\x00\x00\x00\xc3",  # push address; ret - ROP gadget
            b"\xff\x25\x00\x00\x00\x00",  # jmp [address] - IAT hook
        ]

        # Search for suspicious patterns
        for sig in injection_signatures:
            offset = 0
            while offset < len(self.memory) - len(sig):
                if self.memory[offset : offset + len(sig)] == sig:
                    # Check if in executable region
                    if self.is_executable_region(offset):
                        injections.append(
                            {
                                "offset": offset,
                                "signature": sig.hex(),
                                "type": self.identify_injection_type(sig),
                            }
                        )

                offset += 1

        return injections

    def is_executable_region(self, offset):
        """Check if offset is in executable memory region"""
        # Simplified check - look for PE header nearby
        search_range = 0x1000
        start = max(0, offset - search_range)
        end = min(len(self.memory), offset + search_range)

        pe_signature = b"PE\x00\x00"
        return pe_signature in self.memory[start:end]

    def identify_injection_type(self, signature):
        """Identify type of code injection"""
        injection_types = {
            b"\x64\xa1\x30\x00\x00\x00": "Process hollowing",
            b"\x65\x48\x8b\x04\x25\x60\x00\x00\x00": "Process hollowing (x64)",
            b"\x68\x00\x00\x00\x00\xc3": "ROP chain",
            b"\xff\x25\x00\x00\x00\x00": "IAT hooking",
        }

        return injection_types.get(signature, "Unknown")

    def extract_strings(self, min_length=4):
        """Extract human-readable strings from memory"""
        strings = []

        # ASCII strings
        ascii_pattern = rb"[\x20-\x7e]{" + str(min_length).encode() + rb",}"
        for match in re.finditer(ascii_pattern, self.memory):
            strings.append(
                {
                    "offset": match.start(),
                    "string": match.group(0).decode("ascii"),
                    "encoding": "ASCII",
                }
            )

        # Unicode strings
        unicode_pattern = rb"(?:[\x20-\x7e]\x00){" + str(min_length).encode() + rb",}"
        for match in re.finditer(unicode_pattern, self.memory):
            try:
                decoded = match.group(0).decode("utf-16le")
                strings.append(
                    {"offset": match.start(), "string": decoded, "encoding": "UTF-16LE"}
                )
            except:
                pass

        return strings

    def find_handles(self):
        """Find open handles in memory"""
        handles = []

        # HANDLE_TABLE_ENTRY structure pattern
        handle_pattern = b"\x00\x00\x00\x00\x00\x00\x00\xf8"

        offset = 0
        while offset < len(self.memory) - 16:
            if self.memory[offset + 8 : offset + 16] == handle_pattern:
                handle = self.parse_handle(offset)
                if handle:
                    handles.append(handle)

            offset += 16  # Handle table entry size

        return handles

    def parse_handle(self, offset):
        """Parse handle table entry"""
        try:
            handle = {
                "offset": offset,
                "object_pointer": struct.unpack("<Q", self.memory[offset : offset + 8])[
                    0
                ],
                "granted_access": struct.unpack(
                    "<I", self.memory[offset + 8 : offset + 12]
                )[0],
                "handle_attributes": struct.unpack(
                    "<I", self.memory[offset + 12 : offset + 16]
                )[0],
            }

            # Validate handle
            if handle["object_pointer"] & 0xFFFF000000000000 == 0xFFFF000000000000:
                return handle

        except:
            pass

        return None

    def timeline_analysis(self):
        """Create timeline of events from memory artifacts"""
        timeline = []

        # Collect timestamps from various sources
        # Process creation times
        processes = self.find_processes()
        for proc in processes:
            if proc["create_time"] > 0:
                timeline.append(
                    {
                        "timestamp": proc["create_time"],
                        "type": "Process Created",
                        "details": f"PID: {proc['pid']}, Name: {proc['image_name']}",
                    }
                )

        # File timestamps from MFT entries
        mft_entries = self.parse_mft_entries()
        for entry in mft_entries:
            timeline.extend(entry["timestamps"])

        # Network connection timestamps
        network_artifacts = self.find_network_artifacts()
        # Add network events to timeline

        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    def parse_mft_entries(self):
        """Parse MFT entries for file activity"""
        mft_entries = []

        # MFT entry signature
        mft_signature = b"FILE"

        offset = 0
        while offset < len(self.memory) - 1024:
            if self.memory[offset : offset + 4] == mft_signature:
                entry = self.parse_single_mft_entry(offset)
                if entry:
                    mft_entries.append(entry)

            offset += 1024  # MFT entry size

        return mft_entries

    def parse_single_mft_entry(self, offset):
        """Parse single MFT entry"""
        try:
            entry = {
                "offset": offset,
                "sequence": struct.unpack("<H", self.memory[offset + 16 : offset + 18])[
                    0
                ],
                "link_count": struct.unpack(
                    "<H", self.memory[offset + 18 : offset + 20]
                )[0],
                "flags": struct.unpack("<H", self.memory[offset + 22 : offset + 24])[0],
                "timestamps": [],
            }

            # Parse attributes
            attr_offset = struct.unpack("<H", self.memory[offset + 20 : offset + 22])[0]
            current_offset = offset + attr_offset

            while current_offset < offset + 1024:
                attr_type = struct.unpack(
                    "<I", self.memory[current_offset : current_offset + 4]
                )[0]

                if attr_type == 0x10:  # $STANDARD_INFORMATION
                    # Extract timestamps
                    si_offset = current_offset + 24
                    entry["timestamps"] = [
                        {
                            "timestamp": struct.unpack(
                                "<Q", self.memory[si_offset : si_offset + 8]
                            )[0],
                            "type": "File Created",
                            "details": f"MFT Entry: {entry['sequence']}",
                        },
                        {
                            "timestamp": struct.unpack(
                                "<Q", self.memory[si_offset + 8 : si_offset + 16]
                            )[0],
                            "type": "File Modified",
                            "details": f"MFT Entry: {entry['sequence']}",
                        },
                    ]
                    break
                elif attr_type == 0xFFFFFFFF:  # End marker
                    break

                # Move to next attribute
                attr_length = struct.unpack(
                    "<I", self.memory[current_offset + 4 : current_offset + 8]
                )[0]
                current_offset += attr_length

            return entry

        except:
            pass

        return None
