"""
Container Networking

Implementation of Container Network Interface (CNI) plugin, Docker bridge network driver,
and IP address management (IPAM) for container networking.
"""

import ipaddress
import json
import os
import random
import string
import subprocess
from typing import Any, Dict, List, Optional, Set


class CNIPlugin:
    """CNI plugin implementation for container networking"""

    def __init__(self, plugin_name: str, version: str = "0.4.0"):
        self.plugin_name = plugin_name
        self.version = version
        self.network_config = {}

    def add_network(
        self, container_id: str, netns: str, ifname: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        ADD command - Add container to network

        CNI Specification:
        - Create network interface in container namespace
        - Configure IP addresses and routes
        - Return result with IPs and DNS config

        Args:
            container_id: Container ID
            netns: Path to network namespace
            ifname: Interface name inside container
            config: Network configuration

        Returns:
            CNI result with network configuration
        """
        result = {"cniVersion": self.version, "interfaces": [], "ips": [], "dns": {}}

        # Create veth pair
        veth_host = f"veth{container_id[:8]}"
        veth_container = ifname

        # Create veth pair
        subprocess.run(
            [
                "ip",
                "link",
                "add",
                veth_host,
                "type",
                "veth",
                "peer",
                "name",
                veth_container,
            ],
            check=True,
        )

        # Move container end to network namespace
        subprocess.run(
            ["ip", "link", "set", veth_container, "netns", netns], check=True
        )

        # Configure container interface
        self._configure_container_interface(netns, veth_container, config)

        # Configure host interface
        self._configure_host_interface(veth_host, config)

        # Add to result
        result["interfaces"].append(
            {
                "name": veth_container,
                "mac": self._get_mac_address(netns, veth_container),
                "sandbox": netns,
            }
        )

        result["interfaces"].append(
            {"name": veth_host, "mac": self._get_mac_address(None, veth_host)}
        )

        # Configure IP address
        ip_config = config.get("ipam", {})
        if "address" in ip_config:
            result["ips"].append(
                {
                    "version": "4",
                    "interface": 0,
                    "address": ip_config["address"],
                    "gateway": ip_config.get("gateway"),
                }
            )

        # Configure DNS
        dns_config = config.get("dns", {})
        if dns_config:
            result["dns"] = {
                "nameservers": dns_config.get("nameservers", ["8.8.8.8", "8.8.4.4"]),
                "domain": dns_config.get("domain", ""),
                "search": dns_config.get("search", []),
                "options": dns_config.get("options", []),
            }

        return result

    def delete_network(
        self, container_id: str, netns: str, ifname: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        DEL command - Remove container from network

        Args:
            container_id: Container ID
            netns: Path to network namespace
            ifname: Interface name inside container
            config: Network configuration

        Returns:
            Empty result on success
        """
        # Find and delete veth pair
        veth_host = f"veth{container_id[:8]}"

        try:
            # Delete host side veth (peer will be automatically deleted)
            subprocess.run(["ip", "link", "delete", veth_host], check=True)
        except subprocess.CalledProcessError:
            # Interface might already be deleted
            pass

        return {}

    def check_network(
        self, container_id: str, netns: str, ifname: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        CHECK command - Verify container network configuration

        Args:
            container_id: Container ID
            netns: Path to network namespace
            ifname: Interface name inside container
            config: Network configuration

        Returns:
            CNI result with current configuration
        """
        result = {
            "cniVersion": self.version,
            "interfaces": [],
            "ips": [],
            "dns": config.get("dns", {}),
        }

        # Check interface existence and configuration
        try:
            # Get interface information from namespace
            cmd = ["ip", "netns", "exec", netns, "ip", "-j", "addr", "show", ifname]
            output = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if output.stdout:
                iface_info = json.loads(output.stdout)[0]

                result["interfaces"].append(
                    {
                        "name": ifname,
                        "mac": iface_info.get("address", ""),
                        "sandbox": netns,
                    }
                )

                # Extract IP addresses
                for addr_info in iface_info.get("addr_info", []):
                    result["ips"].append(
                        {
                            "version": "4" if addr_info["family"] == "inet" else "6",
                            "interface": 0,
                            "address": f"{addr_info['local']}/{addr_info['prefixlen']}",
                        }
                    )

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            pass

        return result

    def _configure_container_interface(
        self, netns: str, ifname: str, config: Dict[str, Any]
    ):
        """Configure network interface inside container"""
        ip_config = config.get("ipam", {})
        ip_addr = ip_config.get("address")
        gateway = ip_config.get("gateway")

        # Set interface up
        subprocess.run(
            ["ip", "netns", "exec", netns, "ip", "link", "set", "lo", "up"], check=True
        )

        subprocess.run(
            ["ip", "netns", "exec", netns, "ip", "link", "set", ifname, "up"],
            check=True,
        )

        # Assign IP address
        if ip_addr:
            subprocess.run(
                [
                    "ip",
                    "netns",
                    "exec",
                    netns,
                    "ip",
                    "addr",
                    "add",
                    ip_addr,
                    "dev",
                    ifname,
                ],
                check=True,
            )

        # Add default route
        if gateway:
            subprocess.run(
                [
                    "ip",
                    "netns",
                    "exec",
                    netns,
                    "ip",
                    "route",
                    "add",
                    "default",
                    "via",
                    gateway,
                ],
                check=True,
            )

    def _configure_host_interface(self, ifname: str, config: Dict[str, Any]):
        """Configure host side of veth pair"""
        # Attach to bridge if specified
        bridge = config.get("bridge")
        if bridge:
            subprocess.run(["ip", "link", "set", ifname, "master", bridge], check=True)

        # Set interface up
        subprocess.run(["ip", "link", "set", ifname, "up"], check=True)

    def _get_mac_address(self, netns: Optional[str], ifname: str) -> str:
        """Get MAC address of interface"""
        try:
            if netns:
                cmd = ["ip", "netns", "exec", netns, "ip", "-j", "link", "show", ifname]
            else:
                cmd = ["ip", "-j", "link", "show", ifname]

            output = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if output.stdout:
                info = json.loads(output.stdout)[0]
                return info.get("address", "")

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
            pass

        return ""


class DockerBridge:
    """Docker bridge network driver implementation"""

    def __init__(self, bridge_name: str = "docker0"):
        self.bridge_name = bridge_name
        self.subnet = "172.17.0.0/16"
        self.gateway = "172.17.0.1"
        self.ip_allocator = IPAllocator(self.subnet)
        self.mtu = 1500

    def create_bridge(self):
        """Create and configure bridge interface"""
        # Check if bridge already exists
        try:
            subprocess.run(
                ["ip", "link", "show", self.bridge_name],
                capture_output=True,
                check=True,
            )
            print(f"Bridge {self.bridge_name} already exists")
            return
        except subprocess.CalledProcessError:
            pass

        # Create bridge
        subprocess.run(
            ["ip", "link", "add", self.bridge_name, "type", "bridge"], check=True
        )

        # Set MTU
        subprocess.run(
            ["ip", "link", "set", self.bridge_name, "mtu", str(self.mtu)], check=True
        )

        # Set bridge IP
        subprocess.run(
            ["ip", "addr", "add", f"{self.gateway}/16", "dev", self.bridge_name],
            check=True,
        )

        # Bring bridge up
        subprocess.run(["ip", "link", "set", self.bridge_name, "up"], check=True)

        # Enable IP forwarding
        with open("/proc/sys/net/ipv4/ip_forward", "w") as f:
            f.write("1")

        # Configure iptables for NAT
        self._configure_nat()

        # Configure bridge settings
        self._configure_bridge_settings()

    def _configure_nat(self):
        """Configure NAT rules for bridge"""
        # Check if rule already exists
        check_cmd = [
            "iptables",
            "-t",
            "nat",
            "-C",
            "POSTROUTING",
            "-s",
            self.subnet,
            "!",
            "-o",
            self.bridge_name,
            "-j",
            "MASQUERADE",
        ]

        try:
            subprocess.run(check_cmd, capture_output=True, check=True)
            print("NAT rule already exists")
        except subprocess.CalledProcessError:
            # Add MASQUERADE for outgoing traffic
            subprocess.run(
                [
                    "iptables",
                    "-t",
                    "nat",
                    "-A",
                    "POSTROUTING",
                    "-s",
                    self.subnet,
                    "!",
                    "-o",
                    self.bridge_name,
                    "-j",
                    "MASQUERADE",
                ],
                check=True,
            )

        # Allow forwarding - Docker chain
        try:
            subprocess.run(
                ["iptables", "-N", "DOCKER"], capture_output=True, check=True
            )
        except subprocess.CalledProcessError:
            # Chain already exists
            pass

        # Allow forwarding from bridge
        subprocess.run(
            ["iptables", "-A", "FORWARD", "-i", self.bridge_name, "-j", "ACCEPT"],
            check=True,
        )

        # Allow forwarding to bridge
        subprocess.run(
            ["iptables", "-A", "FORWARD", "-o", self.bridge_name, "-j", "ACCEPT"],
            check=True,
        )

    def _configure_bridge_settings(self):
        """Configure bridge kernel parameters"""
        bridge_settings = {
            "bridge-nf-call-iptables": "1",
            "bridge-nf-call-ip6tables": "1",
            "bridge-nf-call-arptables": "1",
        }

        for setting, value in bridge_settings.items():
            path = f"/proc/sys/net/bridge/{setting}"
            if os.path.exists(path):
                with open(path, "w") as f:
                    f.write(value)

    def delete_bridge(self):
        """Delete bridge interface"""
        # Remove iptables rules
        try:
            subprocess.run(
                [
                    "iptables",
                    "-t",
                    "nat",
                    "-D",
                    "POSTROUTING",
                    "-s",
                    self.subnet,
                    "!",
                    "-o",
                    self.bridge_name,
                    "-j",
                    "MASQUERADE",
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            pass

        # Delete bridge
        try:
            subprocess.run(["ip", "link", "delete", self.bridge_name], check=True)
        except subprocess.CalledProcessError:
            pass

    def connect_container(
        self, container_id: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Connect container to bridge network"""
        # Allocate IP address
        ip_addr = self.ip_allocator.allocate()

        # Create network configuration
        network_config = {
            "bridge": self.bridge_name,
            "ipam": {"address": f"{ip_addr}/16", "gateway": self.gateway},
            "dns": {
                "nameservers": config.get("dns_servers", ["8.8.8.8", "8.8.4.4"]),
                "search": config.get("dns_search", []),
            },
        }

        return network_config

    def disconnect_container(self, container_id: str, ip_address: str):
        """Disconnect container from bridge network"""
        # Release IP address
        self.ip_allocator.release(ip_address)


class IPAllocator:
    """IPAM (IP Address Management) for container networks"""

    def __init__(self, subnet: str):
        self.subnet = ipaddress.ip_network(subnet)
        self.allocated: Set[ipaddress.IPv4Address] = set()
        self.released: Set[ipaddress.IPv4Address] = set()

        # Reserve network and broadcast addresses
        self.allocated.add(self.subnet.network_address)
        self.allocated.add(self.subnet.broadcast_address)

        # Reserve gateway (first usable IP)
        gateway = list(self.subnet.hosts())[0]
        self.allocated.add(gateway)

    def allocate(self) -> str:
        """Allocate next available IP address"""
        # Check released IPs first
        if self.released:
            ip = self.released.pop()
            self.allocated.add(ip)
            return str(ip)

        # Find next available
        for ip in self.subnet.hosts():
            if ip not in self.allocated:
                self.allocated.add(ip)
                return str(ip)

        raise Exception("No available IP addresses in subnet")

    def release(self, ip: str):
        """Release IP address back to pool"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj in self.allocated:
                self.allocated.remove(ip_obj)
                self.released.add(ip_obj)
        except ValueError:
            pass

    def is_allocated(self, ip: str) -> bool:
        """Check if IP is allocated"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj in self.allocated
        except ValueError:
            return False

    def get_stats(self) -> Dict[str, int]:
        """Get allocation statistics"""
        total_hosts = int(self.subnet.num_addresses - 2)  # Minus network and broadcast
        allocated = len(self.allocated) - 2  # Minus network and broadcast
        available = total_hosts - allocated + len(self.released)

        return {
            "total": total_hosts,
            "allocated": allocated,
            "available": available,
            "released": len(self.released),
        }


class NetworkManager:
    """High-level network management for containers"""

    def __init__(self):
        self.networks: Dict[str, DockerBridge] = {}
        self.cni_plugin = CNIPlugin("docker-cni")
        self.container_networks: Dict[str, List[str]] = {}

    def create_network(self, name: str, config: Dict[str, Any]) -> str:
        """Create a new network"""
        if name in self.networks:
            raise ValueError(f"Network {name} already exists")

        # Create bridge
        bridge_name = f"br-{name[:12]}"
        subnet = config.get("subnet", self._generate_subnet())

        bridge = DockerBridge(bridge_name)
        bridge.subnet = subnet
        bridge.gateway = str(list(ipaddress.ip_network(subnet).hosts())[0])
        bridge.ip_allocator = IPAllocator(subnet)

        bridge.create_bridge()

        self.networks[name] = bridge

        return name

    def delete_network(self, name: str):
        """Delete a network"""
        if name not in self.networks:
            raise ValueError(f"Network {name} not found")

        # Check if any containers are connected
        for container_id, networks in self.container_networks.items():
            if name in networks:
                raise ValueError(f"Network {name} has active containers")

        # Delete bridge
        bridge = self.networks[name]
        bridge.delete_bridge()

        del self.networks[name]

    def connect_container(
        self,
        container_id: str,
        network_name: str,
        netns: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Connect container to network"""
        if network_name not in self.networks:
            raise ValueError(f"Network {network_name} not found")

        if config is None:
            config = {}

        bridge = self.networks[network_name]

        # Get network configuration
        net_config = bridge.connect_container(container_id, config)

        # Create interface name
        ifname = config.get("interface_name", "eth0")

        # Add to network using CNI
        result = self.cni_plugin.add_network(container_id, netns, ifname, net_config)

        # Track connection
        if container_id not in self.container_networks:
            self.container_networks[container_id] = []
        self.container_networks[container_id].append(network_name)

        return result

    def disconnect_container(
        self, container_id: str, network_name: str, netns: str, ifname: str = "eth0"
    ):
        """Disconnect container from network"""
        if network_name not in self.networks:
            raise ValueError(f"Network {network_name} not found")

        bridge = self.networks[network_name]

        # Get current configuration for cleanup
        config = {"bridge": bridge.bridge_name}

        # Remove from network using CNI
        self.cni_plugin.delete_network(container_id, netns, ifname, config)

        # Update tracking
        if container_id in self.container_networks:
            self.container_networks[container_id].remove(network_name)
            if not self.container_networks[container_id]:
                del self.container_networks[container_id]

    def list_networks(self) -> List[Dict[str, Any]]:
        """List all networks"""
        networks = []

        for name, bridge in self.networks.items():
            stats = bridge.ip_allocator.get_stats()

            networks.append(
                {
                    "name": name,
                    "bridge": bridge.bridge_name,
                    "subnet": bridge.subnet,
                    "gateway": bridge.gateway,
                    "containers": sum(
                        1 for nets in self.container_networks.values() if name in nets
                    ),
                    "ip_stats": stats,
                }
            )

        return networks

    def _generate_subnet(self) -> str:
        """Generate unique subnet for new network"""
        # Start with 172.18.0.0/16 and increment
        base = ipaddress.ip_network("172.18.0.0/16")

        used_subnets = {
            ipaddress.ip_network(bridge.subnet) for bridge in self.networks.values()
        }

        # Find next available /16 subnet
        for i in range(1, 255):
            subnet = ipaddress.ip_network(f"172.{16+i}.0.0/16")
            if subnet not in used_subnets:
                return str(subnet)

        raise ValueError("No available subnets")


# Example usage
def main():
    # Create network manager
    nm = NetworkManager()

    # Create a custom network
    network_name = "my-app-network"
    nm.create_network(network_name, {"subnet": "172.20.0.0/16"})

    # Connect container to network
    container_id = "test-container-001"
    netns = f"/var/run/netns/{container_id}"

    # Create network namespace (in real scenario, container runtime does this)
    os.makedirs("/var/run/netns", exist_ok=True)
    subprocess.run(["ip", "netns", "add", container_id], check=True)

    try:
        # Connect container
        result = nm.connect_container(
            container_id,
            network_name,
            netns,
            {"interface_name": "eth0", "dns_servers": ["1.1.1.1", "1.0.0.1"]},
        )

        print(f"Container connected to network:")
        print(f"  IP: {result['ips'][0]['address']}")
        print(f"  Gateway: {result['ips'][0]['gateway']}")
        print(f"  DNS: {result['dns']['nameservers']}")

        # List networks
        networks = nm.list_networks()
        for net in networks:
            print(f"\nNetwork: {net['name']}")
            print(f"  Bridge: {net['bridge']}")
            print(f"  Subnet: {net['subnet']}")
            print(f"  Connected containers: {net['containers']}")
            print(
                f"  IPs allocated: {net['ip_stats']['allocated']}/{net['ip_stats']['total']}"
            )

    finally:
        # Cleanup
        nm.disconnect_container(container_id, network_name, netns)
        subprocess.run(["ip", "netns", "delete", container_id], check=True)
        nm.delete_network(network_name)


if __name__ == "__main__":
    # Note: This requires root privileges to manipulate network interfaces
    if os.geteuid() != 0:
        print("This script must be run as root")
        exit(1)

    main()
